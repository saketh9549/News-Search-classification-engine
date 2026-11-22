"""Lightweight CLI runner orchestrating Phases 1–6 of the News Search Engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from src.crawler import (
    DEFAULT_MIN_DOMAIN_DELAY_SECONDS,
    AsyncPoliteCrawler,
    PoliteCrawler,
)
from src.extractor import extract_and_store
from src.indexer import DEFAULT_INDEX_DIR, BM25Index
from src.models import FrontierStatus
from src.poller import DEFAULT_MIN_HOST_DELAY_SECONDS, FeedPoller
from src.storage import get_connection
from src.storage.init_db import DEFAULT_DB_PATH, init_db

logger = logging.getLogger(__name__)


def run_init(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    expand_bbc: bool = False,
    expand_feeds: bool = False,
) -> dict[str, object]:
    """Initialize SQLite schema and seed RSS feeds."""
    resolved = init_db(db_path, expand_bbc=expand_bbc, expand_feeds=expand_feeds)
    conn = get_connection(resolved)
    try:
        feed_count = conn.execute("SELECT COUNT(*) FROM rss_feeds;").fetchone()[0]
    finally:
        conn.close()
    return {"db_path": str(resolved), "rss_feeds": feed_count}


def run_poll(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    delay: float = DEFAULT_MIN_HOST_DELAY_SECONDS,
    expand_bbc: bool = False,
    expand_feeds: bool = False,
) -> dict[str, object]:
    """Poll active RSS/Atom feeds and enqueue discovered URLs."""
    init_db(db_path, expand_bbc=expand_bbc, expand_feeds=expand_feeds)
    conn = get_connection(db_path)
    try:
        results = FeedPoller(conn, min_host_delay=delay).poll_all_feeds()
    finally:
        conn.close()

    status_counts = Counter(r.status for r in results)
    return {
        "feeds_polled": len(results),
        "status_counts": dict(status_counts),
        "items_inserted": sum(r.items_inserted for r in results),
        "items_skipped": sum(r.items_skipped for r in results),
    }


def run_crawl_and_extract(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    limit: int = 100,
    delay: float = DEFAULT_MIN_DOMAIN_DELAY_SECONDS,
    max_retries: int = 3,
    domain: str | None = None,
    interleave_domains: bool = True,
    concurrency: int = 8,
) -> dict[str, object]:
    """Politely crawl pending frontier URLs, extract articles, and persist valid items."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        use_interleave = interleave_domains and domain is None
        distinct_due_domains = 0
        if concurrency > 1 and domain is None:
            row = conn.execute(
                "SELECT COUNT(DISTINCT domain) FROM crawl_frontier WHERE status = 'pending';"
            ).fetchone()
            distinct_due_domains = int(row[0] or 0) if row else 0

        if concurrency > 1 and domain is None and distinct_due_domains > 1:
            async_crawler = AsyncPoliteCrawler(
                conn,
                min_domain_delay=delay,
                max_retries=max_retries,
            )
            crawl_results = asyncio.run(
                async_crawler.crawl_pending(
                    limit=limit,
                    domain=domain,
                    interleave_domains=use_interleave,
                    max_concurrency=concurrency,
                )
            )
        else:
            crawler = PoliteCrawler(conn, min_domain_delay=delay, max_retries=max_retries)
            crawl_results = crawler.crawl_pending(
                limit=limit,
                domain=domain,
                interleave_domains=use_interleave,
            )

        extracted_accepted = 0
        extracted_persisted = 0
        extracted_rejected = 0

        for crawl_res in crawl_results:
            if crawl_res.status != FrontierStatus.FETCHED:
                continue
            ext = extract_and_store(conn, crawl_res)
            if ext.accepted:
                extracted_accepted += 1
                if ext.persisted:
                    extracted_persisted += 1
            else:
                extracted_rejected += 1
    finally:
        conn.close()

    crawl_counts = Counter(r.status.value for r in crawl_results)
    return {
        "crawled_total": len(crawl_results),
        "crawl_status_counts": dict(crawl_counts),
        "extracted_accepted": extracted_accepted,
        "extracted_persisted": extracted_persisted,
        "extracted_rejected": extracted_rejected,
    }


def run_train_classifier(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    min_samples_per_class: int = 4,
) -> dict[str, object]:
    """Train and evaluate the supervised ML category classifier via Stratified K-Fold CV."""
    from src.search.classifier import train_and_save_classifier

    init_db(db_path)
    conn = get_connection(db_path)
    try:
        return train_and_save_classifier(
            conn,
            min_samples_per_class=min_samples_per_class,
        )
    finally:
        conn.close()


def run_index(
    db_path: Path = DEFAULT_DB_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    *,
    rebuild: bool = False,
) -> dict[str, object]:
    """Incrementally index persisted SQLite articles into the BM25 search index."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        index = BM25Index(index_dir)
        sync = index.rebuild_from_db(conn) if rebuild else index.sync_from_db(conn)
    finally:
        conn.close()
    return sync.model_dump()


def run_status(
    db_path: Path = DEFAULT_DB_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> dict[str, object]:
    """Return read-only storage and search-index counts."""
    db_summary: dict[str, object] = {"exists": db_path.is_file(), "db_path": str(db_path)}
    if db_path.is_file():
        conn = get_connection(db_path)
        try:
            feeds_total = conn.execute("SELECT COUNT(*) FROM rss_feeds;").fetchone()[0]
            frontier_rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM crawl_frontier GROUP BY status;"
            ).fetchall()
            articles_total = conn.execute("SELECT COUNT(*) FROM articles;").fetchone()[0]
            db_summary.update(
                {
                    "rss_feeds": feeds_total,
                    "crawl_frontier": {r["status"]: r["cnt"] for r in frontier_rows},
                    "articles": articles_total,
                }
            )
        finally:
            conn.close()

    index = BM25Index(index_dir)
    return {"database": db_summary, "index": index.stats()}


def run_search(
    query: str,
    *,
    index_dir: Path = DEFAULT_INDEX_DIR,
    top_k: int = 10,
    category: str | None = None,
    source: str | None = None,
) -> list[dict[str, object]]:
    """Run a read-only multi-field BM25 query against the search index."""
    index = BM25Index(index_dir)
    hits = index.search(query, top_k=top_k, category=category, source=source)
    return [
        {
            "doc_id": h.doc_id,
            "score": h.score,
            "title": h.title,
            "kicker": h.kicker,
            "source": h.source,
            "author": h.author,
            "published_date": h.published_date.isoformat().replace("+00:00", "Z")
            if h.published_date
            else None,
            "canonical_url": h.canonical_url,
            "matched_fields": [f.value for f in h.matched_fields],
        }
        for h in hits
    ]


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Lightweight CLI runner for News Crawler & Search Engine (Phases 1–6).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init", help="Initialize SQLite database and seed 15 feeds.")
    p_init.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p_init.add_argument("--expand-bbc", action="store_true")
    p_init.add_argument("--expand-feeds", action="store_true")

    p_poll = subparsers.add_parser("poll", help="Poll active RSS feeds into crawl_frontier.")
    p_poll.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p_poll.add_argument("--delay", type=float, default=DEFAULT_MIN_HOST_DELAY_SECONDS)
    p_poll.add_argument("--expand-bbc", action="store_true")
    p_poll.add_argument("--expand-feeds", action="store_true")

    p_crawl = subparsers.add_parser(
        "crawl", help="Politely crawl due frontier URLs and extract valid articles."
    )
    p_crawl.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p_crawl.add_argument("--limit", type=int, default=100)
    p_crawl.add_argument("--delay", type=float, default=DEFAULT_MIN_DOMAIN_DELAY_SECONDS)
    p_crawl.add_argument("--max-retries", type=int, default=3)
    p_crawl.add_argument("--domain", type=str, default=None)
    p_crawl.add_argument("--concurrency", type=int, default=8)
    p_crawl.add_argument(
        "--no-interleave",
        action="store_true",
        help="Crawl strictly in discovered order instead of round-robin across domains.",
    )

    p_train = subparsers.add_parser(
        "train-classifier",
        help="Train & cross-validate the supervised ML news category classifier.",
    )
    p_train.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p_train.add_argument("--min-samples", type=int, default=4)

    p_index = subparsers.add_parser(
        "index", help="Incrementally index persisted SQLite articles into the BM25 index."
    )
    p_index.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p_index.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_index.add_argument("--rebuild", action="store_true")

    p_run = subparsers.add_parser(
        "run", help="Run Phases 1–5 sequentially (init -> poll -> crawl/extract -> index)."
    )
    p_run.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p_run.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_run.add_argument("--limit", type=int, default=100)
    p_run.add_argument("--delay", type=float, default=DEFAULT_MIN_DOMAIN_DELAY_SECONDS)
    p_run.add_argument("--concurrency", type=int, default=8)
    p_run.add_argument("--skip-poll", action="store_true")
    p_run.add_argument("--domain", type=str, default=None)
    p_run.add_argument("--expand-bbc", action="store_true")
    p_run.add_argument("--expand-feeds", action="store_true")
    p_run.add_argument("--rebuild", action="store_true")
    p_run.add_argument(
        "--no-interleave",
        action="store_true",
        help="Crawl strictly in discovered order instead of round-robin across domains.",
    )

    p_daemon = subparsers.add_parser(
        "daemon", help="Continuously poll, politely crawl, and incrementally index new stories."
    )
    p_daemon.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p_daemon.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_daemon.add_argument("--interval-minutes", type=float, default=30.0)
    p_daemon.add_argument("--crawl-limit", type=int, default=80)
    p_daemon.add_argument("--delay", type=float, default=DEFAULT_MIN_DOMAIN_DELAY_SECONDS)
    p_daemon.add_argument("--concurrency", type=int, default=8)
    p_daemon.add_argument("--expand-feeds", action="store_true")
    p_daemon.add_argument("--max-cycles", type=int, default=0)

    p_status = subparsers.add_parser("status", help="Show SQLite and BM25 index counts.")
    p_status.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p_status.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)

    p_search = subparsers.add_parser("search", help="Query the persistent multi-field BM25 index.")
    p_search.add_argument("query", type=str)
    p_search.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_search.add_argument("--k", type=int, default=10)
    p_search.add_argument("--category", type=str, default=None)
    p_search.add_argument("--source", type=str, default=None)

    p_serve = subparsers.add_parser("serve", help="Start the FastAPI search UI and REST API.")
    p_serve.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    for sub in (p_init, p_poll, p_crawl, p_train, p_index, p_run, p_daemon, p_status, p_search, p_serve):
        sub.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Enable INFO logging.",
        )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "init":
        payload: object = run_init(
            args.db, expand_bbc=args.expand_bbc, expand_feeds=args.expand_feeds
        )
    elif args.command == "poll":
        payload = run_poll(
            args.db,
            delay=args.delay,
            expand_bbc=args.expand_bbc,
            expand_feeds=args.expand_feeds,
        )
    elif args.command == "crawl":
        payload = run_crawl_and_extract(
            args.db,
            limit=args.limit,
            delay=args.delay,
            max_retries=args.max_retries,
            domain=args.domain,
            interleave_domains=not args.no_interleave,
            concurrency=args.concurrency,
        )
    elif args.command == "train-classifier":
        payload = run_train_classifier(args.db, min_samples_per_class=args.min_samples)
    elif args.command == "index":
        payload = run_index(args.db, args.index_dir, rebuild=args.rebuild)
    elif args.command == "run":
        init_res = run_init(
            args.db, expand_bbc=args.expand_bbc, expand_feeds=args.expand_feeds
        )
        poll_res = (
            None
            if args.skip_poll
            else run_poll(
                args.db,
                delay=args.delay,
                expand_bbc=args.expand_bbc,
                expand_feeds=args.expand_feeds,
            )
        )
        crawl_res = run_crawl_and_extract(
            args.db,
            limit=args.limit,
            delay=args.delay,
            domain=args.domain,
            interleave_domains=not args.no_interleave,
            concurrency=args.concurrency,
        )
        index_res = run_index(args.db, args.index_dir, rebuild=args.rebuild)
        payload = {
            "init": init_res,
            "poll": poll_res,
            "crawl_and_extract": crawl_res,
            "index": index_res,
        }
    elif args.command == "daemon":
        cycles = 0
        last_payload: dict[str, object] = {}
        while True:
            cycles += 1
            init_res = run_init(args.db, expand_feeds=args.expand_feeds)
            poll_res = run_poll(args.db, expand_feeds=args.expand_feeds)
            crawl_res = run_crawl_and_extract(
                args.db,
                limit=args.crawl_limit,
                delay=args.delay,
                concurrency=args.concurrency,
            )
            index_res = run_index(args.db, args.index_dir)
            last_payload = {
                "cycle": cycles,
                "init": init_res,
                "poll": poll_res,
                "crawl_and_extract": crawl_res,
                "index": index_res,
            }
            logger.info("Completed daemon cycle %d: %s", cycles, last_payload)
            if args.max_cycles > 0 and cycles >= args.max_cycles:
                break
            time.sleep(max(1.0, args.interval_minutes * 60.0))
        payload = last_payload
    elif args.command == "status":
        payload = run_status(args.db, args.index_dir)
    elif args.command == "search":
        payload = run_search(
            args.query,
            index_dir=args.index_dir,
            top_k=args.k,
            category=args.category,
            source=args.source,
        )
    elif args.command == "serve":
        import uvicorn

        from src.search.api import create_app

        uvicorn.run(create_app(args.index_dir), host=args.host, port=args.port)
        return 0
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
