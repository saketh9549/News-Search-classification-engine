"""Query-driven live news retrieval: conditional RSS poll -> targeted polite crawl -> incremental index."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import perf_counter

import httpx

from src.crawler.crawler import AsyncPoliteCrawler
from src.crawler.robots import RobotsCache
from src.crawler.schemas import DEFAULT_MIN_DOMAIN_DELAY_SECONDS
from src.extractor.extractor import extract_and_store
from src.models.schemas import FrontierItem, FrontierStatus
from src.poller.live_discovery import discover_query_links_from_registered_domains
from src.poller.poller import AsyncFeedPoller
from src.search.service import SearchService
from src.storage.init_db import DEFAULT_DB_PATH, get_connection, init_db
from src.storage.repository import find_matching_pending_frontier_items

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, object]], Awaitable[None]]

# Domains known to block automated bot UAs with permanent HTTP 403
_DEFAULT_BLOCKED_DOMAINS: frozenset[str] = frozenset(
    {"ndtv.com", "www.ndtv.com", "nytimes.com", "www.nytimes.com"}
)


class LiveQueryRetriever:
    """Politely fetches, extracts, classifies, and indexes fresh news matching a user query."""

    def __init__(
        self,
        service: SearchService,
        *,
        db_path: Path | str = DEFAULT_DB_PATH,
        min_domain_delay: float = DEFAULT_MIN_DOMAIN_DELAY_SECONDS,
        poll_max_age_seconds: float = 180.0,
    ) -> None:
        self._service = service
        self._db_path = Path(db_path)
        self._min_domain_delay = float(min_domain_delay)
        self._poll_max_age_seconds = float(poll_max_age_seconds)
        self._robots_cache = RobotsCache()
        self._last_domain_request: dict[str, float] = {}
        self._blocked_domains: set[str] = set(_DEFAULT_BLOCKED_DOMAINS)
        self._op_locks: dict[int, asyncio.Lock] = {}

    def _get_op_lock(self) -> asyncio.Lock:
        loop_id = id(asyncio.get_running_loop())
        lock = self._op_locks.get(loop_id)
        if lock is None:
            lock = asyncio.Lock()
            self._op_locks[loop_id] = lock
        return lock

    async def _emit(
        self,
        cb: ProgressCallback | None,
        stage: str,
        message: str,
        **extra: object,
    ) -> None:
        if cb is not None:
            await cb({"stage": stage, "message": message, **extra})

    async def retrieve_for_query(
        self,
        query: str,
        *,
        max_crawl: int = 8,
        force_poll: bool = False,
        enable_discovery: bool = True,
        client: httpx.AsyncClient | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
        monotonic_fn: Callable[[], float] | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> dict[str, object]:
        """Poll stale RSS feeds, match query links, politely crawl, extract, and index incrementally."""
        started = perf_counter()
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return {
                "query": "",
                "feeds_polled": 0,
                "feeds_updated": 0,
                "rss_new_links": 0,
                "discovery_links": 0,
                "matched_pending": 0,
                "crawled_domains": [],
                "fetched": 0,
                "extracted_persisted": 0,
                "extracted_rejected": 0,
                "newly_indexed": 0,
                "new_doc_ids": [],
                "total_indexed": self._service.index.doc_count,
                "elapsed_ms": 0.0,
            }

        async with self._get_op_lock():
            init_db(self._db_path)
            conn = get_connection(self._db_path)
            try:
                # 1. Conditional async RSS poll across registered feeds (only if stale or forced)
                poll_age = 0.0 if force_poll else self._poll_max_age_seconds
                await self._emit(
                    progress_cb,
                    "polling",
                    "Checking registered news RSS feeds for breaking stories (ETag/304)...",
                )
                poller = AsyncFeedPoller(
                    conn,
                    client=client,
                    min_host_delay=1.5 if sleep_fn is None else 0.0,
                    sleep_fn=sleep_fn,
                    monotonic_fn=monotonic_fn,
                )
                poll_results = await poller.poll_stale_feeds(
                    max_age_seconds=poll_age,
                    max_concurrency=10,
                )
                feeds_updated = sum(1 for r in poll_results if r.status == "updated")
                rss_new_links = sum(r.items_inserted for r in poll_results)

                # 2. Match pending frontier links (RSS headline + slug + summary + freshness)
                await self._emit(
                    progress_cb,
                    "matching",
                    f"Matching uncrawled news links to \"{cleaned_query}\"...",
                    feeds_polled=len(poll_results),
                    rss_new_links=rss_new_links,
                )
                matched_items = find_matching_pending_frontier_items(
                    conn,
                    cleaned_query,
                    limit=max_crawl,
                    max_per_domain=2,
                    exclude_domains=self._blocked_domains,
                )

                # 3. Domain-scoped live RSS discovery fallback strictly on registered outlets
                discovery_links = 0
                if enable_discovery and len(matched_items) < min(3, max_crawl):
                    await self._emit(
                        progress_cb,
                        "discovering",
                        f"Scanning registered news outlets for more links on \"{cleaned_query}\"...",
                        matched_so_far=len(matched_items),
                    )
                    discovered = await discover_query_links_from_registered_domains(
                        conn,
                        cleaned_query,
                        client=client,
                        limit=max_crawl,
                    )
                    discovery_links = len(discovered)
                    if discovered:
                        seen_urls = {item.url for item in matched_items}
                        domain_counts: dict[str, int] = {}
                        for it in matched_items:
                            domain_counts[it.domain] = domain_counts.get(it.domain, 0) + 1
                        for item in discovered:
                            if len(matched_items) >= max_crawl:
                                break
                            if (
                                item.url in seen_urls
                                or item.domain.lower() in self._blocked_domains
                                or domain_counts.get(item.domain, 0) >= 2
                            ):
                                continue
                            seen_urls.add(item.url)
                            domain_counts[item.domain] = domain_counts.get(item.domain, 0) + 1
                            matched_items.append(item)

                targets: list[FrontierItem] = matched_items[:max_crawl]
                crawled_domains = sorted({item.domain for item in targets})
                fetched_count = 0
                new_doc_ids: list[str] = []
                rejected_count = 0
                newly_indexed = 0

                # 4. Concurrent polite crawl strictly obeying robots.txt & >= 5.0s per domain
                if targets:
                    await self._emit(
                        progress_cb,
                        "crawling",
                        f"Politely crawling {len(targets)} matching article(s) across {len(crawled_domains)} outlet(s) (robots.txt + >= 5s/host)...",
                        matched_pending=len(targets),
                        crawled_domains=crawled_domains,
                    )
                    crawler = AsyncPoliteCrawler(
                        conn,
                        client=client,
                        robots_cache=self._robots_cache,
                        min_domain_delay=self._min_domain_delay,
                        sleep_fn=sleep_fn,
                        monotonic_fn=monotonic_fn,
                    )
                    # Share cross-query per-domain timestamps so consecutive queries obey >= 5s/host
                    crawler._last_domain_request = self._last_domain_request
                    crawl_results = await crawler.crawl_items(
                        targets,
                        max_concurrency=8,
                        stop_on_block=True,
                    )

                    for res in crawl_results:
                        if res.http_status in {401, 403, 429}:
                            self._blocked_domains.add(res.domain.lower())

                    fetched_results = [
                        r for r in crawl_results if r.status == FrontierStatus.FETCHED
                    ]
                    fetched_count = len(fetched_results)

                    # 5. Extract clean articles, persist to SQLite, classify & incrementally index
                    if fetched_results:
                        await self._emit(
                            progress_cb,
                            "indexing",
                            f"Extracting, ML-classifying, and indexing {fetched_count} freshly crawled page(s)...",
                            fetched=fetched_count,
                        )
                        for crawl_res in fetched_results:
                            ext = extract_and_store(conn, crawl_res)
                            if ext.accepted and ext.article is not None:
                                if ext.persisted:
                                    new_doc_ids.append(ext.article.doc_id)
                            else:
                                rejected_count += 1

                        sync_res = self._service.index.sync_from_db(conn)
                        newly_indexed = sync_res.newly_indexed
                        if newly_indexed > 0:
                            self._service._ensure_caches()
            finally:
                conn.close()

        elapsed_ms = round((perf_counter() - started) * 1000.0, 2)
        summary: dict[str, object] = {
            "query": cleaned_query,
            "feeds_polled": len(poll_results),
            "feeds_updated": feeds_updated,
            "rss_new_links": rss_new_links,
            "discovery_links": discovery_links,
            "matched_pending": len(targets),
            "crawled_domains": crawled_domains,
            "fetched": fetched_count,
            "extracted_persisted": len(new_doc_ids),
            "extracted_rejected": rejected_count,
            "newly_indexed": newly_indexed,
            "new_doc_ids": new_doc_ids,
            "total_indexed": self._service.index.doc_count,
            "elapsed_ms": elapsed_ms,
        }
        return summary
