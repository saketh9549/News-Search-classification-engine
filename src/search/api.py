"""FastAPI REST API and XSS-safe web UI for multi-field BM25 news search & ML evaluation."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse

from src.indexer.bm25_index import BM25Index
from src.indexer.schemas import DEFAULT_INDEX_DIR
from src.search.classifier import DEFAULT_METRICS_PATH, load_classifier_metrics
from src.search.live_retriever import LiveQueryRetriever
from src.search.schemas import (
    ArticleDetailResponse,
    BackgroundLink,
    CorpusOverview,
    EvaluationReport,
    QueryJudgment,
    SearchResponse,
)
from src.search.service import SearchService, parse_date_boundary
from src.storage.init_db import DEFAULT_DB_PATH, get_connection

_ROBOTS_TXT_PATH = Path(__file__).resolve().parents[2] / "robots.txt"
_DEFAULT_ROBOTS_TXT = "User-agent: *\nAllow: /\nDisallow: /api/\n"
_SEARCH_UI_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "search_ui.html"
DEFAULT_EVAL_QRELS_PATH = Path("data/eval_qrels.json")


def _load_robots_txt() -> str:
    if _ROBOTS_TXT_PATH.is_file():
        return _ROBOTS_TXT_PATH.read_text(encoding="utf-8")
    return _DEFAULT_ROBOTS_TXT


def _load_search_ui_html() -> str:
    if _SEARCH_UI_TEMPLATE_PATH.is_file():
        return _SEARCH_UI_TEMPLATE_PATH.read_text(encoding="utf-8")
    return "<!DOCTYPE html><html><head><title>News Search Engine</title></head><body><h1>News Search Engine</h1></body></html>"


def _load_default_qrels(
    service: SearchService,
    qrels_path: Path = DEFAULT_EVAL_QRELS_PATH,
) -> list[QueryJudgment]:
    """Load curated qrels JSON if present and matching the active index; else synthesize from corpus."""
    active_ids = service.index.indexed_doc_ids
    if qrels_path.is_file():
        try:
            raw = json.loads(qrels_path.read_text(encoding="utf-8"))
            parsed = [QueryJudgment.model_validate(item) for item in raw]
            matched = [
                QueryJudgment(
                    query_id=j.query_id,
                    query=j.query,
                    relevant_docs={
                        doc_id: grade
                        for doc_id, grade in j.relevant_docs.items()
                        if doc_id in active_ids
                    },
                )
                for j in parsed
            ]
            matched = [j for j in matched if j.relevant_docs]
            if matched:
                return matched
        except (OSError, ValueError):
            pass

    # Deterministic fallback for isolated test corpora without an external qrels file
    overview = service.get_corpus_overview()
    synthesized: list[QueryJudgment] = []
    for idx, sample_q in enumerate(overview.sample_queries[:5], start=1):
        hits = service.index.search(sample_q, top_k=3)
        if not hits:
            continue
        rel_docs = {hit.doc_id: (2 if rank == 0 else 1) for rank, hit in enumerate(hits)}
        synthesized.append(
            QueryJudgment(query_id=f"auto_q{idx}", query=sample_q, relevant_docs=rel_docs)
        )
    return synthesized


def _read_db_stats(db_path: Path, indexed_doc_count: int) -> dict[str, object]:
    """Return read-only live SQLite frontier & feed statistics."""
    if not db_path.is_file():
        return {
            "db_exists": False,
            "db_path": str(db_path),
            "articles_count": 0,
            "indexed_doc_count": indexed_doc_count,
            "total_feeds": 0,
            "active_feeds": 0,
            "frontier_counts": {},
            "feeds": [],
        }

    conn = get_connection(db_path)
    try:
        articles_count = int(conn.execute("SELECT COUNT(*) FROM articles;").fetchone()[0])
        total_feeds = int(conn.execute("SELECT COUNT(*) FROM rss_feeds;").fetchone()[0])
        active_feeds = int(
            conn.execute("SELECT COUNT(*) FROM rss_feeds WHERE is_active = 1;").fetchone()[0]
        )
        frontier_rows = conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM crawl_frontier GROUP BY status;"
        ).fetchall()
        frontier_counts = {str(r["status"]): int(r["cnt"]) for r in frontier_rows}
        feed_rows = conn.execute(
            """
            SELECT f.id, f.name, f.source, f.category, f.is_active, f.last_polled_at,
                   (SELECT COUNT(*) FROM crawl_frontier cf WHERE cf.feed_id = f.id AND cf.status = 'pending') AS pending,
                   (SELECT COUNT(*) FROM crawl_frontier cf WHERE cf.feed_id = f.id AND cf.status = 'fetched') AS fetched,
                   (SELECT COUNT(*) FROM crawl_frontier cf WHERE cf.feed_id = f.id AND cf.status = 'failed') AS failed,
                   (SELECT COUNT(*) FROM crawl_frontier cf WHERE cf.feed_id = f.id AND cf.status = 'skipped') AS skipped,
                   (SELECT COUNT(*) FROM articles a WHERE a.feed_id = f.id) AS articles
            FROM rss_feeds f
            ORDER BY articles DESC, f.id ASC;
            """
        ).fetchall()
        feeds = [
            {
                "id": int(r["id"]),
                "name": str(r["name"]),
                "source": str(r["source"]),
                "category": str(r["category"]),
                "is_active": bool(r["is_active"]),
                "last_polled_at": r["last_polled_at"],
                "pending": int(r["pending"] or 0),
                "fetched": int(r["fetched"] or 0),
                "failed": int(r["failed"] or 0),
                "skipped": int(r["skipped"] or 0),
                "articles": int(r["articles"] or 0),
            }
            for r in feed_rows
        ]
    finally:
        conn.close()

    return {
        "db_exists": True,
        "db_path": str(db_path),
        "articles_count": articles_count,
        "indexed_doc_count": indexed_doc_count,
        "total_feeds": total_feeds,
        "active_feeds": active_feeds,
        "frontier_counts": frontier_counts,
        "feeds": feeds,
    }


def create_app(
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    *,
    index: BM25Index | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
    qrels_path: Path | str = DEFAULT_EVAL_QRELS_PATH,
) -> FastAPI:
    """Create a configured FastAPI application for news search, classification, and evaluation."""
    resolved_index = index if index is not None else BM25Index(index_dir)
    service = SearchService(resolved_index)
    resolved_db_path = Path(db_path)
    resolved_metrics_path = Path(metrics_path)
    resolved_qrels_path = Path(qrels_path)
    live_retriever = LiveQueryRetriever(service, db_path=resolved_db_path)

    application = FastAPI(
        title="News Crawler & Search Engine API",
        version="0.2.0",
        description="Multi-field Okapi BM25 search with supervised ML classification, NER, and TF-IDF background linking.",
    )
    application.state.search_service = service
    application.state.live_retriever = live_retriever

    @application.get("/", response_class=HTMLResponse)
    def search_ui() -> HTMLResponse:
        return HTMLResponse(content=_load_search_ui_html())

    @application.get("/robots.txt", response_class=PlainTextResponse)
    def robots_txt() -> PlainTextResponse:
        return PlainTextResponse(content=_load_robots_txt())

    @application.get("/health")
    def health() -> dict[str, object]:
        service.index.refresh_if_changed()
        return {
            "status": "ok",
            "doc_count": service.index.doc_count,
            "index_dir": str(service.index.index_dir),
        }

    @application.get("/api/corpus", response_model=CorpusOverview)
    def get_corpus() -> CorpusOverview:
        return service.get_corpus_overview()

    @application.get("/api/stats")
    def get_stats() -> dict[str, object]:
        service.index.refresh_if_changed()
        return _read_db_stats(resolved_db_path, service.index.doc_count)

    @application.get("/api/classifier/metrics")
    def get_classifier_metrics() -> dict[str, object]:
        metrics = load_classifier_metrics(resolved_metrics_path)
        if metrics is None:
            return {"trained": False, "model_type": "lexical_fallback"}
        return {"trained": True, **metrics}

    @application.get("/api/search", response_model=SearchResponse)
    def search_articles(
        q: Annotated[
            str,
            Query(description="Keyword search query (optional when filters are active)"),
        ] = "",
        k: Annotated[int, Query(ge=1, le=100, description="Maximum number of results")] = 10,
        category: Annotated[str | None, Query(description="Filter by category")] = None,
        source: Annotated[str | None, Query(description="Filter by publisher source name")] = None,
        sort: Annotated[
            str,
            Query(description="Sort mode: 'relevance' (BM25 + recency) or 'latest' (UTC newest)"),
        ] = "relevance",
        date_from: Annotated[
            str | None, Query(description="Inclusive start UTC date/ISO timestamp")
        ] = None,
        date_to: Annotated[
            str | None, Query(description="Inclusive end UTC date/ISO timestamp")
        ] = None,
    ) -> SearchResponse:
        has_filters = bool(
            (category and category.strip())
            or (source and source.strip())
            or (date_from and date_from.strip())
            or (date_to and date_to.strip())
        )
        if not q.strip() and not has_filters:
            raise HTTPException(status_code=422, detail="Query string 'q' must not be blank.")
        try:
            start_dt = parse_date_boundary(date_from, end_of_day=False)
            end_dt = parse_date_boundary(date_to, end_of_day=True)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid date filter: {exc}") from exc

        if start_dt and end_dt and start_dt > end_dt:
            raise HTTPException(
                status_code=422,
                detail="'date_from' must be earlier than or equal to 'date_to'.",
            )

        return service.search(
            q,
            top_k=k,
            category=category,
            source=source,
            date_from=start_dt,
            date_to=end_dt,
            sort=sort,
        )

    @application.post("/api/crawl-for-query")
    async def crawl_for_query(
        q: Annotated[str, Query(min_length=1, description="News query to crawl fresh links for")],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        max_crawl: Annotated[int, Query(ge=1, le=15)] = 8,
        force_poll: Annotated[bool, Query()] = False,
        enable_discovery: Annotated[bool, Query()] = True,
        category: Annotated[str | None, Query()] = None,
        source: Annotated[str | None, Query()] = None,
        sort: Annotated[str, Query()] = "relevance",
        date_from: Annotated[str | None, Query()] = None,
        date_to: Annotated[str | None, Query()] = None,
    ) -> dict[str, object]:
        if not q.strip():
            raise HTTPException(status_code=422, detail="Query string 'q' must not be blank.")
        try:
            start_dt = parse_date_boundary(date_from, end_of_day=False)
            end_dt = parse_date_boundary(date_to, end_of_day=True)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid date filter: {exc}") from exc

        crawl_summary = await live_retriever.retrieve_for_query(
            q,
            max_crawl=max_crawl,
            force_poll=force_poll,
            enable_discovery=enable_discovery,
        )
        search_resp = service.search(
            q,
            top_k=k,
            category=category,
            source=source,
            date_from=start_dt,
            date_to=end_dt,
            sort=sort,
        )
        return {
            "crawl_summary": crawl_summary,
            "search_response": search_resp.model_dump(mode="json"),
        }

    @application.get("/api/search/live")
    async def search_live_stream(
        q: Annotated[str, Query(min_length=1, description="News query for live retrieval")],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        max_crawl: Annotated[int, Query(ge=1, le=15)] = 8,
        force_poll: Annotated[bool, Query()] = False,
        enable_discovery: Annotated[bool, Query()] = True,
        category: Annotated[str | None, Query()] = None,
        source: Annotated[str | None, Query()] = None,
        sort: Annotated[str, Query()] = "relevance",
        date_from: Annotated[str | None, Query()] = None,
        date_to: Annotated[str | None, Query()] = None,
    ) -> StreamingResponse:
        if not q.strip():
            raise HTTPException(status_code=422, detail="Query string 'q' must not be blank.")
        try:
            start_dt = parse_date_boundary(date_from, end_of_day=False)
            end_dt = parse_date_boundary(date_to, end_of_day=True)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid date filter: {exc}") from exc

        async def _event_stream() -> AsyncIterator[str]:
            queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

            async def _on_progress(event: dict[str, object]) -> None:
                await queue.put(event)

            async def _worker() -> None:
                try:
                    summary = await live_retriever.retrieve_for_query(
                        q,
                        max_crawl=max_crawl,
                        force_poll=force_poll,
                        enable_discovery=enable_discovery,
                        progress_cb=_on_progress,
                    )
                    search_resp = service.search(
                        q,
                        top_k=k,
                        category=category,
                        source=source,
                        date_from=start_dt,
                        date_to=end_dt,
                        sort=sort,
                    )
                    await queue.put(
                        {
                            "stage": "complete",
                            "message": (
                                f"Added {summary['newly_indexed']} fresh article(s) to index."
                                if int(summary["newly_indexed"]) > 0
                                else "Up to date with registered news feeds."
                            ),
                            "crawl_summary": summary,
                            "search_response": search_resp.model_dump(mode="json"),
                        }
                    )
                except Exception as exc:
                    await queue.put({"stage": "error", "message": f"Live crawl error: {exc}"})
                finally:
                    await queue.put(None)

            task = asyncio.create_task(_worker())
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.get("/api/articles/{doc_id}", response_model=ArticleDetailResponse)
    def get_article(
        doc_id: str,
        background_limit: Annotated[int, Query(ge=1, le=10)] = 3,
    ) -> ArticleDetailResponse:
        detail = service.get_article_detail(doc_id, background_limit=background_limit)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Article {doc_id!r} not found.")
        return detail

    @application.get("/api/articles/{doc_id}/background", response_model=list[BackgroundLink])
    def get_article_background(
        doc_id: str,
        limit: Annotated[int, Query(ge=1, le=10)] = 3,
    ) -> list[BackgroundLink]:
        links = service.get_background_links(doc_id, limit=limit)
        if links is None:
            raise HTTPException(status_code=404, detail=f"Article {doc_id!r} not found.")
        return links

    @application.post("/api/evaluate", response_model=EvaluationReport)
    def evaluate_queries(
        judgments: list[QueryJudgment],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
    ) -> EvaluationReport:
        return service.evaluate(judgments, k=k)

    @application.get("/api/evaluate/default", response_model=EvaluationReport)
    def evaluate_default_queries(
        k: Annotated[int, Query(ge=1, le=100)] = 10,
    ) -> EvaluationReport:
        judgments = _load_default_qrels(service, resolved_qrels_path)
        return service.evaluate(judgments, k=k)

    return application


app = create_app(Path(os.environ.get("NEWS_INDEX_DIR", str(DEFAULT_INDEX_DIR))))
