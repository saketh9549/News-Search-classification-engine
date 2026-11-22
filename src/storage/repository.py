"""Permanent SQLite repository operations for feeds, frontier queue, and articles."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime

from src.models.schemas import (
    Article,
    Feed,
    FrontierItem,
    FrontierStatus,
    to_iso8601_utc,
    utc_now,
)

_UNSET = object()


def _row_to_feed(row: sqlite3.Row) -> Feed:
    return Feed(
        id=row["id"],
        name=row["name"],
        source=row["source"],
        category=row["category"],
        url=row["url"],
        etag=row["etag"],
        last_modified=row["last_modified"],
        content_hash=row["content_hash"],
        last_polled_at=row["last_polled_at"],
        is_active=bool(row["is_active"]),
    )


def _row_to_frontier_item(row: sqlite3.Row) -> FrontierItem:
    keys = row.keys()
    return FrontierItem(
        id=row["id"],
        feed_id=row["feed_id"],
        url=row["url"],
        guid=row["guid"],
        domain=row["domain"],
        status=FrontierStatus(row["status"]),
        retry_count=row["retry_count"],
        next_attempt_at=row["next_attempt_at"],
        last_http_status=row["last_http_status"],
        discovered_at=row["discovered_at"],
        rss_title=row["rss_title"] if "rss_title" in keys else None,
        rss_summary=row["rss_summary"] if "rss_summary" in keys else None,
        rss_published_at=row["rss_published_at"] if "rss_published_at" in keys else None,
    )


def _row_to_article(row: sqlite3.Row) -> Article:
    return Article(
        doc_id=row["doc_id"],
        feed_id=row["feed_id"],
        url=row["url"],
        canonical_url=row["canonical_url"],
        title=row["title"],
        published_date=row["published_date"],
        author=row["author"],
        kicker=row["kicker"],
        body_text=row["body_text"],
        content_hash=row["content_hash"],
        fetched_at=row["fetched_at"],
    )


def get_active_feeds(conn: sqlite3.Connection) -> list[Feed]:
    """Return all active RSS feeds ordered by ID."""
    rows = conn.execute(
        "SELECT * FROM rss_feeds WHERE is_active = 1 ORDER BY id ASC;"
    ).fetchall()
    return [_row_to_feed(row) for row in rows]


def get_feed_by_id(conn: sqlite3.Connection, feed_id: int) -> Feed | None:
    """Return a single feed by primary key, or None if not found."""
    row = conn.execute(
        "SELECT * FROM rss_feeds WHERE id = ?;",
        (feed_id,),
    ).fetchone()
    return _row_to_feed(row) if row is not None else None


def update_feed_poll_state(
    conn: sqlite3.Connection,
    feed_id: int,
    polled_at: datetime,
    *,
    etag: str | None | object = _UNSET,
    last_modified: str | None | object = _UNSET,
    content_hash: str | None | object = _UNSET,
) -> None:
    """Update feed polling timestamp and optional conditional-caching metadata."""
    polled_iso = to_iso8601_utc(polled_at)
    assignments = ["last_polled_at = ?"]
    params: list[object] = [polled_iso]

    if etag is not _UNSET:
        assignments.append("etag = ?")
        params.append(etag)
    if last_modified is not _UNSET:
        assignments.append("last_modified = ?")
        params.append(last_modified)
    if content_hash is not _UNSET:
        assignments.append("content_hash = ?")
        params.append(content_hash)

    params.append(feed_id)
    sql = f"UPDATE rss_feeds SET {', '.join(assignments)} WHERE id = ?;"
    with conn:
        conn.execute(sql, params)


def enqueue_frontier_items(
    conn: sqlite3.Connection,
    items: Sequence[FrontierItem],
) -> int:
    """Idempotently insert frontier items, ignoring URL or (domain, guid) conflicts."""
    if not items:
        return 0

    rows = [
        (
            item.feed_id,
            item.url,
            item.guid,
            item.domain,
            item.status.value,
            item.retry_count,
            to_iso8601_utc(item.next_attempt_at) if item.next_attempt_at else None,
            item.last_http_status,
            to_iso8601_utc(item.discovered_at),
            item.rss_title,
            item.rss_summary,
            to_iso8601_utc(item.rss_published_at) if item.rss_published_at else None,
        )
        for item in items
    ]
    before = conn.total_changes
    with conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO crawl_frontier (
                feed_id,
                url,
                guid,
                domain,
                status,
                retry_count,
                next_attempt_at,
                last_http_status,
                discovered_at,
                rss_title,
                rss_summary,
                rss_published_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
    inserted = conn.total_changes - before

    # Backfill RSS metadata on previously enqueued pending items that lacked title/summary
    backfill_rows = [
        (
            item.rss_title,
            item.rss_summary,
            to_iso8601_utc(item.rss_published_at) if item.rss_published_at else None,
            item.url,
        )
        for item in items
        if item.rss_title or item.rss_summary or item.rss_published_at
    ]
    if backfill_rows:
        with conn:
            conn.executemany(
                """
                UPDATE crawl_frontier
                SET rss_title = COALESCE(rss_title, ?),
                    rss_summary = COALESCE(rss_summary, ?),
                    rss_published_at = COALESCE(rss_published_at, ?)
                WHERE url = ?
                  AND (rss_title IS NULL OR rss_summary IS NULL OR rss_published_at IS NULL);
                """,
                backfill_rows,
            )

    return inserted


def reset_stale_in_progress_items(conn: sqlite3.Connection) -> int:
    """Recover frontier items left stuck in 'in_progress' from interrupted runs."""
    cursor = conn.execute(
        """
        UPDATE crawl_frontier
        SET status = 'pending',
            next_attempt_at = NULL
        WHERE status = 'in_progress';
        """
    )
    conn.commit()
    return int(cursor.rowcount)


def get_due_frontier_items(
    conn: sqlite3.Connection,
    *,
    now: datetime | None = None,
    limit: int = 50,
    domain: str | None = None,
    interleave_domains: bool = False,
) -> list[FrontierItem]:
    """Fetch pending frontier items whose next_attempt_at is due."""
    now_iso = to_iso8601_utc(now or utc_now())
    clean_domain = domain.strip().lower() if domain else None
    if clean_domain:
        rows = conn.execute(
            """
            SELECT * FROM crawl_frontier
            WHERE status = 'pending'
              AND domain = ?
              AND (next_attempt_at IS NULL OR julianday(next_attempt_at) <= julianday(?))
            ORDER BY julianday(discovered_at) ASC, id ASC
            LIMIT ?;
            """,
            (clean_domain, now_iso, limit),
        ).fetchall()
    elif interleave_domains:
        rows = conn.execute(
            """
            SELECT id, feed_id, url, guid, domain, status, retry_count,
                   next_attempt_at, last_http_status, discovered_at,
                   rss_title, rss_summary, rss_published_at
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY domain
                           ORDER BY _feed_rank ASC,
                                    julianday(COALESCE(rss_published_at, discovered_at)) DESC,
                                    id ASC
                       ) AS _domain_rank
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY feed_id
                               ORDER BY julianday(COALESCE(rss_published_at, discovered_at)) DESC,
                                        id ASC
                           ) AS _feed_rank
                    FROM crawl_frontier
                    WHERE status = 'pending'
                      AND (next_attempt_at IS NULL OR julianday(next_attempt_at) <= julianday(?))
                )
            )
            ORDER BY _domain_rank ASC,
                     julianday(COALESCE(rss_published_at, discovered_at)) DESC,
                     id ASC
            LIMIT ?;
            """,
            (now_iso, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM crawl_frontier
            WHERE status = 'pending'
              AND (next_attempt_at IS NULL OR julianday(next_attempt_at) <= julianday(?))
            ORDER BY julianday(discovered_at) ASC, id ASC
            LIMIT ?;
            """,
            (now_iso, limit),
        ).fetchall()
    return [_row_to_frontier_item(row) for row in rows]


def find_matching_pending_frontier_items(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 8,
    max_per_domain: int = 2,
    now: datetime | None = None,
    exclude_domains: set[str] | frozenset[str] | None = None,
) -> list[FrontierItem]:
    """Select the highest-relevance, freshest uncrawled frontier items matching a search query."""
    import math
    import re
    from urllib.parse import urlsplit

    from src.indexer.preprocessing import preprocess_text
    from src.search.expansion import inspect_and_expand_query

    if not query or not query.strip() or limit <= 0:
        return []

    analysis = inspect_and_expand_query(query)
    concept_groups = [set(g) for g in analysis.concept_groups if g]
    if not concept_groups:
        stems = preprocess_text(query)
        concept_groups = [{s} for s in dict.fromkeys(stems)]
    if not concept_groups:
        return []

    raw_phrase = " ".join(preprocess_text(query))
    now_dt = now or utc_now()
    now_iso = to_iso8601_utc(now_dt)
    excluded = {d.lower() for d in (exclude_domains or ())}

    rows = conn.execute(
        """
        SELECT id, feed_id, url, guid, domain, status, retry_count,
               next_attempt_at, last_http_status, discovered_at,
               rss_title, rss_summary, rss_published_at
        FROM crawl_frontier
        WHERE status = 'pending'
          AND (next_attempt_at IS NULL OR julianday(next_attempt_at) <= julianday(?))
        ORDER BY julianday(COALESCE(rss_published_at, discovered_at)) DESC, id DESC
        LIMIT 3000;
        """,
        (now_iso,),
    ).fetchall()

    slug_split_re = re.compile(r"[^a-zA-Z0-9]+")
    num_concepts = len(concept_groups)
    required_concepts = 1 if num_concepts == 1 else (2 if num_concepts == 2 else math.ceil(num_concepts * 0.6))

    scored: list[tuple[float, int, float, FrontierItem]] = []
    for row in rows:
        dom = str(row["domain"] or "").lower()
        if dom in excluded:
            continue
        item = _row_to_frontier_item(row)
        title_str = item.rss_title or ""
        summary_str = item.rss_summary or ""
        slug_str = slug_split_re.sub(" ", urlsplit(item.url).path or "")

        title_stems = preprocess_text(title_str) if title_str else []
        slug_stems = preprocess_text(slug_str) if slug_str else []
        summary_stems = preprocess_text(summary_str) if summary_str else []

        title_set = set(title_stems)
        slug_set = set(slug_stems)
        summary_set = set(summary_stems)
        combined_set = title_set | slug_set | summary_set
        if not combined_set:
            continue

        matched_concepts = sum(1 for group in concept_groups if group & combined_set)
        if matched_concepts < required_concepts:
            continue

        score = 0.0
        for group in concept_groups:
            if group & title_set:
                score += 3.0
            if group & slug_set:
                score += 2.0
            if group & summary_set:
                score += 1.0

        if raw_phrase and len(raw_phrase.split()) >= 2:
            if f" {raw_phrase} " in f" {' '.join(title_stems)} ":
                score += 4.0
            elif f" {raw_phrase} " in f" {' '.join(slug_stems)} ":
                score += 2.5

        ref_dt = item.rss_published_at or item.discovered_at
        age_hours = max(0.0, (now_dt - ref_dt).total_seconds() / 3600.0)
        freshness_bonus = 1.5 * math.exp(-age_hours / 48.0)
        total_score = score * (matched_concepts / num_concepts) + freshness_bonus
        scored.append((total_score, matched_concepts, ref_dt.timestamp(), item))

    scored.sort(key=lambda entry: (-entry[0], -entry[1], -entry[2], entry[3].id or 0))

    selected: list[FrontierItem] = []
    domain_counts: dict[str, int] = {}
    for _, _, _, item in scored:
        if len(selected) >= limit:
            break
        dom = item.domain.lower()
        if domain_counts.get(dom, 0) >= max_per_domain:
            continue
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        selected.append(item)
    return selected


def update_frontier_status(
    conn: sqlite3.Connection,
    item_id: int,
    *,
    status: FrontierStatus,
    retry_count: int | None = None,
    next_attempt_at: datetime | None | object = _UNSET,
    last_http_status: int | None | object = _UNSET,
) -> None:
    """Update crawl state, retry counter, backoff schedule, and HTTP status for a frontier item."""
    assignments = ["status = ?"]
    params: list[object] = [status.value]

    if retry_count is not None:
        assignments.append("retry_count = ?")
        params.append(retry_count)
    if next_attempt_at is not _UNSET:
        assignments.append("next_attempt_at = ?")
        params.append(
            to_iso8601_utc(next_attempt_at)
            if isinstance(next_attempt_at, datetime)
            else None
        )
    if last_http_status is not _UNSET:
        assignments.append("last_http_status = ?")
        params.append(last_http_status)

    params.append(item_id)
    sql = f"UPDATE crawl_frontier SET {', '.join(assignments)} WHERE id = ?;"
    with conn:
        conn.execute(sql, params)


def save_article(conn: sqlite3.Connection, article: Article) -> bool:
    """Persist a validated Article idempotently; returns True if newly inserted."""
    before = conn.total_changes
    with conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO articles (
                doc_id,
                feed_id,
                url,
                canonical_url,
                title,
                published_date,
                author,
                kicker,
                body_text,
                content_hash,
                fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                article.doc_id,
                article.feed_id,
                article.url,
                article.canonical_url,
                article.title,
                to_iso8601_utc(article.published_date) if article.published_date else None,
                article.author,
                article.kicker,
                article.body_text,
                article.content_hash,
                to_iso8601_utc(article.fetched_at),
            ),
        )
    return (conn.total_changes - before) > 0


def get_article_by_doc_id(conn: sqlite3.Connection, doc_id: str) -> Article | None:
    """Fetch a persisted Article by doc_id, or None if not found."""
    row = conn.execute(
        "SELECT * FROM articles WHERE doc_id = ?;",
        (doc_id,),
    ).fetchone()
    return _row_to_article(row) if row is not None else None
