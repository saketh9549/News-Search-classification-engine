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
                discovered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
    return conn.total_changes - before


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
                   next_attempt_at, last_http_status, discovered_at
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY domain
                           ORDER BY _feed_rank ASC, julianday(discovered_at) ASC, id ASC
                       ) AS _domain_rank
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY feed_id
                               ORDER BY julianday(discovered_at) ASC, id ASC
                           ) AS _feed_rank
                    FROM crawl_frontier
                    WHERE status = 'pending'
                      AND (next_attempt_at IS NULL OR julianday(next_attempt_at) <= julianday(?))
                )
            )
            ORDER BY _domain_rank ASC, julianday(discovered_at) ASC, id ASC
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
