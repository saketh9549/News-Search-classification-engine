"""Phase 1 tests: package layout, Pydantic UTC schemas, idempotent DB init, and constraints."""

from __future__ import annotations

import importlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models import Article, Feed, FrontierItem, FrontierStatus, to_iso8601_utc
from src.storage.init_db import (
    SCHEMA_STATEMENTS,
    SEED_FEEDS,
    create_schema,
    get_connection,
    init_db,
    main,
    seed_feeds,
)

EXPECTED_FEED_URLS = {
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
    "https://www.thehindu.com/news/national/feeder/default.rss",
    "https://www.thehindu.com/business/feeder/default.rss",
    "https://www.thehindu.com/sport/feeder/default.rss",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://feeds.feedburner.com/ndtvnews-top-stories",
    "https://indianexpress.com/section/india/feed/",
    "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
    "https://www.indiatoday.in/rss/home",
    "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
    "https://www.theguardian.com/world/rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
}


def test_required_packages_exist_and_import() -> None:
    modules = (
        "src.poller",
        "src.crawler",
        "src.extractor",
        "src.indexer",
        "src.search",
        "src.models",
        "src.storage",
    )
    for mod in modules:
        pkg_dir = Path(*mod.split("."))
        assert (pkg_dir / "__init__.py").is_file()
        assert importlib.import_module(mod) is not None


def test_schema_columns_match_specification(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"
    init_db(db_path)

    conn = get_connection(db_path)
    try:
        def cols(table: str) -> list[str]:
            rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
            return [row["name"] for row in rows]

        assert cols("rss_feeds") == [
            "id",
            "name",
            "source",
            "category",
            "url",
            "etag",
            "last_modified",
            "content_hash",
            "last_polled_at",
            "is_active",
        ]
        assert cols("crawl_frontier") == [
            "id",
            "feed_id",
            "url",
            "guid",
            "domain",
            "status",
            "retry_count",
            "next_attempt_at",
            "last_http_status",
            "discovered_at",
        ]
        assert cols("articles") == [
            "doc_id",
            "feed_id",
            "url",
            "canonical_url",
            "title",
            "published_date",
            "author",
            "kicker",
            "body_text",
            "content_hash",
            "fetched_at",
        ]
    finally:
        conn.close()


def test_15_feeds_seeded_and_valid_models(tmp_path: Path) -> None:
    db_path = tmp_path / "seeded.db"
    init_db(db_path)

    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM rss_feeds ORDER BY id;").fetchall()
        assert len(rows) == 15
        urls = {row["url"] for row in rows}
        assert urls == EXPECTED_FEED_URLS

        for row in rows:
            feed = Feed(
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
            assert feed.is_active is True
    finally:
        conn.close()


def test_init_runs_twice_with_no_changes(tmp_path: Path) -> None:
    db_path = tmp_path / "idempotent.db"
    init_db(db_path)

    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE rss_feeds
                SET etag = '"abc-123"',
                    last_modified = 'Sat, 14 Nov 2026 16:00:00 GMT',
                    content_hash = 'deadbeef',
                    last_polled_at = '2026-11-14T16:00:00Z'
                WHERE id = 1;
                """
            )
        before_rows = [dict(r) for r in conn.execute("SELECT * FROM rss_feeds ORDER BY id;").fetchall()]
        changes_before = conn.total_changes
        create_schema(conn)
        inserted = seed_feeds(conn)
        changes_after = conn.total_changes
    finally:
        conn.close()

    assert inserted == 0
    assert changes_after - changes_before == 0

    init_db(db_path)

    conn = get_connection(db_path)
    try:
        after_rows = [dict(r) for r in conn.execute("SELECT * FROM rss_feeds ORDER BY id;").fetchall()]
        assert len(after_rows) == 15
        assert after_rows == before_rows
        assert after_rows[0]["etag"] == '"abc-123"'
        assert after_rows[0]["content_hash"] == "deadbeef"
    finally:
        conn.close()


def test_unique_and_integrity_constraints_enforced(tmp_path: Path) -> None:
    db_path = tmp_path / "constraints.db"
    init_db(db_path)

    conn = get_connection(db_path)
    try:
        feed_row = conn.execute("SELECT id, url FROM rss_feeds ORDER BY id LIMIT 1;").fetchone()
        feed_id = feed_row["id"]
        feed_url = feed_row["url"]

        # rss_feeds.url must be unique
        with pytest.raises(sqlite3.IntegrityError):
            with conn:
                conn.execute(
                    """
                    INSERT INTO rss_feeds (name, source, category, url, is_active)
                    VALUES ('Dup Feed', 'BBC', 'world', ?, 1);
                    """,
                    (feed_url,),
                )

        # crawl_frontier.url must be unique
        article_url = "https://www.bbc.com/news/articles/example-1"
        with conn:
            conn.execute(
                """
                INSERT INTO crawl_frontier (feed_id, url, guid, domain, status, retry_count, discovered_at)
                VALUES (?, ?, 'bbc-guid-1', 'www.bbc.com', 'pending', 0, '2026-11-14T16:30:00Z');
                """,
                (feed_id, article_url),
            )

        with pytest.raises(sqlite3.IntegrityError):
            with conn:
                conn.execute(
                    """
                    INSERT INTO crawl_frontier (feed_id, url, guid, domain, status, retry_count, discovered_at)
                    VALUES (?, ?, 'bbc-guid-other', 'www.bbc.com', 'pending', 0, '2026-11-14T16:31:00Z');
                    """,
                    (feed_id, article_url),
                )

        # (domain, guid) must be unique when guid is present, even with a different URL
        with pytest.raises(sqlite3.IntegrityError):
            with conn:
                conn.execute(
                    """
                    INSERT INTO crawl_frontier (feed_id, url, guid, domain, status, retry_count, discovered_at)
                    VALUES (?, 'https://www.bbc.com/news/articles/example-1?utm=rss', 'bbc-guid-1', 'www.bbc.com', 'pending', 0, '2026-11-14T16:31:00Z');
                    """,
                    (feed_id,),
                )

        # NULL guid falls back to canonical url uniqueness on the same domain
        with conn:
            conn.execute(
                """
                INSERT INTO crawl_frontier (feed_id, url, guid, domain, status, retry_count, discovered_at)
                VALUES (?, 'https://www.bbc.com/news/articles/no-guid-1', NULL, 'www.bbc.com', 'pending', 0, '2026-11-14T16:31:00Z');
                """,
                (feed_id,),
            )
            conn.execute(
                """
                INSERT INTO crawl_frontier (feed_id, url, guid, domain, status, retry_count, discovered_at)
                VALUES (?, 'https://www.bbc.com/news/articles/no-guid-2', NULL, 'www.bbc.com', 'pending', 0, '2026-11-14T16:32:00Z');
                """,
                (feed_id,),
            )

        # articles.canonical_url and articles.doc_id must be unique
        canonical = "https://www.bbc.com/news/articles/example-1"
        with conn:
            conn.execute(
                """
                INSERT INTO articles (
                    doc_id, feed_id, url, canonical_url, title,
                    published_date, author, kicker, body_text, content_hash, fetched_at
                ) VALUES (
                    'doc-1', ?, ?, ?, 'Headline One',
                    '2026-11-14T16:00:00Z', 'Reporter', 'world',
                    'Clean article body text.', 'hash-1', '2026-11-14T16:35:00Z'
                );
                """,
                (feed_id, article_url, canonical),
            )

        with pytest.raises(sqlite3.IntegrityError):
            with conn:
                conn.execute(
                    """
                    INSERT INTO articles (
                        doc_id, feed_id, url, canonical_url, title,
                        published_date, author, kicker, body_text, content_hash, fetched_at
                    ) VALUES (
                        'doc-2', ?, 'https://www.bbc.com/news/articles/example-1?utm=rss', ?,
                        'Headline Duplicate Canonical',
                        '2026-11-14T16:00:00Z', 'Reporter', 'world',
                        'Different tracking URL same canonical.', 'hash-2', '2026-11-14T16:36:00Z'
                    );
                    """,
                    (feed_id, canonical),
                )

        with pytest.raises(sqlite3.IntegrityError):
            with conn:
                conn.execute(
                    """
                    INSERT INTO articles (
                        doc_id, feed_id, url, canonical_url, title,
                        published_date, author, kicker, body_text, content_hash, fetched_at
                    ) VALUES (
                        'doc-1', ?, 'https://www.bbc.com/news/articles/example-2',
                        'https://www.bbc.com/news/articles/example-2',
                        'Headline Duplicate Doc ID',
                        '2026-11-14T16:00:00Z', 'Reporter', 'world',
                        'Different canonical same doc_id.', 'hash-3', '2026-11-14T16:37:00Z'
                    );
                    """,
                    (feed_id,),
                )

        # Foreign key enforcement
        with pytest.raises(sqlite3.IntegrityError):
            with conn:
                conn.execute(
                    """
                    INSERT INTO crawl_frontier (feed_id, url, domain, status, retry_count, discovered_at)
                    VALUES (999999, 'https://example.com/orphan', 'example.com', 'pending', 0, '2026-11-14T16:30:00Z');
                    """
                )
    finally:
        conn.close()


def test_pydantic_models_utc_standardization_and_validation() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    local_dt = datetime(2026, 11, 14, 22, 0, 0, tzinfo=ist)

    feed = Feed(
        name="Example Feed",
        source="Example",
        category="world",
        url="https://example.com/rss.xml",
        last_polled_at=local_dt,
    )
    assert feed.last_polled_at is not None
    assert feed.last_polled_at.tzinfo == timezone.utc
    assert to_iso8601_utc(feed.last_polled_at) == "2026-11-14T16:30:00Z"
    assert feed.model_dump(mode="json")["last_polled_at"] == "2026-11-14T16:30:00Z"

    item = FrontierItem(
        feed_id=1,
        url="https://WWW.TheHindu.com/news/national/article123.ece",
        guid="   ",
        domain="WWW.TheHindu.com",
        discovered_at="2026-11-14T22:00:00+05:30",  # type: ignore[arg-type]
    )
    assert item.domain == "www.thehindu.com"
    assert item.guid is None
    assert item.status == FrontierStatus.PENDING
    assert item.discovered_at.tzinfo == timezone.utc
    assert to_iso8601_utc(item.discovered_at) == "2026-11-14T16:30:00Z"

    article = Article(
        doc_id="abc123",
        feed_id=1,
        url="https://example.com/story?ref=rss",
        canonical_url="https://example.com/story",
        title="  Major Policy Announcement  ",
        published_date="2026-11-14T16:30:00Z",  # type: ignore[arg-type]
        author="   ",
        kicker="world",
        body_text="Clean story body text.",
        content_hash="sha256-digest",
    )
    assert article.title == "Major Policy Announcement"
    assert article.author is None
    assert article.published_date is not None
    assert article.published_date.tzinfo == timezone.utc

    # Naive timestamps must be rejected
    with pytest.raises(ValidationError):
        Feed(
            name="Bad Timestamp",
            source="Example",
            category="world",
            url="https://example.com/rss.xml",
            last_polled_at=datetime(2026, 11, 14, 16, 30, 0),
        )

    # Non-HTTP URLs must be rejected
    with pytest.raises(ValidationError):
        FrontierItem(
            feed_id=1,
            url="ftp://example.com/file.xml",
            domain="example.com",
        )

    # Blank headline or body must be rejected
    with pytest.raises(ValidationError):
        Article(
            doc_id="bad-1",
            feed_id=1,
            url="https://example.com/bad",
            canonical_url="https://example.com/bad",
            title="   ",
            body_text="Some body.",
            content_hash="hash",
        )

    assert len(SEED_FEEDS) == 15


def test_schema_statements_non_destructive_and_cli_works(tmp_path: Path) -> None:
    for stmt in SCHEMA_STATEMENTS:
        upper = stmt.upper()
        assert "DROP " not in upper
        assert "DELETE " not in upper
        assert (
            "CREATE TABLE IF NOT EXISTS" in upper
            or "CREATE INDEX IF NOT EXISTS" in upper
            or "CREATE UNIQUE INDEX IF NOT EXISTS" in upper
        )

    db_path = tmp_path / "cli.db"
    assert main(["--db", str(db_path)]) == 0
    assert db_path.is_file()

    conn = get_connection(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM rss_feeds;").fetchone()[0]
        assert count == 15
    finally:
        conn.close()


def test_preexisting_frontier_without_guid_is_migrated_safely(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE rss_feeds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    category TEXT NOT NULL,
                    url TEXT NOT NULL UNIQUE,
                    etag TEXT,
                    last_modified TEXT,
                    content_hash TEXT,
                    last_polled_at TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE crawl_frontier (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feed_id INTEGER NOT NULL REFERENCES rss_feeds(id),
                    url TEXT NOT NULL UNIQUE,
                    domain TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT,
                    last_http_status INTEGER,
                    discovered_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "INSERT INTO rss_feeds (name, source, category, url) VALUES ('F', 'S', 'world', 'https://feeds.bbci.co.uk/news/world/rss.xml');"
            )
            conn.execute(
                "INSERT INTO crawl_frontier (feed_id, url, domain, discovered_at) VALUES (1, 'https://www.bbc.com/news/1', 'www.bbc.com', '2026-11-14T16:00:00Z');"
            )
    finally:
        conn.close()

    init_db(db_path)

    conn = get_connection(db_path)
    try:
        cols = [row["name"] for row in conn.execute("PRAGMA table_info(crawl_frontier);").fetchall()]
        assert "guid" in cols
        row = conn.execute("SELECT url, guid FROM crawl_frontier WHERE id = 1;").fetchone()
        assert row["url"] == "https://www.bbc.com/news/1"
        assert row["guid"] is None
        indexes = {
            row["name"]
            for row in conn.execute("PRAGMA index_list(crawl_frontier);").fetchall()
        }
        assert "idx_crawl_frontier_domain_guid" in indexes
    finally:
        conn.close()
