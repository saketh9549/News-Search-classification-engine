"""Idempotent SQLite schema initialization and RSS feed seeding."""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from src.models.schemas import Feed, to_iso8601_utc

DEFAULT_DB_PATH = Path("data/news_search.db")

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS rss_feeds (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        name            TEXT NOT NULL,
        source          TEXT NOT NULL,
        category        TEXT NOT NULL,
        url             TEXT NOT NULL UNIQUE,
        etag            TEXT,
        last_modified   TEXT,
        content_hash    TEXT,
        last_polled_at  TEXT,
        is_active       INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS crawl_frontier (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        feed_id          INTEGER NOT NULL REFERENCES rss_feeds(id),
        url              TEXT NOT NULL UNIQUE,
        guid             TEXT,
        domain           TEXT NOT NULL,
        status           TEXT NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending', 'in_progress', 'fetched', 'failed', 'skipped')),
        retry_count      INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
        next_attempt_at  TEXT,
        last_http_status INTEGER,
        discovered_at    TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS articles (
        doc_id          TEXT PRIMARY KEY,
        feed_id         INTEGER NOT NULL REFERENCES rss_feeds(id),
        url             TEXT NOT NULL,
        canonical_url   TEXT NOT NULL UNIQUE,
        title           TEXT NOT NULL CHECK (length(trim(title)) > 0),
        published_date  TEXT,
        author          TEXT,
        kicker          TEXT,
        body_text       TEXT NOT NULL CHECK (length(trim(body_text)) > 0),
        content_hash    TEXT NOT NULL,
        fetched_at      TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_crawl_frontier_status_next_attempt
        ON crawl_frontier (status, next_attempt_at);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_crawl_frontier_domain
        ON crawl_frontier (domain);
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_crawl_frontier_domain_guid
        ON crawl_frontier (domain, guid)
        WHERE guid IS NOT NULL;
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_crawl_frontier_feed_id
        ON crawl_frontier (feed_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_articles_feed_id
        ON articles (feed_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_articles_published_date
        ON articles (published_date);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_articles_kicker
        ON articles (kicker);
    """,
)

SEED_FEEDS: tuple[Feed, ...] = (
    Feed(
        name="BBC News - World",
        source="BBC News",
        category="world",
        url="https://feeds.bbci.co.uk/news/world/rss.xml",
    ),
    Feed(
        name="BBC News - Technology",
        source="BBC News",
        category="technology",
        url="https://feeds.bbci.co.uk/news/technology/rss.xml",
    ),
    Feed(
        name="BBC News - India",
        source="BBC News",
        category="india",
        url="https://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
    ),
    Feed(
        name="The Hindu - National",
        source="The Hindu",
        category="national",
        url="https://www.thehindu.com/news/national/feeder/default.rss",
    ),
    Feed(
        name="The Hindu - Business",
        source="The Hindu",
        category="business",
        url="https://www.thehindu.com/business/feeder/default.rss",
    ),
    Feed(
        name="The Hindu - Sport",
        source="The Hindu",
        category="sport",
        url="https://www.thehindu.com/sport/feeder/default.rss",
    ),
    Feed(
        name="Times of India - Top Stories",
        source="Times of India",
        category="top_stories",
        url="https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    ),
    Feed(
        name="NDTV - Top Stories",
        source="NDTV",
        category="top_stories",
        url="https://feeds.feedburner.com/ndtvnews-top-stories",
    ),
    Feed(
        name="The Indian Express - India",
        source="The Indian Express",
        category="india",
        url="https://indianexpress.com/section/india/feed/",
    ),
    Feed(
        name="Hindustan Times - India News",
        source="Hindustan Times",
        category="india",
        url="https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
    ),
    Feed(
        name="India Today - Home",
        source="India Today",
        category="top_stories",
        url="https://www.indiatoday.in/rss/home",
    ),
    Feed(
        name="The Economic Times - Top Stories",
        source="The Economic Times",
        category="business",
        url="https://economictimes.indiatimes.com/rssfeedstopstories.cms",
    ),
    Feed(
        name="The Guardian - World",
        source="The Guardian",
        category="world",
        url="https://www.theguardian.com/world/rss",
    ),
    Feed(
        name="Al Jazeera - All News",
        source="Al Jazeera",
        category="world",
        url="https://www.aljazeera.com/xml/rss/all.xml",
    ),
    Feed(
        name="The New York Times - Technology",
        source="The New York Times",
        category="technology",
        url="https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    ),
)

_INSERT_FEED_SQL = """
INSERT OR IGNORE INTO rss_feeds (
    name,
    source,
    category,
    url,
    etag,
    last_modified,
    content_hash,
    last_polled_at,
    is_active
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys and Row factory enabled."""
    target = str(db_path)
    if target != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    if target != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def _ensure_frontier_guid_column(conn: sqlite3.Connection) -> None:
    """Add optional guid column to pre-existing crawl_frontier tables without data loss."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(crawl_frontier);").fetchall()
    }
    if columns and "guid" not in columns:
        conn.execute("ALTER TABLE crawl_frontier ADD COLUMN guid TEXT;")


def create_schema(conn: sqlite3.Connection) -> None:
    """Idempotently create tables and indexes if they do not exist."""
    with conn:
        for statement in SCHEMA_STATEMENTS[:2]:
            conn.execute(statement)
        _ensure_frontier_guid_column(conn)
        for statement in SCHEMA_STATEMENTS[2:]:
            conn.execute(statement)


def seed_feeds(conn: sqlite3.Connection, feeds: Sequence[Feed] = SEED_FEEDS) -> int:
    """Insert initial RSS feeds idempotently, returning the number of new rows inserted."""
    before = conn.total_changes
    rows = [
        (
            feed.name,
            feed.source,
            feed.category,
            feed.url,
            feed.etag,
            feed.last_modified,
            feed.content_hash,
            to_iso8601_utc(feed.last_polled_at) if feed.last_polled_at else None,
            1 if feed.is_active else 0,
        )
        for feed in feeds
    ]
    with conn:
        conn.executemany(_INSERT_FEED_SQL, rows)
    return conn.total_changes - before


ADDITIONAL_BBC_FEEDS: tuple[Feed, ...] = (
    Feed(
        name="BBC News - Business",
        source="BBC News",
        category="business",
        url="https://feeds.bbci.co.uk/news/business/rss.xml",
    ),
    Feed(
        name="BBC News - Science & Environment",
        source="BBC News",
        category="science",
        url="https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    ),
    Feed(
        name="BBC News - Health",
        source="BBC News",
        category="health",
        url="https://feeds.bbci.co.uk/news/health/rss.xml",
    ),
    Feed(
        name="BBC Sport",
        source="BBC News",
        category="sport",
        url="https://feeds.bbci.co.uk/sport/rss.xml",
    ),
    Feed(
        name="BBC News - Entertainment & Arts",
        source="BBC News",
        category="entertainment",
        url="https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
    ),
    Feed(
        name="BBC News - UK",
        source="BBC News",
        category="uk",
        url="https://feeds.bbci.co.uk/news/uk/rss.xml",
    ),
    Feed(
        name="BBC News - Asia",
        source="BBC News",
        category="asia",
        url="https://feeds.bbci.co.uk/news/world/asia/rss.xml",
    ),
)


def seed_additional_bbc_feeds(conn: sqlite3.Connection) -> int:
    """Idempotently seed additional BBC News section feeds without altering default 15 feeds."""
    before = conn.total_changes
    with conn:
        for feed in ADDITIONAL_BBC_FEEDS:
            conn.execute(
                """
                INSERT OR IGNORE INTO rss_feeds (name, source, category, url, is_active)
                VALUES (?, ?, ?, ?, ?);
                """,
                (feed.name, feed.source, feed.category, feed.url, int(feed.is_active)),
            )
    return conn.total_changes - before


def init_db(db_path: Path | str = DEFAULT_DB_PATH, *, expand_bbc: bool = False) -> Path:
    """Initialize SQLite storage and seed the 15 RSS feeds idempotently."""
    conn = get_connection(db_path)
    try:
        create_schema(conn)
        seed_feeds(conn)
        if expand_bbc:
            seed_additional_bbc_feeds(conn)
    finally:
        conn.close()
    return Path(db_path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for initializing the SQLite database."""
    parser = argparse.ArgumentParser(description="Initialize News Search Engine SQLite database.")
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database file (default: data/news_search.db).",
    )
    args = parser.parse_args(argv)
    db_file = init_db(args.db)
    print(f"Database initialized at {db_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
