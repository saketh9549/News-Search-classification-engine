"""Phase 2 tests: RSS/Atom feed poller, conditional caching, hashing, dedup, and resilience."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import respx

from src.models import FrontierStatus
from src.poller import DEFAULT_USER_AGENT, FeedPoller, parse_feed_xml
from src.storage import get_connection, get_feed_by_id, update_feed_poll_state
from src.storage.init_db import init_db

SAMPLE_RSS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>BBC News - World</title>
    <link>https://www.bbc.co.uk/news/world</link>
    <item>
      <title>Global Climate Summit Opens</title>
      <link>https://www.bbc.co.uk/news/world-12345?utm_source=rss&amp;utm_medium=feed&amp;ref=home</link>
      <guid isPermaLink="false">bbc-story-12345</guid>
      <pubDate>Sat, 14 Nov 2026 16:00:00 GMT</pubDate>
      <description>World leaders gathered for climate talks.</description>
    </item>
    <item>
      <title>Tech Regulation Update</title>
      <link>https://www.bbc.co.uk/news/tech-67890?id=99&amp;fbclid=xyz</link>
      <pubDate>Sat, 14 Nov 2026 22:00:00 +0530</pubDate>
      <description>New rules take effect next month.</description>
    </item>
  </channel>
</rss>
"""

SAMPLE_ATOM_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Guardian World</title>
  <entry>
    <title>Ocean Treaty Signed</title>
    <link href="https://www.theguardian.com/world/2026/nov/14/ocean-treaty?utm_campaign=atom" />
    <id>https://www.theguardian.com/world/2026/nov/14/ocean-treaty?utm_source=atom</id>
    <updated>2026-11-14T15:45:00Z</updated>
    <summary>Countries ratified the high-seas accord.</summary>
  </entry>
</feed>
"""


def _setup_single_feed_db(tmp_path: Path) -> tuple[Path, int]:
    db_path = tmp_path / "poller.db"
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("UPDATE rss_feeds SET is_active = 0 WHERE id != 1;")
        feed = get_feed_by_id(conn, 1)
        assert feed is not None
        return db_path, feed.id  # type: ignore[return-value]
    finally:
        conn.close()


def test_parse_feed_xml_rss_and_atom_normalizes_urls_and_guids() -> None:
    rss_outcome = parse_feed_xml(SAMPLE_RSS_XML)
    assert rss_outcome.skipped_count == 0
    assert len(rss_outcome.entries) == 2

    first, second = rss_outcome.entries
    assert first.title == "Global Climate Summit Opens"
    assert first.link == "https://www.bbc.co.uk/news/world-12345"
    assert first.domain == "www.bbc.co.uk"
    assert first.guid == "bbc-story-12345"
    assert first.published_at == datetime(2026, 11, 14, 16, 0, tzinfo=timezone.utc)

    # Second entry omits <guid> -> falls back to normalized link (preserving ?id=99, stripping fbclid)
    assert second.link == "https://www.bbc.co.uk/news/tech-67890?id=99"
    assert second.guid == "https://www.bbc.co.uk/news/tech-67890?id=99"
    assert second.published_at == datetime(2026, 11, 14, 16, 30, tzinfo=timezone.utc)

    atom_outcome = parse_feed_xml(SAMPLE_ATOM_XML)
    assert len(atom_outcome.entries) == 1
    atom_entry = atom_outcome.entries[0]
    assert atom_entry.link == "https://www.theguardian.com/world/2026/nov/14/ocean-treaty"
    assert atom_entry.guid == "https://www.theguardian.com/world/2026/nov/14/ocean-treaty"
    assert atom_entry.domain == "www.theguardian.com"


def test_fresh_ingestion_http_200_populates_frontier_and_metadata(tmp_path: Path) -> None:
    db_path, feed_id = _setup_single_feed_db(tmp_path)
    fixed_now = datetime(2026, 11, 14, 17, 0, 0, tzinfo=timezone.utc)

    conn = get_connection(db_path)
    try:
        feed = get_feed_by_id(conn, feed_id)
        assert feed is not None

        with respx.mock(assert_all_called=True) as mock_http:
            route = mock_http.get(feed.url).mock(
                return_value=httpx.Response(
                    200,
                    content=SAMPLE_RSS_XML,
                    headers={
                        "ETag": '"etag-v1"',
                        "Last-Modified": "Sat, 14 Nov 2026 16:50:00 GMT",
                    },
                )
            )

            with httpx.Client() as client:
                poller = FeedPoller(
                    conn,
                    client=client,
                    min_host_delay=0.0,
                    now_fn=lambda: fixed_now,
                )
                results = poller.poll_all_feeds()

            sent_headers = route.calls[0].request.headers
            assert sent_headers["user-agent"] == DEFAULT_USER_AGENT

        assert len(results) == 1
        result = results[0]
        assert result.status == "updated"
        assert result.http_status == 200
        assert result.items_parsed == 2
        assert result.items_inserted == 2
        assert result.items_skipped == 0

        updated_feed = get_feed_by_id(conn, feed_id)
        assert updated_feed is not None
        assert updated_feed.etag == '"etag-v1"'
        assert updated_feed.last_modified == "Sat, 14 Nov 2026 16:50:00 GMT"
        assert updated_feed.content_hash == hashlib.sha256(SAMPLE_RSS_XML).hexdigest()
        assert updated_feed.last_polled_at == fixed_now

        rows = conn.execute("SELECT * FROM crawl_frontier ORDER BY id ASC;").fetchall()
        assert len(rows) == 2
        assert rows[0]["url"] == "https://www.bbc.co.uk/news/world-12345"
        assert rows[0]["guid"] == "bbc-story-12345"
        assert rows[0]["domain"] == "www.bbc.co.uk"
        assert rows[0]["status"] == FrontierStatus.PENDING.value
        assert rows[0]["discovered_at"] == "2026-11-14T17:00:00Z"
        assert rows[1]["guid"] == "https://www.bbc.co.uk/news/tech-67890?id=99"
    finally:
        conn.close()


def test_conditional_caching_http_304_skips_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path, feed_id = _setup_single_feed_db(tmp_path)
    initial_time = datetime(2026, 11, 14, 16, 0, 0, tzinfo=timezone.utc)
    poll_time = datetime(2026, 11, 14, 17, 5, 0, tzinfo=timezone.utc)

    conn = get_connection(db_path)
    try:
        update_feed_poll_state(
            conn,
            feed_id,
            initial_time,
            etag='"cached-etag"',
            last_modified="Sat, 14 Nov 2026 15:55:00 GMT",
            content_hash="existing-sha256",
        )
        feed = get_feed_by_id(conn, feed_id)
        assert feed is not None

        def fail_if_parsed(_: bytes) -> object:
            raise AssertionError("parse_feed_xml must not be called on HTTP 304")

        monkeypatch.setattr("src.poller.poller.parse_feed_xml", fail_if_parsed)

        with respx.mock(assert_all_called=True) as mock_http:
            route = mock_http.get(feed.url).mock(return_value=httpx.Response(304))

            with httpx.Client() as client:
                poller = FeedPoller(
                    conn,
                    client=client,
                    min_host_delay=0.0,
                    now_fn=lambda: poll_time,
                )
                results = poller.poll_all_feeds()

            req_headers = route.calls[0].request.headers
            assert req_headers["if-none-match"] == '"cached-etag"'
            assert req_headers["if-modified-since"] == "Sat, 14 Nov 2026 15:55:00 GMT"

        assert len(results) == 1
        assert results[0].status == "not_modified"
        assert results[0].http_status == 304
        assert results[0].items_inserted == 0

        after_feed = get_feed_by_id(conn, feed_id)
        assert after_feed is not None
        assert after_feed.last_polled_at == poll_time
        assert after_feed.etag == '"cached-etag"'
        assert after_feed.content_hash == "existing-sha256"
        assert conn.execute("SELECT COUNT(*) FROM crawl_frontier;").fetchone()[0] == 0
    finally:
        conn.close()


def test_content_hash_match_on_http_200_skips_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path, feed_id = _setup_single_feed_db(tmp_path)
    initial_time = datetime(2026, 11, 14, 16, 0, 0, tzinfo=timezone.utc)
    poll_time = datetime(2026, 11, 14, 17, 10, 0, tzinfo=timezone.utc)
    xml_hash = hashlib.sha256(SAMPLE_RSS_XML).hexdigest()

    conn = get_connection(db_path)
    try:
        update_feed_poll_state(
            conn,
            feed_id,
            initial_time,
            etag='"v1"',
            last_modified="Sat, 14 Nov 2026 16:00:00 GMT",
            content_hash=xml_hash,
        )
        feed = get_feed_by_id(conn, feed_id)
        assert feed is not None

        def fail_if_parsed(_: bytes) -> object:
            raise AssertionError("parse_feed_xml must not be called when SHA-256 matches")

        monkeypatch.setattr("src.poller.poller.parse_feed_xml", fail_if_parsed)

        with respx.mock(assert_all_called=True) as mock_http:
            mock_http.get(feed.url).mock(
                return_value=httpx.Response(200, content=SAMPLE_RSS_XML)
            )
            with httpx.Client() as client:
                poller = FeedPoller(
                    conn,
                    client=client,
                    min_host_delay=0.0,
                    now_fn=lambda: poll_time,
                )
                results = poller.poll_all_feeds()

        assert len(results) == 1
        assert results[0].status == "unchanged_hash"
        assert results[0].http_status == 200
        assert results[0].items_inserted == 0

        after_feed = get_feed_by_id(conn, feed_id)
        assert after_feed is not None
        assert after_feed.last_polled_at == poll_time
        assert conn.execute("SELECT COUNT(*) FROM crawl_frontier;").fetchone()[0] == 0
    finally:
        conn.close()


def test_deduplication_ignores_seen_urls_and_domain_guids(tmp_path: Path) -> None:
    db_path, feed_id = _setup_single_feed_db(tmp_path)
    conn = get_connection(db_path)
    try:
        feed = get_feed_by_id(conn, feed_id)
        assert feed is not None

        updated_rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>BBC News - World</title>
    <item>
      <title>Global Climate Summit Opens (Republished Link)</title>
      <link>https://www.bbc.co.uk/news/world-12345-republished</link>
      <guid isPermaLink="false">bbc-story-12345</guid>
    </item>
    <item>
      <title>Tech Regulation Update</title>
      <link>https://www.bbc.co.uk/news/tech-67890?id=99&amp;utm_source=refresh</link>
    </item>
    <item>
      <title>Brand New Breaking Story</title>
      <link>https://www.bbc.co.uk/news/world-99999</link>
      <guid isPermaLink="false">bbc-story-99999</guid>
    </item>
  </channel>
</rss>
"""

        with respx.mock(assert_all_called=True) as mock_http:
            mock_http.get(feed.url).mock(
                side_effect=[
                    httpx.Response(200, content=SAMPLE_RSS_XML, headers={"ETag": '"v1"'}),
                    httpx.Response(200, content=updated_rss_xml, headers={"ETag": '"v2"'}),
                ]
            )
            with httpx.Client() as client:
                poller = FeedPoller(conn, client=client, min_host_delay=0.0)
                first_run = poller.poll_all_feeds()
                second_run = poller.poll_all_feeds()

        assert first_run[0].items_inserted == 2
        assert second_run[0].items_parsed == 3
        assert second_run[0].items_inserted == 1

        total = conn.execute("SELECT COUNT(*) FROM crawl_frontier;").fetchone()[0]
        assert total == 3
    finally:
        conn.close()


def test_fault_tolerance_handles_http_errors_timeouts_and_malformed_xml(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "faults.db"
    init_db(db_path)

    partial_bad_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Mixed Feed</title>
    <item>
      <title>Missing Link Entry</title>
    </item>
    <item>
      <title>Valid Entry Survives</title>
      <link>https://www.thehindu.com/news/national/valid-story.ece</link>
      <guid>hindu-valid-1</guid>
    </item>
  </channel>
</rss>
"""

    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("UPDATE rss_feeds SET is_active = CASE WHEN id <= 6 THEN 1 ELSE 0 END;")
        feeds = [get_feed_by_id(conn, idx) for idx in range(1, 7)]
        assert all(f is not None for f in feeds)

        with respx.mock(assert_all_called=True) as mock_http:
            mock_http.get(feeds[0].url).mock(return_value=httpx.Response(500))  # type: ignore[union-attr]
            mock_http.get(feeds[1].url).mock(return_value=httpx.Response(429))  # type: ignore[union-attr]
            mock_http.get(feeds[2].url).mock(side_effect=httpx.ReadTimeout("timed out"))  # type: ignore[union-attr]
            mock_http.get(feeds[3].url).mock(side_effect=httpx.ConnectError("dns failed"))  # type: ignore[union-attr]
            mock_http.get(feeds[4].url).mock(
                return_value=httpx.Response(200, content=b"<rss><channel><item><title>Broken")
            )  # type: ignore[union-attr]
            mock_http.get(feeds[5].url).mock(
                return_value=httpx.Response(200, content=partial_bad_xml)
            )  # type: ignore[union-attr]

            with httpx.Client() as client:
                poller = FeedPoller(conn, client=client, min_host_delay=0.0)
                results = poller.poll_all_feeds()

        by_id = {r.feed_id: r for r in results}
        assert len(results) == 6
        assert by_id[1].status == "error" and by_id[1].http_status == 500
        assert by_id[2].status == "error" and by_id[2].http_status == 429
        assert by_id[3].status == "error" and "timeout" in (by_id[3].error or "").lower()
        assert by_id[4].status == "error" and "network" in (by_id[4].error or "").lower()
        assert by_id[5].status == "error" and "malformed" in (by_id[5].error or "").lower()
        assert by_id[6].status == "updated"
        assert by_id[6].items_parsed == 1
        assert by_id[6].items_inserted == 1
        assert by_id[6].items_skipped == 1

        rows = conn.execute("SELECT url, guid FROM crawl_frontier;").fetchall()
        assert len(rows) == 1
        assert rows[0]["url"] == "https://www.thehindu.com/news/national/valid-story.ece"
        assert rows[0]["guid"] == "hindu-valid-1"
    finally:
        conn.close()
