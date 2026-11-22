"""Phase 3 tests: domain politeness, robots.txt compliance, backoff, and crawler state."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import respx

from src.crawler import DEFAULT_USER_AGENT, PoliteCrawler, compute_backoff_seconds
from src.models import FrontierItem, FrontierStatus
from src.storage import enqueue_frontier_items, get_connection, get_due_frontier_items
from src.storage.init_db import init_db


def _seed_frontier(db_path: Path, urls: list[tuple[str, str]]) -> None:
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        items = [
            FrontierItem(feed_id=1, url=url, guid=url, domain=domain)
            for url, domain in urls
        ]
        enqueue_frontier_items(conn, items)
    finally:
        conn.close()


def test_crawler_rejects_unsafe_politeness_delay(tmp_path: Path) -> None:
    db_path = tmp_path / "unsafe.db"
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with pytest.raises(ValueError, match="min_domain_delay"):
            PoliteCrawler(conn, min_domain_delay=1.0)
    finally:
        conn.close()


def test_politeness_delays_same_domain_and_respects_crawl_delay(tmp_path: Path) -> None:
    db_path = tmp_path / "polite.db"
    _seed_frontier(
        db_path,
        [
            ("https://www.thehindu.com/news/story-1.ece", "www.thehindu.com"),
            ("https://www.thehindu.com/news/story-2.ece", "www.thehindu.com"),
            ("https://www.bbc.co.uk/news/story-3", "www.bbc.co.uk"),
        ],
    )

    mono_clock = [100.0]
    sleeps: list[tuple[float, str]] = []

    def fake_mono() -> float:
        return mono_clock[0]

    def fake_sleep(seconds: float) -> None:
        sleeps.append((seconds, "sleep"))
        mono_clock[0] += seconds

    conn = get_connection(db_path)
    try:
        with respx.mock(assert_all_called=True) as mock_http:
            hindu_robots = mock_http.get("https://www.thehindu.com/robots.txt").mock(
                return_value=httpx.Response(
                    200,
                    text="User-agent: *\nCrawl-delay: 8\nAllow: /\n",
                )
            )
            bbc_robots = mock_http.get("https://www.bbc.co.uk/robots.txt").mock(
                return_value=httpx.Response(404)
            )
            s1 = mock_http.get("https://www.thehindu.com/news/story-1.ece").mock(
                return_value=httpx.Response(
                    200,
                    text="<html><body><h1>One</h1></body></html>",
                    headers={"content-type": "text/html"},
                )
            )
            s2 = mock_http.get("https://www.thehindu.com/news/story-2.ece").mock(
                return_value=httpx.Response(
                    200,
                    text="<html><body><h1>Two</h1></body></html>",
                    headers={"content-type": "text/html"},
                )
            )
            s3 = mock_http.get("https://www.bbc.co.uk/news/story-3").mock(
                return_value=httpx.Response(
                    200,
                    text="<html><body><h1>Three</h1></body></html>",
                    headers={"content-type": "text/html"},
                )
            )

            with httpx.Client() as client:
                crawler = PoliteCrawler(
                    conn,
                    client=client,
                    min_domain_delay=5.0,
                    monotonic_fn=fake_mono,
                    sleep_fn=fake_sleep,
                )
                results = crawler.crawl_pending(limit=10)

            # robots.txt fetched once per origin, User-Agent sent everywhere
            assert hindu_robots.call_count == 1
            assert bbc_robots.call_count == 1
            assert s1.calls[0].request.headers["user-agent"] == DEFAULT_USER_AGENT
            assert s2.call_count == 1
            assert s3.call_count == 1

        assert [r.status for r in results] == [
            FrontierStatus.FETCHED,
            FrontierStatus.FETCHED,
            FrontierStatus.FETCHED,
        ]
        # hindu robots -> story-1 waited >= 8s (Crawl-delay: 8);
        # story-1 -> story-2 waited 8s (Crawl-delay: 8);
        # bbc robots -> story-3 waited 5s (min_domain_delay);
        # cross-domain switch to bbc robots did NOT sleep!
        sleep_values = [s[0] for s in sleeps]
        assert sleep_values == [8.0, 8.0, 5.0]
    finally:
        conn.close()


def test_robots_disallow_skips_url_without_fetching(tmp_path: Path) -> None:
    db_path = tmp_path / "robots.db"
    _seed_frontier(
        db_path,
        [
            ("https://example.com/private/secret.html", "example.com"),
            ("https://example.com/news/public.html", "example.com"),
        ],
    )

    mono = [100.0]

    def fake_sleep(sec: float) -> None:
        mono[0] += sec

    conn = get_connection(db_path)
    try:
        with respx.mock(assert_all_called=True) as mock_http:
            mock_http.get("https://example.com/robots.txt").mock(
                return_value=httpx.Response(
                    200,
                    text="User-agent: *\nDisallow: /private/\n",
                )
            )
            public_route = mock_http.get("https://example.com/news/public.html").mock(
                return_value=httpx.Response(
                    200,
                    text="<html><body><article><p>Public story</p></article></body></html>",
                    headers={"content-type": "text/html"},
                )
            )
            # Note: private URL is intentionally NOT mocked; respx would fail if requested!
            with httpx.Client() as client:
                crawler = PoliteCrawler(
                    conn,
                    client=client,
                    monotonic_fn=lambda: mono[0],
                    sleep_fn=fake_sleep,
                )
                results = crawler.crawl_pending(limit=10)

        assert public_route.call_count == 1
        assert results[0].status == FrontierStatus.SKIPPED
        assert results[1].status == FrontierStatus.FETCHED

        rows = conn.execute("SELECT url, status FROM crawl_frontier ORDER BY id;").fetchall()
        assert rows[0]["status"] == "skipped"
        assert rows[1]["status"] == "fetched"
    finally:
        conn.close()


def test_exponential_backoff_on_429_timeout_and_max_retries(tmp_path: Path) -> None:
    db_path = tmp_path / "backoff.db"
    _seed_frontier(
        db_path,
        [
            ("https://example.org/news/rate-limited", "example.org"),
            ("https://example.net/news/not-found", "example.net"),
        ],
    )

    assert compute_backoff_seconds(1, base_seconds=30.0) == 30.0
    assert compute_backoff_seconds(2, base_seconds=30.0) == 60.0
    assert compute_backoff_seconds(3, base_seconds=30.0) == 120.0
    assert compute_backoff_seconds(1, base_seconds=30.0, retry_after_header="90") == 90.0

    current_now = [datetime(2026, 11, 14, 17, 0, 0, tzinfo=timezone.utc)]
    mono = [100.0]

    def fake_sleep(sec: float) -> None:
        mono[0] += sec

    conn = get_connection(db_path)
    try:
        with respx.mock(assert_all_called=True) as mock_http:
            mock_http.get("https://example.org/robots.txt").mock(return_value=httpx.Response(404))
            mock_http.get("https://example.net/robots.txt").mock(return_value=httpx.Response(404))
            mock_http.get("https://example.org/news/rate-limited").mock(
                side_effect=[
                    httpx.Response(429, headers={"Retry-After": "90"}),
                    httpx.ReadTimeout("socket timeout"),
                    httpx.Response(503),
                ]
            )
            mock_http.get("https://example.net/news/not-found").mock(
                return_value=httpx.Response(404)
            )

            with httpx.Client() as client:
                crawler = PoliteCrawler(
                    conn,
                    client=client,
                    max_retries=2,
                    base_backoff_seconds=30.0,
                    now_fn=lambda: current_now[0],
                    monotonic_fn=lambda: mono[0],
                    sleep_fn=fake_sleep,
                )

                # Pass 1: example.org gets 429 (retry 1 -> +90s), example.net gets permanent 404 -> FAILED
                first_pass = crawler.crawl_pending(limit=10)
                assert first_pass[0].status == FrontierStatus.PENDING
                assert first_pass[0].http_status == 429
                assert first_pass[1].status == FrontierStatus.FAILED
                assert first_pass[1].http_status == 404

                # Not due yet at +30s
                assert get_due_frontier_items(conn, now=current_now[0] + timedelta(seconds=30)) == []

                # Due at +90s -> ReadTimeout (retry 2 -> +60s)
                current_now[0] = current_now[0] + timedelta(seconds=90)
                second_pass = crawler.crawl_pending(limit=10)
                assert len(second_pass) == 1
                assert second_pass[0].status == FrontierStatus.PENDING

                # Due at +60s -> 503 exceeds max_retries=2 -> transitions to FAILED
                current_now[0] = current_now[0] + timedelta(seconds=60)
                third_pass = crawler.crawl_pending(limit=10)
                assert len(third_pass) == 1
                assert third_pass[0].status == FrontierStatus.FAILED
                assert third_pass[0].http_status == 503

        row = conn.execute(
            "SELECT status, retry_count, last_http_status FROM crawl_frontier WHERE domain = 'example.org';"
        ).fetchone()
        assert row["status"] == "failed"
        assert row["retry_count"] == 3
        assert row["last_http_status"] == 503
    finally:
        conn.close()


def test_robots_bom_interleave_recovery_and_403_domain_skip(tmp_path: Path) -> None:
    from src.models import normalize_url

    db_path = tmp_path / "interleave.db"
    _seed_frontier(
        db_path,
        [
            ("https://site-a.example/s1", "site-a.example"),
            ("https://site-a.example/s2", "site-a.example"),
            ("https://site-b.example/secret/private", "site-b.example"),
            ("https://site-b.example/public", "site-b.example"),
            ("https://site-c.example/blocked-1", "site-c.example"),
            ("https://site-c.example/blocked-2", "site-c.example"),
        ],
    )

    conn = get_connection(db_path)
    try:
        # Simulate crashed in_progress item on site-a.example/s1
        conn.execute("UPDATE crawl_frontier SET status = 'in_progress' WHERE id = 1;")
        conn.commit()

        mono = [100.0]

        def fake_sleep(sec: float) -> None:
            mono[0] += sec

        with respx.mock(assert_all_called=True) as mock_http:
            mock_http.get("https://site-a.example/robots.txt").mock(
                return_value=httpx.Response(200, text="User-agent: *\nAllow: /\n")
            )
            # BOM-prefixed robots.txt on site-b must STILL parse User-agent: * Disallow: /secret/
            mock_http.get("https://site-b.example/robots.txt").mock(
                return_value=httpx.Response(
                    200,
                    text="\ufeffUser-agent: *\nDisallow: /secret/\n",
                )
            )
            mock_http.get("https://site-c.example/robots.txt").mock(
                return_value=httpx.Response(404)
            )
            mock_http.get("https://site-a.example/s1").mock(
                return_value=httpx.Response(200, text="<html><body>A1</body></html>", headers={"content-type": "text/html"})
            )
            mock_http.get("https://site-a.example/s2").mock(
                return_value=httpx.Response(200, text="<html><body>A2</body></html>", headers={"content-type": "text/html"})
            )
            mock_http.get("https://site-b.example/public").mock(
                return_value=httpx.Response(200, text="<html><body>B1</body></html>", headers={"content-type": "text/html"})
            )
            # blocked-1 returns 403 -> blocked-2 is skipped within this batch (so never requested!)
            blocked_route = mock_http.get("https://site-c.example/blocked-1").mock(
                return_value=httpx.Response(403)
            )

            with httpx.Client() as client:
                crawler = PoliteCrawler(
                    conn,
                    client=client,
                    min_domain_delay=5.0,
                    monotonic_fn=lambda: mono[0],
                    sleep_fn=fake_sleep,
                )
                results = crawler.crawl_pending(limit=10, interleave_domains=True)

        assert blocked_route.call_count == 1
        statuses_by_url = {r.url: r.status for r in results}
        assert statuses_by_url[normalize_url("https://site-a.example/s1")] == FrontierStatus.FETCHED
        assert statuses_by_url[normalize_url("https://site-a.example/s2")] == FrontierStatus.FETCHED
        assert statuses_by_url[normalize_url("https://site-b.example/secret/private")] == FrontierStatus.SKIPPED
        assert statuses_by_url[normalize_url("https://site-b.example/public")] == FrontierStatus.FETCHED
        assert statuses_by_url[normalize_url("https://site-c.example/blocked-1")] == FrontierStatus.FAILED
        assert normalize_url("https://site-c.example/blocked-2") not in statuses_by_url
    finally:
        conn.close()


def test_async_polite_crawler_concurrent_domains_and_per_host_delay(tmp_path: Path) -> None:
    """Verify AsyncPoliteCrawler enforces >= 5s per host while crawling distinct domains concurrently."""
    import asyncio

    from src.crawler import AsyncPoliteCrawler
    from src.models.urls import normalize_url

    db_path = tmp_path / "async_crawl.db"
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        enqueue_frontier_items(
            conn,
            [
                FrontierItem(feed_id=1, url="https://alpha.example/a1", domain="alpha.example"),
                FrontierItem(feed_id=1, url="https://alpha.example/a2", domain="alpha.example"),
                FrontierItem(feed_id=2, url="https://beta.example/b1", domain="beta.example"),
                FrontierItem(feed_id=2, url="https://beta.example/secret/no", domain="beta.example"),
            ],
        )

        mono = [100.0]
        sleeps: list[float] = []

        async def fake_asleep(sec: float) -> None:
            sleeps.append(sec)
            mono[0] += sec

        with respx.mock(assert_all_called=False) as mock_http:
            mock_http.get("https://alpha.example/robots.txt").mock(return_value=httpx.Response(404))
            mock_http.get("https://beta.example/robots.txt").mock(
                return_value=httpx.Response(200, text="User-agent: *\nDisallow: /secret/\n")
            )
            mock_http.get("https://alpha.example/a1").mock(
                return_value=httpx.Response(200, text="<html><body>A1</body></html>", headers={"content-type": "text/html"})
            )
            mock_http.get("https://alpha.example/a2").mock(
                return_value=httpx.Response(200, text="<html><body>A2</body></html>", headers={"content-type": "text/html"})
            )
            mock_http.get("https://beta.example/b1").mock(
                return_value=httpx.Response(200, text="<html><body>B1</body></html>", headers={"content-type": "text/html"})
            )

            async def _run() -> list:
                async with httpx.AsyncClient() as client:
                    crawler = AsyncPoliteCrawler(
                        conn,
                        client=client,
                        min_domain_delay=5.0,
                        monotonic_fn=lambda: mono[0],
                        sleep_fn=fake_asleep,
                    )
                    return await crawler.crawl_pending(limit=10, max_concurrency=4)

            results = asyncio.run(_run())

        by_url = {r.url: r.status for r in results}
        assert by_url[normalize_url("https://alpha.example/a1")] == FrontierStatus.FETCHED
        assert by_url[normalize_url("https://alpha.example/a2")] == FrontierStatus.FETCHED
        assert by_url[normalize_url("https://beta.example/b1")] == FrontierStatus.FETCHED
        assert by_url[normalize_url("https://beta.example/secret/no")] == FrontierStatus.SKIPPED
        # Must enforce >= 5.0s per-host waits (robots->a1, a1->a2, robots->b1)
        assert len(sleeps) >= 2
        assert all(s >= 5.0 - 1e-6 for s in sleeps)
    finally:
        conn.close()
