"""Synchronous RSS/Atom feed poller with conditional HTTP caching and deduplication."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

import httpx

from src.models.schemas import Feed, FrontierItem, FrontierStatus, utc_now
from src.models.urls import extract_domain
from src.poller.parser import FeedParseError, compute_content_hash, parse_feed_xml
from src.poller.schemas import (
    DEFAULT_MIN_HOST_DELAY_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    FeedPollResult,
)
from src.storage.repository import (
    enqueue_frontier_items,
    get_active_feeds,
    update_feed_poll_state,
)

logger = logging.getLogger(__name__)


class FeedPoller:
    """Polls registered RSS/Atom feeds with conditional HTTP headers and SHA-256 dedup."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        client: httpx.Client | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        min_host_delay: float = DEFAULT_MIN_HOST_DELAY_SECONDS,
        now_fn: Callable[[], datetime] = utc_now,
        monotonic_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._conn = conn
        self._external_client = client
        self._user_agent = user_agent
        self._timeout_seconds = timeout_seconds
        self._min_host_delay = max(0.0, min_host_delay)
        self._now_fn = now_fn
        self._monotonic_fn = monotonic_fn or time.monotonic
        self._sleep_fn = sleep_fn or time.sleep
        self._locks_guard = threading.Lock()
        self._host_locks: dict[str, threading.Lock] = {}
        self._last_host_request: dict[str, float] = {}

    def _get_host_lock(self, host: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._host_locks.get(host)
            if lock is None:
                lock = threading.Lock()
                self._host_locks[host] = lock
            return lock

    def _wait_for_host_politeness(self, host: str) -> None:
        if self._min_host_delay <= 0.0:
            return
        with self._get_host_lock(host):
            last_ts = self._last_host_request.get(host)
            if last_ts is not None:
                elapsed = self._monotonic_fn() - last_ts
                remaining = self._min_host_delay - elapsed
                if remaining > 0:
                    self._sleep_fn(remaining)
            self._last_host_request[host] = self._monotonic_fn()

    def _build_request_headers(self, feed: Feed) -> dict[str, str]:
        headers = {
            "User-Agent": self._user_agent,
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
        }
        if feed.etag:
            headers["If-None-Match"] = feed.etag
        if feed.last_modified:
            headers["If-Modified-Since"] = feed.last_modified
        return headers

    def poll_feed(
        self,
        feed: Feed,
        *,
        client: httpx.Client | None = None,
    ) -> FeedPollResult:
        """Fetch a single feed conditionally, detect changes, and enqueue new frontier items."""
        if feed.id is None:
            raise ValueError("Feed must have a database id before polling")

        active_client = client or self._external_client
        if active_client is None:
            with httpx.Client(
                timeout=self._timeout_seconds,
                follow_redirects=True,
            ) as managed_client:
                return self.poll_feed(feed, client=managed_client)

        host = extract_domain(feed.url)
        self._wait_for_host_politeness(host)

        now = self._now_fn()
        headers = self._build_request_headers(feed)

        try:
            response = active_client.get(feed.url, headers=headers)
        except httpx.TimeoutException as exc:
            logger.warning("Timeout polling feed %s (%s): %s", feed.id, feed.url, exc)
            return FeedPollResult(
                feed_id=feed.id,
                feed_url=feed.url,
                status="error",
                error=f"Request timeout: {exc}",
            )
        except httpx.HTTPError as exc:
            logger.warning("Network error polling feed %s (%s): %s", feed.id, feed.url, exc)
            return FeedPollResult(
                feed_id=feed.id,
                feed_url=feed.url,
                status="error",
                error=f"Network error: {exc}",
            )

        status_code = response.status_code

        if status_code == 304:
            update_feed_poll_state(self._conn, feed.id, now)
            return FeedPollResult(
                feed_id=feed.id,
                feed_url=feed.url,
                status="not_modified",
                http_status=304,
            )

        if status_code == 429 or status_code >= 400:
            logger.warning(
                "Skipping feed %s (%s) due to HTTP %s",
                feed.id,
                feed.url,
                status_code,
            )
            return FeedPollResult(
                feed_id=feed.id,
                feed_url=feed.url,
                status="error",
                http_status=status_code,
                error=f"HTTP {status_code}",
            )

        raw_bytes = response.content
        content_hash = compute_content_hash(raw_bytes)

        if feed.content_hash is not None and content_hash == feed.content_hash:
            update_feed_poll_state(self._conn, feed.id, now)
            return FeedPollResult(
                feed_id=feed.id,
                feed_url=feed.url,
                status="unchanged_hash",
                http_status=status_code,
            )

        try:
            outcome = parse_feed_xml(raw_bytes)
        except FeedParseError as exc:
            logger.warning("Malformed XML from feed %s (%s): %s", feed.id, feed.url, exc)
            return FeedPollResult(
                feed_id=feed.id,
                feed_url=feed.url,
                status="error",
                http_status=status_code,
                error=str(exc),
            )

        frontier_items = [
            FrontierItem(
                feed_id=feed.id,
                url=entry.link,
                guid=entry.guid,
                domain=entry.domain,
                status=FrontierStatus.PENDING,
                discovered_at=now,
                rss_title=entry.title,
                rss_summary=entry.summary,
                rss_published_at=entry.published_at,
            )
            for entry in outcome.entries
        ]
        inserted = enqueue_frontier_items(self._conn, frontier_items)

        new_etag = response.headers.get("etag")
        new_last_modified = response.headers.get("last-modified")
        update_feed_poll_state(
            self._conn,
            feed.id,
            now,
            etag=new_etag,
            last_modified=new_last_modified,
            content_hash=content_hash,
        )

        return FeedPollResult(
            feed_id=feed.id,
            feed_url=feed.url,
            status="updated",
            http_status=status_code,
            items_parsed=len(outcome.entries),
            items_inserted=inserted,
            items_skipped=outcome.skipped_count,
        )

    def poll_all_feeds(self) -> list[FeedPollResult]:
        """Poll all active feeds in the registry, continuing gracefully on individual errors."""
        feeds = get_active_feeds(self._conn)
        if not feeds:
            return []

        if self._external_client is not None:
            return [self.poll_feed(feed, client=self._external_client) for feed in feeds]

        with httpx.Client(
            timeout=self._timeout_seconds,
            follow_redirects=True,
        ) as client:
            return [self.poll_feed(feed, client=client) for feed in feeds]


def poll_all_feeds(
    conn: sqlite3.Connection,
    *,
    client: httpx.Client | None = None,
    min_host_delay: float = DEFAULT_MIN_HOST_DELAY_SECONDS,
) -> list[FeedPollResult]:
    """Convenience function to poll all active feeds once."""
    poller = FeedPoller(conn, client=client, min_host_delay=min_host_delay)
    return poller.poll_all_feeds()


class AsyncFeedPoller:
    """Async conditional RSS/Atom poller running distinct RSS hosts concurrently."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        client: httpx.AsyncClient | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        min_host_delay: float = DEFAULT_MIN_HOST_DELAY_SECONDS,
        now_fn: Callable[[], datetime] = utc_now,
        monotonic_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._conn = conn
        self._external_client = client
        self._user_agent = user_agent
        self._timeout_seconds = timeout_seconds
        self._min_host_delay = max(0.0, min_host_delay)
        self._now_fn = now_fn
        self._monotonic_fn = monotonic_fn or time.monotonic
        self._sleep_fn = sleep_fn or asyncio.sleep
        self._host_locks: dict[tuple[int, str], asyncio.Lock] = {}
        self._last_host_request: dict[str, float] = {}

    def _get_host_lock(self, host: str) -> asyncio.Lock:
        key = (id(asyncio.get_running_loop()), host)
        lock = self._host_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._host_locks[key] = lock
        return lock

    async def _wait_for_host_politeness(self, host: str) -> None:
        if self._min_host_delay <= 0.0:
            return
        async with self._get_host_lock(host):
            last_ts = self._last_host_request.get(host)
            if last_ts is not None:
                elapsed = self._monotonic_fn() - last_ts
                remaining = self._min_host_delay - elapsed
                if remaining > 0:
                    await self._sleep_fn(remaining)
            self._last_host_request[host] = self._monotonic_fn()

    def _build_request_headers(self, feed: Feed) -> dict[str, str]:
        headers = {
            "User-Agent": self._user_agent,
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
        }
        if feed.etag:
            headers["If-None-Match"] = feed.etag
        if feed.last_modified:
            headers["If-Modified-Since"] = feed.last_modified
        return headers

    async def poll_feed(
        self,
        feed: Feed,
        *,
        client: httpx.AsyncClient,
    ) -> FeedPollResult:
        if feed.id is None:
            raise ValueError("Feed must have a database id before polling")

        host = extract_domain(feed.url)
        await self._wait_for_host_politeness(host)

        now = self._now_fn()
        headers = self._build_request_headers(feed)
        try:
            response = await client.get(feed.url, headers=headers)
        except httpx.TimeoutException as exc:
            logger.warning("Timeout polling feed %s (%s): %s", feed.id, feed.url, exc)
            return FeedPollResult(
                feed_id=feed.id,
                feed_url=feed.url,
                status="error",
                error=f"Request timeout: {exc}",
            )
        except httpx.HTTPError as exc:
            logger.warning("Network error polling feed %s (%s): %s", feed.id, feed.url, exc)
            return FeedPollResult(
                feed_id=feed.id,
                feed_url=feed.url,
                status="error",
                error=f"Network error: {exc}",
            )

        status_code = response.status_code
        if status_code == 304:
            update_feed_poll_state(self._conn, feed.id, now)
            return FeedPollResult(
                feed_id=feed.id,
                feed_url=feed.url,
                status="not_modified",
                http_status=304,
            )

        if status_code == 429 or status_code >= 400:
            logger.warning("Skipping feed %s (%s) due to HTTP %s", feed.id, feed.url, status_code)
            return FeedPollResult(
                feed_id=feed.id,
                feed_url=feed.url,
                status="error",
                http_status=status_code,
                error=f"HTTP {status_code}",
            )

        raw_bytes = response.content
        content_hash = compute_content_hash(raw_bytes)
        if feed.content_hash is not None and content_hash == feed.content_hash:
            update_feed_poll_state(self._conn, feed.id, now)
            return FeedPollResult(
                feed_id=feed.id,
                feed_url=feed.url,
                status="unchanged_hash",
                http_status=status_code,
            )

        try:
            outcome = parse_feed_xml(raw_bytes)
        except FeedParseError as exc:
            logger.warning("Malformed XML from feed %s (%s): %s", feed.id, feed.url, exc)
            return FeedPollResult(
                feed_id=feed.id,
                feed_url=feed.url,
                status="error",
                http_status=status_code,
                error=str(exc),
            )

        frontier_items = [
            FrontierItem(
                feed_id=feed.id,
                url=entry.link,
                guid=entry.guid,
                domain=entry.domain,
                status=FrontierStatus.PENDING,
                discovered_at=now,
                rss_title=entry.title,
                rss_summary=entry.summary,
                rss_published_at=entry.published_at,
            )
            for entry in outcome.entries
        ]
        inserted = enqueue_frontier_items(self._conn, frontier_items)
        update_feed_poll_state(
            self._conn,
            feed.id,
            now,
            etag=response.headers.get("etag"),
            last_modified=response.headers.get("last-modified"),
            content_hash=content_hash,
        )
        return FeedPollResult(
            feed_id=feed.id,
            feed_url=feed.url,
            status="updated",
            http_status=status_code,
            items_parsed=len(outcome.entries),
            items_inserted=inserted,
            items_skipped=outcome.skipped_count,
        )

    async def poll_stale_feeds(
        self,
        *,
        max_age_seconds: float = 180.0,
        max_concurrency: int = 8,
    ) -> list[FeedPollResult]:
        """Poll active feeds whose last_polled_at is missing or older than ``max_age_seconds``."""
        feeds = get_active_feeds(self._conn)
        if not feeds:
            return []

        now = self._now_fn()
        cutoff = now - timedelta(seconds=max(0.0, max_age_seconds))
        due_feeds = [
            f for f in feeds if f.last_polled_at is None or f.last_polled_at <= cutoff
        ]
        if not due_feeds:
            return []

        by_host: dict[str, list[Feed]] = defaultdict(list)
        for feed in due_feeds:
            by_host[extract_domain(feed.url)].append(feed)

        async def _run_batch(client: httpx.AsyncClient) -> list[FeedPollResult]:
            results: list[FeedPollResult] = []
            sem = asyncio.Semaphore(max(1, max_concurrency))

            async def _host_worker(host_feeds: list[Feed]) -> None:
                for feed in host_feeds:
                    async with sem:
                        res = await self.poll_feed(feed, client=client)
                    results.append(res)

            await asyncio.gather(*(_host_worker(fs) for fs in by_host.values()))
            return results

        if self._external_client is not None:
            return await _run_batch(self._external_client)

        async with httpx.AsyncClient(
            timeout=self._timeout_seconds,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=max(10, max_concurrency * 2)),
        ) as client:
            return await _run_batch(client)
