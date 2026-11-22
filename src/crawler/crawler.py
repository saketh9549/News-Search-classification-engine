"""Polite web crawler with per-domain rate limiting, robots.txt compliance, and backoff."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

import httpx

from src.crawler.robots import RobotsCache, RobotsFetchError
from src.crawler.schemas import (
    DEFAULT_BASE_BACKOFF_SECONDS,
    DEFAULT_MAX_BACKOFF_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MIN_DOMAIN_DELAY_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    CrawlResult,
)
from src.models.schemas import FrontierItem, FrontierStatus, utc_now
from src.models.urls import normalize_url
from src.storage.repository import (
    get_due_frontier_items,
    reset_stale_in_progress_items,
    update_frontier_status,
)

logger = logging.getLogger(__name__)


def compute_backoff_seconds(
    retry_count: int,
    *,
    base_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
    max_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
    retry_after_header: str | None = None,
) -> float:
    """Compute exponential backoff delay, honoring numeric Retry-After headers when larger."""
    exponent = max(0, retry_count - 1)
    delay = min(max_seconds, base_seconds * (2**exponent))
    if retry_after_header:
        try:
            retry_after_val = float(retry_after_header.strip())
            if retry_after_val > 0:
                delay = max(delay, min(max_seconds, retry_after_val))
        except ValueError:
            pass
    return delay


class PoliteCrawler:
    """Fetches frontier URLs while enforcing per-domain politeness and exponential backoff."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        client: httpx.Client | None = None,
        robots_cache: RobotsCache | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        min_domain_delay: float = DEFAULT_MIN_DOMAIN_DELAY_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
        max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
        now_fn: Callable[[], datetime] = utc_now,
        monotonic_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        if min_domain_delay < DEFAULT_MIN_DOMAIN_DELAY_SECONDS:
            raise ValueError(
                f"min_domain_delay must be >= {DEFAULT_MIN_DOMAIN_DELAY_SECONDS} seconds per politeness rules"
            )
        resolved_mono = monotonic_fn or time.monotonic
        resolved_sleep = sleep_fn or time.sleep
        self._conn = conn
        self._external_client = client
        self._user_agent = user_agent
        self._min_domain_delay = float(min_domain_delay)
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._now_fn = now_fn
        self._monotonic_fn = resolved_mono
        self._sleep_fn = resolved_sleep
        self._robots_cache = robots_cache or RobotsCache(
            user_agent=user_agent,
            timeout_seconds=timeout_seconds,
            monotonic_fn=resolved_mono,
        )
        self._locks_guard = threading.Lock()
        self._domain_locks: dict[str, threading.Lock] = {}
        self._last_domain_request: dict[str, float] = {}

    def _get_domain_lock(self, domain: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._domain_locks.get(domain)
            if lock is None:
                lock = threading.Lock()
                self._domain_locks[domain] = lock
            return lock

    def _wait_for_domain(self, domain: str, required_delay: float | None = None) -> None:
        delay = max(self._min_domain_delay, required_delay or 0.0)
        with self._get_domain_lock(domain):
            last_ts = self._last_domain_request.get(domain)
            if last_ts is not None:
                elapsed = self._monotonic_fn() - last_ts
                remaining = delay - elapsed
                if remaining > 0:
                    self._sleep_fn(remaining)
            self._last_domain_request[domain] = self._monotonic_fn()

    def _record_transient_failure(
        self,
        item: FrontierItem,
        *,
        now: datetime,
        http_status: int | None,
        error: str,
        retry_after_header: str | None = None,
    ) -> CrawlResult:
        assert item.id is not None
        next_retry_count = item.retry_count + 1
        if next_retry_count > self._max_retries:
            update_frontier_status(
                self._conn,
                item.id,
                status=FrontierStatus.FAILED,
                retry_count=next_retry_count,
                next_attempt_at=None,
                last_http_status=http_status,
            )
            return CrawlResult(
                frontier_id=item.id,
                feed_id=item.feed_id,
                url=item.url,
                final_url=item.url,
                domain=item.domain,
                status=FrontierStatus.FAILED,
                http_status=http_status,
                fetched_at=now,
                error=error,
            )

        backoff = compute_backoff_seconds(
            next_retry_count,
            base_seconds=self._base_backoff_seconds,
            max_seconds=self._max_backoff_seconds,
            retry_after_header=retry_after_header,
        )
        next_attempt = now + timedelta(seconds=backoff)
        update_frontier_status(
            self._conn,
            item.id,
            status=FrontierStatus.PENDING,
            retry_count=next_retry_count,
            next_attempt_at=next_attempt,
            last_http_status=http_status,
        )
        return CrawlResult(
            frontier_id=item.id,
            feed_id=item.feed_id,
            url=item.url,
            final_url=item.url,
            domain=item.domain,
            status=FrontierStatus.PENDING,
            http_status=http_status,
            fetched_at=now,
            error=error,
        )

    def crawl_item(
        self,
        item: FrontierItem,
        *,
        client: httpx.Client | None = None,
    ) -> CrawlResult:
        """Fetch raw HTML for a single frontier item while respecting robots.txt and delays."""
        if item.id is None:
            raise ValueError("FrontierItem must have a database id before crawling")

        active_client = client or self._external_client
        if active_client is None:
            with httpx.Client(
                timeout=self._timeout_seconds,
                follow_redirects=True,
            ) as managed_client:
                return self.crawl_item(item, client=managed_client)

        now = self._now_fn()
        update_frontier_status(self._conn, item.id, status=FrontierStatus.IN_PROGRESS)

        try:
            decision = self._robots_cache.check(
                item.url,
                client=active_client,
                before_fetch=lambda host: self._wait_for_domain(host, self._min_domain_delay),
            )
        except RobotsFetchError as exc:
            logger.warning("Deferring %s due to robots.txt error: %s", item.url, exc)
            return self._record_transient_failure(
                item,
                now=now,
                http_status=None,
                error=str(exc),
            )

        if not decision.allowed:
            logger.info("Skipping disallowed URL %s: %s", item.url, decision.reason)
            update_frontier_status(
                self._conn,
                item.id,
                status=FrontierStatus.SKIPPED,
                next_attempt_at=None,
                last_http_status=None,
            )
            return CrawlResult(
                frontier_id=item.id,
                feed_id=item.feed_id,
                url=item.url,
                final_url=item.url,
                domain=item.domain,
                status=FrontierStatus.SKIPPED,
                fetched_at=now,
                error=decision.reason,
            )

        required_delay = max(self._min_domain_delay, decision.crawl_delay or 0.0)
        self._wait_for_domain(item.domain, required_delay)

        try:
            response = active_client.get(
                item.url,
                headers={
                    "User-Agent": self._user_agent,
                    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
                },
            )
        except httpx.TimeoutException as exc:
            logger.warning("Timeout crawling %s: %s", item.url, exc)
            return self._record_transient_failure(
                item,
                now=now,
                http_status=None,
                error=f"Request timeout: {exc}",
            )
        except httpx.HTTPError as exc:
            logger.warning("Network error crawling %s: %s", item.url, exc)
            return self._record_transient_failure(
                item,
                now=now,
                http_status=None,
                error=f"Network error: {exc}",
            )

        status_code = response.status_code
        if status_code == 429 or status_code >= 500:
            logger.warning("Transient HTTP %s for %s", status_code, item.url)
            return self._record_transient_failure(
                item,
                now=now,
                http_status=status_code,
                error=f"HTTP {status_code}",
                retry_after_header=response.headers.get("retry-after"),
            )

        if status_code >= 400:
            logger.warning("Permanent HTTP %s for %s", status_code, item.url)
            update_frontier_status(
                self._conn,
                item.id,
                status=FrontierStatus.FAILED,
                next_attempt_at=None,
                last_http_status=status_code,
            )
            return CrawlResult(
                frontier_id=item.id,
                feed_id=item.feed_id,
                url=item.url,
                final_url=item.url,
                domain=item.domain,
                status=FrontierStatus.FAILED,
                http_status=status_code,
                fetched_at=now,
                error=f"HTTP {status_code}",
            )

        content_type = response.headers.get("content-type", "").lower()
        if content_type and not (
            "html" in content_type or "xhtml" in content_type or "text/plain" in content_type
        ):
            logger.info("Skipping non-HTML content-type %r at %s", content_type, item.url)
            update_frontier_status(
                self._conn,
                item.id,
                status=FrontierStatus.SKIPPED,
                next_attempt_at=None,
                last_http_status=status_code,
            )
            return CrawlResult(
                frontier_id=item.id,
                feed_id=item.feed_id,
                url=item.url,
                final_url=item.url,
                domain=item.domain,
                status=FrontierStatus.SKIPPED,
                http_status=status_code,
                fetched_at=now,
                error=f"Unsupported content-type: {content_type}",
            )

        raw_final = str(response.url) if response.url else item.url
        try:
            final_url = normalize_url(raw_final)
        except ValueError:
            final_url = item.url

        update_frontier_status(
            self._conn,
            item.id,
            status=FrontierStatus.FETCHED,
            next_attempt_at=None,
            last_http_status=status_code,
        )
        return CrawlResult(
            frontier_id=item.id,
            feed_id=item.feed_id,
            url=item.url,
            final_url=final_url,
            domain=item.domain,
            status=FrontierStatus.FETCHED,
            http_status=status_code,
            html=response.text,
            fetched_at=now,
        )

    def crawl_pending(
        self,
        *,
        limit: int = 20,
        domain: str | None = None,
        interleave_domains: bool = False,
        recover_stale: bool = True,
    ) -> list[CrawlResult]:
        """Fetch due pending items from crawl_frontier and crawl them politely."""
        if recover_stale:
            recovered = reset_stale_in_progress_items(self._conn)
            if recovered:
                logger.info("Recovered %d stale in_progress frontier item(s)", recovered)

        fetch_limit = limit * 3 if (interleave_domains and domain is None) else limit
        items = get_due_frontier_items(
            self._conn,
            now=self._now_fn(),
            limit=fetch_limit,
            domain=domain,
            interleave_domains=interleave_domains,
        )
        if not items:
            return []

        def _run_batch(client: httpx.Client) -> list[CrawlResult]:
            results: list[CrawlResult] = []
            blocked_domains: set[str] = set()
            for item in items:
                if len(results) >= limit:
                    break
                if interleave_domains and item.domain in blocked_domains:
                    continue
                res = self.crawl_item(item, client=client)
                results.append(res)
                if interleave_domains and res.http_status in {401, 403, 429}:
                    blocked_domains.add(item.domain)
            return results

        if self._external_client is not None:
            return _run_batch(self._external_client)

        with httpx.Client(
            timeout=self._timeout_seconds,
            follow_redirects=True,
        ) as client:
            return _run_batch(client)


class AsyncPoliteCrawler:
    """Async multi-domain crawler enforcing strict per-domain >= 5s delays concurrently across hosts."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        client: httpx.AsyncClient | None = None,
        robots_cache: RobotsCache | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        min_domain_delay: float = DEFAULT_MIN_DOMAIN_DELAY_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
        max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
        now_fn: Callable[[], datetime] = utc_now,
        monotonic_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if min_domain_delay < DEFAULT_MIN_DOMAIN_DELAY_SECONDS:
            raise ValueError(
                f"min_domain_delay must be >= {DEFAULT_MIN_DOMAIN_DELAY_SECONDS} seconds per politeness rules"
            )
        resolved_mono = monotonic_fn or time.monotonic
        resolved_sleep = sleep_fn or asyncio.sleep
        self._conn = conn
        self._external_client = client
        self._user_agent = user_agent
        self._min_domain_delay = float(min_domain_delay)
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._now_fn = now_fn
        self._monotonic_fn = resolved_mono
        self._sleep_fn = resolved_sleep
        self._robots_cache = robots_cache or RobotsCache(
            user_agent=user_agent,
            timeout_seconds=timeout_seconds,
            monotonic_fn=resolved_mono,
        )
        self._domain_locks: dict[tuple[int, str], asyncio.Lock] = {}
        self._last_domain_request: dict[str, float] = {}

    def _get_domain_lock(self, domain: str) -> asyncio.Lock:
        key = (id(asyncio.get_running_loop()), domain)
        lock = self._domain_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._domain_locks[key] = lock
        return lock

    async def _wait_for_domain(self, domain: str, required_delay: float | None = None) -> None:
        delay = max(self._min_domain_delay, required_delay or 0.0)
        async with self._get_domain_lock(domain):
            last_ts = self._last_domain_request.get(domain)
            if last_ts is not None:
                elapsed = self._monotonic_fn() - last_ts
                remaining = delay - elapsed
                if remaining > 0:
                    await self._sleep_fn(remaining)
            self._last_domain_request[domain] = self._monotonic_fn()

    def _record_transient_failure(
        self,
        item: FrontierItem,
        *,
        now: datetime,
        http_status: int | None,
        error: str,
        retry_after_header: str | None = None,
    ) -> CrawlResult:
        assert item.id is not None
        next_retry_count = item.retry_count + 1
        if next_retry_count > self._max_retries:
            update_frontier_status(
                self._conn,
                item.id,
                status=FrontierStatus.FAILED,
                retry_count=next_retry_count,
                next_attempt_at=None,
                last_http_status=http_status,
            )
            return CrawlResult(
                frontier_id=item.id,
                feed_id=item.feed_id,
                url=item.url,
                final_url=item.url,
                domain=item.domain,
                status=FrontierStatus.FAILED,
                http_status=http_status,
                fetched_at=now,
                error=error,
            )

        backoff = compute_backoff_seconds(
            next_retry_count,
            base_seconds=self._base_backoff_seconds,
            max_seconds=self._max_backoff_seconds,
            retry_after_header=retry_after_header,
        )
        next_attempt = now + timedelta(seconds=backoff)
        update_frontier_status(
            self._conn,
            item.id,
            status=FrontierStatus.PENDING,
            retry_count=next_retry_count,
            next_attempt_at=next_attempt,
            last_http_status=http_status,
        )
        return CrawlResult(
            frontier_id=item.id,
            feed_id=item.feed_id,
            url=item.url,
            final_url=item.url,
            domain=item.domain,
            status=FrontierStatus.PENDING,
            http_status=http_status,
            fetched_at=now,
            error=error,
        )

    async def crawl_item(
        self,
        item: FrontierItem,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> CrawlResult:
        """Async fetch for a single frontier item respecting robots.txt and per-domain delays."""
        if item.id is None:
            raise ValueError("FrontierItem must have a database id before crawling")

        active_client = client or self._external_client
        if active_client is None:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                follow_redirects=True,
            ) as managed_client:
                return await self.crawl_item(item, client=managed_client)

        now = self._now_fn()
        update_frontier_status(self._conn, item.id, status=FrontierStatus.IN_PROGRESS)

        try:
            decision = await self._robots_cache.acheck(
                item.url,
                client=active_client,
                before_fetch=lambda host: self._wait_for_domain(host, self._min_domain_delay),
            )
        except RobotsFetchError as exc:
            logger.warning("Deferring %s due to robots.txt error: %s", item.url, exc)
            return self._record_transient_failure(
                item,
                now=now,
                http_status=None,
                error=str(exc),
            )

        if not decision.allowed:
            logger.info("Skipping disallowed URL %s: %s", item.url, decision.reason)
            update_frontier_status(
                self._conn,
                item.id,
                status=FrontierStatus.SKIPPED,
                next_attempt_at=None,
                last_http_status=None,
            )
            return CrawlResult(
                frontier_id=item.id,
                feed_id=item.feed_id,
                url=item.url,
                final_url=item.url,
                domain=item.domain,
                status=FrontierStatus.SKIPPED,
                fetched_at=now,
                error=decision.reason,
            )

        required_delay = max(self._min_domain_delay, decision.crawl_delay or 0.0)
        await self._wait_for_domain(item.domain, required_delay)

        try:
            response = await active_client.get(
                item.url,
                headers={
                    "User-Agent": self._user_agent,
                    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
                },
            )
        except httpx.TimeoutException as exc:
            logger.warning("Timeout crawling %s: %s", item.url, exc)
            return self._record_transient_failure(
                item,
                now=now,
                http_status=None,
                error=f"Request timeout: {exc}",
            )
        except httpx.HTTPError as exc:
            logger.warning("Network error crawling %s: %s", item.url, exc)
            return self._record_transient_failure(
                item,
                now=now,
                http_status=None,
                error=f"Network error: {exc}",
            )

        status_code = response.status_code
        if status_code == 429 or status_code >= 500:
            logger.warning("Transient HTTP %s for %s", status_code, item.url)
            return self._record_transient_failure(
                item,
                now=now,
                http_status=status_code,
                error=f"HTTP {status_code}",
                retry_after_header=response.headers.get("retry-after"),
            )

        if status_code >= 400:
            logger.warning("Permanent HTTP %s for %s", status_code, item.url)
            update_frontier_status(
                self._conn,
                item.id,
                status=FrontierStatus.FAILED,
                next_attempt_at=None,
                last_http_status=status_code,
            )
            return CrawlResult(
                frontier_id=item.id,
                feed_id=item.feed_id,
                url=item.url,
                final_url=item.url,
                domain=item.domain,
                status=FrontierStatus.FAILED,
                http_status=status_code,
                fetched_at=now,
                error=f"HTTP {status_code}",
            )

        content_type = response.headers.get("content-type", "").lower()
        if content_type and not (
            "html" in content_type or "xhtml" in content_type or "text/plain" in content_type
        ):
            logger.info("Skipping non-HTML content-type %r at %s", content_type, item.url)
            update_frontier_status(
                self._conn,
                item.id,
                status=FrontierStatus.SKIPPED,
                next_attempt_at=None,
                last_http_status=status_code,
            )
            return CrawlResult(
                frontier_id=item.id,
                feed_id=item.feed_id,
                url=item.url,
                final_url=item.url,
                domain=item.domain,
                status=FrontierStatus.SKIPPED,
                http_status=status_code,
                fetched_at=now,
                error=f"Unsupported content-type: {content_type}",
            )

        raw_final = str(response.url) if response.url else item.url
        try:
            final_url = normalize_url(raw_final)
        except ValueError:
            final_url = item.url

        update_frontier_status(
            self._conn,
            item.id,
            status=FrontierStatus.FETCHED,
            next_attempt_at=None,
            last_http_status=status_code,
        )
        return CrawlResult(
            frontier_id=item.id,
            feed_id=item.feed_id,
            url=item.url,
            final_url=final_url,
            domain=item.domain,
            status=FrontierStatus.FETCHED,
            http_status=status_code,
            html=response.text,
            fetched_at=now,
        )

    async def crawl_items(
        self,
        items: list[FrontierItem],
        *,
        max_concurrency: int = 8,
        stop_on_block: bool = True,
    ) -> list[CrawlResult]:
        """Crawl a specific list of FrontierItems concurrently across distinct domains."""
        if not items:
            return []

        per_domain_queues: dict[str, deque[FrontierItem]] = defaultdict(deque)
        for item in items:
            per_domain_queues[item.domain].append(item)

        async def _run_batch(client: httpx.AsyncClient) -> list[CrawlResult]:
            results: list[CrawlResult] = []
            sem = asyncio.Semaphore(max(1, max_concurrency))

            async def _domain_worker(dom: str, queue: deque[FrontierItem]) -> None:
                while queue:
                    current = queue.popleft()
                    async with sem:
                        res = await self.crawl_item(current, client=client)
                    results.append(res)
                    if stop_on_block and res.http_status in {401, 403, 429}:
                        logger.info(
                            "Pausing domain %s for remainder of batch after HTTP %s",
                            dom,
                            res.http_status,
                        )
                        return

            await asyncio.gather(
                *(_domain_worker(dom, q) for dom, q in per_domain_queues.items())
            )
            return results

        if self._external_client is not None:
            return await _run_batch(self._external_client)

        async with httpx.AsyncClient(
            timeout=self._timeout_seconds,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=max(10, max_concurrency * 2)),
        ) as client:
            return await _run_batch(client)

    async def crawl_pending(
        self,
        *,
        limit: int = 20,
        domain: str | None = None,
        interleave_domains: bool = True,
        recover_stale: bool = True,
        max_concurrency: int = 8,
    ) -> list[CrawlResult]:
        """Crawl due items concurrently across distinct domains while keeping >= 5s per host."""
        if recover_stale:
            recovered = reset_stale_in_progress_items(self._conn)
            if recovered:
                logger.info("Recovered %d stale in_progress frontier item(s)", recovered)

        fetch_limit = limit * 3 if (interleave_domains and domain is None) else limit
        items = get_due_frontier_items(
            self._conn,
            now=self._now_fn(),
            limit=fetch_limit,
            domain=domain,
            interleave_domains=interleave_domains,
        )
        if not items:
            return []

        per_domain_queues: dict[str, deque[FrontierItem]] = defaultdict(deque)
        for item in items:
            per_domain_queues[item.domain].append(item)

        async def _run_batch(client: httpx.AsyncClient) -> list[CrawlResult]:
            results: list[CrawlResult] = []
            claimed = 0
            sem = asyncio.Semaphore(max(1, max_concurrency))

            async def _domain_worker(dom: str, queue: deque[FrontierItem]) -> None:
                nonlocal claimed
                while queue:
                    if claimed >= limit:
                        return
                    claimed += 1
                    current = queue.popleft()
                    async with sem:
                        res = await self.crawl_item(current, client=client)
                    results.append(res)
                    if interleave_domains and res.http_status in {401, 403, 429}:
                        logger.info(
                            "Pausing domain %s for remainder of batch after HTTP %s",
                            dom,
                            res.http_status,
                        )
                        return

            await asyncio.gather(
                *(_domain_worker(dom, q) for dom, q in per_domain_queues.items())
            )
            return results

        if self._external_client is not None:
            return await _run_batch(self._external_client)

        async with httpx.AsyncClient(
            timeout=self._timeout_seconds,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=max(10, max_concurrency * 2)),
        ) as client:
            return await _run_batch(client)
