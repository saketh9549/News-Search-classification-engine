"""robots.txt fetching, parsing, and per-origin caching."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from src.crawler.schemas import (
    DEFAULT_ROBOTS_TTL_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    RobotsDecision,
)

logger = logging.getLogger(__name__)


class RobotsFetchError(RuntimeError):
    """Raised when robots.txt cannot be fetched reliably and crawling must defer."""


class RobotsCache:
    """Thread-safe TTL cache of RobotFileParser rules per (scheme, netloc)."""

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        ttl_seconds: float = DEFAULT_ROBOTS_TTL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        monotonic_fn: Callable[[], float] | None = None,
    ) -> None:
        self._user_agent = user_agent
        self._ttl_seconds = ttl_seconds
        self._timeout_seconds = timeout_seconds
        if monotonic_fn is None:
            import time

            monotonic_fn = time.monotonic
        self._monotonic_fn = monotonic_fn
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], tuple[float, RobotFileParser]] = {}
        self._async_locks: dict[tuple[int, tuple[str, str]], asyncio.Lock] = {}

    def _get_async_lock(self, origin: tuple[str, str]) -> asyncio.Lock:
        loop_id = id(asyncio.get_running_loop())
        key = (loop_id, origin)
        with self._lock:
            lock = self._async_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._async_locks[key] = lock
            return lock

    def _get_cached_parser(self, origin: tuple[str, str]) -> RobotFileParser | None:
        now = self._monotonic_fn()
        with self._lock:
            entry = self._cache.get(origin)
            if entry is None:
                return None
            expires_at, parser = entry
            if now >= expires_at:
                del self._cache[origin]
                return None
            return parser

    def _store_parser(self, origin: tuple[str, str], parser: RobotFileParser) -> None:
        expires_at = self._monotonic_fn() + self._ttl_seconds
        with self._lock:
            self._cache[origin] = (expires_at, parser)

    def check(
        self,
        url: str,
        *,
        client: httpx.Client,
        before_fetch: Callable[[str], None] | None = None,
    ) -> RobotsDecision:
        """Fetch/lookup robots.txt for ``url`` and return whether fetching is permitted."""
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()
        host = (parts.hostname or netloc).lower()
        origin = (scheme, netloc)

        parser = self._get_cached_parser(origin)
        if parser is None:
            if before_fetch is not None:
                before_fetch(host)

            robots_url = f"{scheme}://{netloc}/robots.txt"
            try:
                response = client.get(
                    robots_url,
                    headers={"User-Agent": self._user_agent},
                    timeout=self._timeout_seconds,
                )
            except httpx.HTTPError as exc:
                raise RobotsFetchError(
                    f"Network error fetching robots.txt for {netloc}: {exc}"
                ) from exc

            parser = RobotFileParser()
            parser.set_url(robots_url)

            if response.status_code == 200:
                clean_robots = response.text.lstrip("\ufeff")
                parser.parse(clean_robots.splitlines())
            elif response.status_code in {404, 410}:
                parser.allow_all = True
            elif response.status_code == 429 or response.status_code >= 500:
                raise RobotsFetchError(
                    f"HTTP {response.status_code} fetching robots.txt for {netloc}"
                )
            else:
                parser.allow_all = True

            self._store_parser(origin, parser)

        allowed = parser.can_fetch(self._user_agent, url)
        raw_delay = parser.crawl_delay(self._user_agent)
        crawl_delay = float(raw_delay) if raw_delay is not None else None
        reason = None if allowed else f"Disallowed by robots.txt for {self._user_agent}"
        return RobotsDecision(allowed=allowed, crawl_delay=crawl_delay, reason=reason)

    async def acheck(
        self,
        url: str,
        *,
        client: httpx.AsyncClient,
        before_fetch: Callable[[str], Awaitable[None]] | None = None,
    ) -> RobotsDecision:
        """Async fetch/lookup of robots.txt for ``url`` using the shared TTL cache."""
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()
        host = (parts.hostname or netloc).lower()
        origin = (scheme, netloc)

        parser = self._get_cached_parser(origin)
        if parser is None:
            async with self._get_async_lock(origin):
                parser = self._get_cached_parser(origin)
                if parser is None:
                    if before_fetch is not None:
                        await before_fetch(host)

                    robots_url = f"{scheme}://{netloc}/robots.txt"
                    try:
                        response = await client.get(
                            robots_url,
                            headers={"User-Agent": self._user_agent},
                            timeout=self._timeout_seconds,
                        )
                    except httpx.HTTPError as exc:
                        raise RobotsFetchError(
                            f"Network error fetching robots.txt for {netloc}: {exc}"
                        ) from exc

                    parser = RobotFileParser()
                    parser.set_url(robots_url)

                    if response.status_code == 200:
                        clean_robots = response.text.lstrip("\ufeff")
                        parser.parse(clean_robots.splitlines())
                    elif response.status_code in {404, 410}:
                        parser.allow_all = True
                    elif response.status_code == 429 or response.status_code >= 500:
                        raise RobotsFetchError(
                            f"HTTP {response.status_code} fetching robots.txt for {netloc}"
                        )
                    else:
                        parser.allow_all = True

                    self._store_parser(origin, parser)

        allowed = parser.can_fetch(self._user_agent, url)
        raw_delay = parser.crawl_delay(self._user_agent)
        crawl_delay = float(raw_delay) if raw_delay is not None else None
        reason = None if allowed else f"Disallowed by robots.txt for {self._user_agent}"
        return RobotsDecision(allowed=allowed, crawl_delay=crawl_delay, reason=reason)
