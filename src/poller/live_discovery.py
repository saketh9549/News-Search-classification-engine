"""Domain-scoped live RSS discovery restricted strictly to registered news publisher domains."""

from __future__ import annotations

import logging
import sqlite3
from urllib.parse import parse_qs, quote_plus, urlsplit

import feedparser
import httpx

from src.models.schemas import FrontierItem, FrontierStatus, utc_now
from src.models.urls import extract_domain, normalize_url
from src.poller.parser import is_article_title, is_article_url, parse_feed_timestamp
from src.poller.schemas import DEFAULT_TIMEOUT_SECONDS, DEFAULT_USER_AGENT
from src.storage.repository import enqueue_frontier_items, get_active_feeds

logger = logging.getLogger(__name__)

# Registered publisher web domains corresponding to our RSS feed sources (excluding 403-blocked outlets)
ALLOWED_PUBLISHER_ROOTS: tuple[str, ...] = (
    "bbc.co.uk",
    "bbc.com",
    "thehindu.com",
    "theguardian.com",
    "indianexpress.com",
    "timesofindia.indiatimes.com",
    "hindustantimes.com",
    "economictimes.indiatimes.com",
    "indiatoday.in",
    "aljazeera.com",
)


def _root_matches(host: str, root: str) -> bool:
    h = host.lower()
    r = root.lower()
    return h == r or h.endswith("." + r)


def _resolve_target_url(raw_link: str) -> str | None:
    """Unwrap Bing News redirect links (apiclick.aspx?url=...) into clean direct publisher URLs."""
    if not raw_link:
        return None
    parts = urlsplit(raw_link.strip())
    host = (parts.hostname or "").lower()
    if host.endswith("bing.com") and "apiclick" in parts.path.lower():
        qs = parse_qs(parts.query)
        target = qs.get("url")
        if target and target[0]:
            return target[0].strip()
    return raw_link.strip()


def _domain_to_feed_id_map(conn: sqlite3.Connection) -> dict[str, int]:
    """Map registered publisher root domains to an active feed_id in rss_feeds."""
    feeds = get_active_feeds(conn)
    mapping: dict[str, int] = {}
    for feed in feeds:
        if feed.id is None:
            continue
        feed_host = extract_domain(feed.url)
        for root in ALLOWED_PUBLISHER_ROOTS:
            if root not in mapping and (
                _root_matches(feed_host, root)
                or (root in {"bbc.co.uk", "bbc.com"} and "bbci.co.uk" in feed_host)
            ):
                mapping[root] = feed.id
    fallback_id = feeds[0].id if feeds and feeds[0].id is not None else 1
    for root in ALLOWED_PUBLISHER_ROOTS:
        mapping.setdefault(root, fallback_id)
    return mapping


async def discover_query_links_from_registered_domains(
    conn: sqlite3.Connection,
    query: str,
    *,
    client: httpx.AsyncClient | None = None,
    limit: int = 8,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[FrontierItem]:
    """Discover recent article links for ``query`` strictly within registered publisher domains.

    Uses Bing News RSS with explicit ``site:`` constraints limited to ``ALLOWED_PUBLISHER_ROOTS``,
    validates every discovered URL through ``normalize_url`` and ``is_article_url``, and enqueues
    them idempotently into ``crawl_frontier`` for polite crawling.
    """
    cleaned_query = " ".join((query or "").strip().split())
    if not cleaned_query or limit <= 0:
        return []

    site_clause = " OR ".join(f"site:{root}" for root in ALLOWED_PUBLISHER_ROOTS)
    search_expr = f"{cleaned_query} ({site_clause})"
    rss_url = f"https://www.bing.com/news/search?q={quote_plus(search_expr)}&format=rss&mkt=en-IN"

    async def _fetch(active_client: httpx.AsyncClient) -> bytes | None:
        try:
            resp = await active_client.get(
                rss_url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
                },
                timeout=timeout_seconds,
            )
            if resp.status_code != 200:
                logger.warning("Domain-scoped live discovery returned HTTP %s", resp.status_code)
                return None
            return resp.content
        except httpx.HTTPError as exc:
            logger.warning("Domain-scoped live discovery network error: %s", exc)
            return None

    if client is not None:
        raw_bytes = await _fetch(client)
    else:
        async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as managed:
            raw_bytes = await _fetch(managed)

    if not raw_bytes:
        return []

    parsed = feedparser.parse(raw_bytes)
    feed_id_by_root = _domain_to_feed_id_map(conn)
    now = utc_now()
    discovered: list[FrontierItem] = []
    seen_urls: set[str] = set()

    for entry in parsed.entries:
        raw_link = entry.get("link")
        if not isinstance(raw_link, str) or not raw_link.strip():
            continue
        direct_url = _resolve_target_url(raw_link)
        if not direct_url:
            continue
        try:
            normalized = normalize_url(direct_url)
            domain = extract_domain(normalized)
        except ValueError:
            continue

        matched_root = next((r for r in ALLOWED_PUBLISHER_ROOTS if _root_matches(domain, r)), None)
        if matched_root is None:
            continue
        if normalized in seen_urls or not is_article_url(normalized):
            continue

        title = str(entry.get("title") or "").strip() or None
        if not is_article_title(title):
            continue
        summary = str(entry.get("summary") or entry.get("description") or "").strip() or None
        pub_raw = entry.get("published") or entry.get("updated")
        pub_dt = parse_feed_timestamp(str(pub_raw)) if pub_raw else None

        seen_urls.add(normalized)
        discovered.append(
            FrontierItem(
                feed_id=feed_id_by_root[matched_root],
                url=normalized,
                guid=normalized,
                domain=domain,
                status=FrontierStatus.PENDING,
                discovered_at=now,
                rss_title=title,
                rss_summary=summary,
                rss_published_at=pub_dt,
            )
        )
        if len(discovered) >= limit:
            break

    if not discovered:
        return []

    enqueue_frontier_items(conn, discovered)

    # Return hydrated pending FrontierItems (with database ids) ready for AsyncPoliteCrawler
    placeholders = ",".join("?" for _ in discovered)
    rows = conn.execute(
        f"""
        SELECT id, feed_id, url, guid, domain, status, retry_count,
               next_attempt_at, last_http_status, discovered_at,
               rss_title, rss_summary, rss_published_at
        FROM crawl_frontier
        WHERE status = 'pending' AND url IN ({placeholders});
        """,
        [item.url for item in discovered],
    ).fetchall()
    from src.storage.repository import _row_to_frontier_item

    return [_row_to_frontier_item(r) for r in rows]
