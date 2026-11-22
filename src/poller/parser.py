"""Safe RSS 2.0 / Atom XML parsing, hashing, and timestamp normalization."""

from __future__ import annotations

import hashlib
import html
import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from xml.etree.ElementTree import ParseError

import defusedxml.ElementTree as SafeET
import feedparser
from dateutil import parser as dateutil_parser
from defusedxml.common import DefusedXmlException

from src.models.urls import extract_domain, normalize_url
from src.poller.schemas import FeedParseOutcome, ParsedFeedEntry

logger = logging.getLogger(__name__)

_ALLOWED_ROOT_TAGS: frozenset[str] = frozenset({"rss", "feed", "rdf"})
_WHITESPACE_RE = re.compile(r"\s+")
_COMMON_TZINFOS = {
    "UTC": timezone.utc,
    "GMT": timezone.utc,
    "Z": timezone.utc,
    "IST": timezone(timedelta(hours=5, minutes=30)),
    "EST": timezone(timedelta(hours=-5)),
    "EDT": timezone(timedelta(hours=-4)),
    "CST": timezone(timedelta(hours=-6)),
    "CDT": timezone(timedelta(hours=-5)),
    "PST": timezone(timedelta(hours=-8)),
    "PDT": timezone(timedelta(hours=-7)),
    "BST": timezone(timedelta(hours=1)),
}


class FeedParseError(ValueError):
    """Raised when an XML feed payload is malformed, unsafe, or not an RSS/Atom feed."""


def compute_content_hash(raw_bytes: bytes) -> str:
    """Compute SHA-256 hex digest of raw feed payload bytes."""
    return hashlib.sha256(raw_bytes).hexdigest()


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    unescaped = html.unescape(value)
    return _WHITESPACE_RE.sub(" ", unescaped).strip()


def parse_feed_timestamp(raw: object) -> datetime | None:
    """Parse RFC 2822 or ISO 8601 date strings into timezone-aware UTC datetimes."""
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None

    dt: datetime | None = None
    try:
        dt = parsedate_to_datetime(cleaned)
    except (TypeError, ValueError):
        try:
            dt = dateutil_parser.parse(cleaned, tzinfos=_COMMON_TZINFOS)
        except (ValueError, OverflowError, TypeError):
            logger.warning("Unable to parse feed timestamp: %r", cleaned)
            return None

    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _validate_xml_payload(raw_bytes: bytes) -> None:
    if not raw_bytes or not raw_bytes.strip():
        raise FeedParseError("Empty XML response payload")

    try:
        root = SafeET.fromstring(raw_bytes)
    except (DefusedXmlException, ParseError, ValueError) as exc:
        raise FeedParseError(f"Malformed or unsafe XML payload: {exc}") from exc

    local_tag = root.tag.rsplit("}", 1)[-1].lower()
    if local_tag not in _ALLOWED_ROOT_TAGS:
        raise FeedParseError(f"Unexpected XML root element {root.tag!r}; expected RSS or Atom")


def _extract_guid(entry: dict[str, object], fallback_url: str) -> str:
    raw_guid = _clean_text(entry.get("id") or entry.get("guid"))
    if not raw_guid:
        return fallback_url
    if raw_guid.lower().startswith(("http://", "https://")):
        try:
            return normalize_url(raw_guid)
        except ValueError:
            return raw_guid
    return raw_guid


_NON_ARTICLE_PATH_RE = re.compile(
    r"/(?:iplayer|sounds|podcasts?|audio|news/videos|videos?|news/av|av|"
    r"programmes?|live|live-updates|liveblog|gallery|photos?|photo-gallery|"
    r"web-stories|visualstories|anchor)(?:/|$)",
    re.IGNORECASE,
)
_NON_ARTICLE_TITLE_RE = re.compile(
    r"(?:^(?:watch|listen|video|podcast)\s*:|\s-\s*bbc\s+sounds$)",
    re.IGNORECASE,
)
_NON_ENGLISH_ARTICLE_HOSTS: frozenset[str] = frozenset({"aajtak.in", "www.aajtak.in"})


def is_article_url(url: str) -> bool:
    """Return False for non-article multimedia/iPlayer/podcast/live or non-English companion feed URLs."""
    parsed = urlparse(url.strip())
    if (parsed.hostname or "").lower() in _NON_ENGLISH_ARTICLE_HOSTS:
        return False
    return not bool(_NON_ARTICLE_PATH_RE.search(parsed.path or ""))


def is_article_title(title: str | None) -> bool:
    """Return False for video/audio/podcast clip titles that are not text news articles."""
    if not title:
        return True
    return not bool(_NON_ARTICLE_TITLE_RE.search(title.strip()))


def parse_feed_xml(raw_bytes: bytes) -> FeedParseOutcome:
    """Safely validate and parse RSS 2.0 / Atom XML bytes into normalized entries."""
    _validate_xml_payload(raw_bytes)
    parsed = feedparser.parse(raw_bytes)

    entries: list[ParsedFeedEntry] = []
    skipped_count = 0

    for idx, entry in enumerate(parsed.entries):
        raw_link = entry.get("link")
        if not isinstance(raw_link, str) or not raw_link.strip():
            logger.warning("Skipping feed entry #%d missing link", idx)
            skipped_count += 1
            continue

        try:
            normalized_link = normalize_url(raw_link)
            domain = extract_domain(normalized_link)
            if not is_article_url(normalized_link):
                logger.info("Skipping non-article feed entry #%d (%s)", idx, normalized_link)
                skipped_count += 1
                continue
            raw_title = _clean_text(entry.get("title"))
            if not is_article_title(raw_title):
                logger.info("Skipping multimedia clip entry #%d (%s)", idx, raw_title)
                skipped_count += 1
                continue
        except ValueError as exc:
            logger.warning("Skipping feed entry #%d with invalid link %r: %s", idx, raw_link, exc)
            skipped_count += 1
            continue

        title = _clean_text(entry.get("title"))
        if not title:
            logger.warning("Skipping feed entry %s missing title", normalized_link)
            skipped_count += 1
            continue

        guid = _extract_guid(entry, normalized_link)
        published_at = parse_feed_timestamp(
            entry.get("published")
            or entry.get("updated")
            or entry.get("pubDate")
            or entry.get("dc_date")
        )
        summary = _clean_text(entry.get("summary") or entry.get("description")) or None

        try:
            entries.append(
                ParsedFeedEntry(
                    title=title,
                    link=normalized_link,
                    domain=domain,
                    guid=guid,
                    published_at=published_at,
                    summary=summary,
                )
            )
        except ValueError as exc:
            logger.warning("Skipping invalid feed entry %s: %s", normalized_link, exc)
            skipped_count += 1

    return FeedParseOutcome(entries=entries, skipped_count=skipped_count)
