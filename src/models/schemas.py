"""Pydantic v2 domain schemas shared across pipeline modules."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    TypeAdapter,
)

_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)
_CATEGORY_WS_RE = re.compile(r"\s+")
_CATEGORY_ALIASES: dict[str, str] = {
    "sports": "sport",
    "sport front page": "sport",
    "cricket": "sport",
    "football": "sport",
    "tennis": "sport",
    "athletics": "sport",
    "formula 1": "sport",
    "tech": "technology",
    "science & technology": "technology",
    "world news": "world",
    "international": "world",
    "india news": "india",
    "national news": "national",
    "business & economy": "business",
    "industry": "business",
    "markets": "business",
    "economy": "business",
    "companies": "business",
    "finance": "business",
    "science & environment": "science",
    "environment": "science",
    "climate": "science",
    "entertainment & arts": "entertainment",
    "arts": "entertainment",
    "bbc indepth": "world",
    "bbc verify": "world",
    "top_stories": "world",
    "top stories": "world",
    "news": "world",
    "jobs": "business",
    "west bengal": "india",
    "keralam": "india",
    "northern ireland": "uk",
    "wales": "uk",
    "scotland": "uk",
    "england": "uk",
}


CANONICAL_CATEGORIES: tuple[str, ...] = (
    "world",
    "india",
    "business",
    "technology",
    "sport",
    "science_health",
    "politics",
    "entertainment",
)

_CANONICAL_CATEGORY_MAP: dict[str, str] = {
    "world": "world",
    "international": "world",
    "global": "world",
    "europe": "world",
    "asia": "world",
    "africa": "world",
    "middle east": "world",
    "us & canada": "world",
    "latin america": "world",
    "uk": "world",
    "england": "world",
    "scotland": "world",
    "wales": "world",
    "northern ireland": "world",
    "world news": "world",
    "india": "india",
    "india news": "india",
    "states": "india",
    "national": "india",
    "national news": "india",
    "west bengal": "india",
    "keralam": "india",
    "cities": "india",
    "business": "business",
    "business & economy": "business",
    "economy": "business",
    "markets": "business",
    "finance": "business",
    "industry": "business",
    "companies": "business",
    "jobs": "business",
    "money": "business",
    "technology": "technology",
    "tech": "technology",
    "ai": "technology",
    "computing": "technology",
    "gadgets": "technology",
    "sport": "sport",
    "sports": "sport",
    "sport front page": "sport",
    "cricket": "sport",
    "football": "sport",
    "tennis": "sport",
    "athletics": "sport",
    "formula 1": "sport",
    "badminton": "sport",
    "hockey": "sport",
    "olympics": "sport",
    "science_health": "science_health",
    "science": "science_health",
    "health": "science_health",
    "science & environment": "science_health",
    "science & technology": "science_health",
    "sci-tech": "science_health",
    "environment": "science_health",
    "climate": "science_health",
    "medicine": "science_health",
    "space": "science_health",
    "politics": "politics",
    "elections": "politics",
    "policy": "politics",
    "government": "politics",
    "parliament": "politics",
    "us politics": "politics",
    "uk politics": "politics",
    "entertainment": "entertainment",
    "culture": "entertainment",
    "arts": "entertainment",
    "entertainment & arts": "entertainment",
    "lifestyle": "entertainment",
    "movies": "entertainment",
    "music": "entertainment",
    "books": "entertainment",
    "film": "entertainment",
    "tv": "entertainment",
    "stage": "entertainment",
    "television & radio": "entertainment",
    "art and design": "entertainment",
}

GENERIC_FEED_CATEGORIES: frozenset[str] = frozenset(
    {"top_stories", "top stories", "news", "home", "bbc indepth", "bbc verify", "latest"}
)


def normalize_category(raw: str | None) -> str | None:
    """Normalize article category/kicker labels to canonical lowercase slugs."""
    if raw is None:
        return None
    cleaned = _CATEGORY_WS_RE.sub(" ", raw).strip().lower()
    if not cleaned:
        return None
    return _CATEGORY_ALIASES.get(cleaned, cleaned)


def map_to_canonical_category(raw: str | None) -> str | None:
    """Map any RSS section or HTML kicker into the 8-class canonical news taxonomy."""
    if raw is None:
        return None
    cleaned = _CATEGORY_WS_RE.sub(" ", raw).strip().lower()
    if not cleaned or cleaned in GENERIC_FEED_CATEGORIES:
        return None
    if cleaned in _CANONICAL_CATEGORY_MAP:
        return _CANONICAL_CATEGORY_MAP[cleaned]
    normalized = normalize_category(cleaned)
    if normalized and normalized in _CANONICAL_CATEGORY_MAP:
        return _CANONICAL_CATEGORY_MAP[normalized]
    return None


def utc_now() -> datetime:
    """Return the current timestamp in UTC."""
    return datetime.now(timezone.utc)


def to_iso8601_utc(value: datetime) -> str:
    """Format a timezone-aware datetime as canonical ISO 8601 UTC string."""
    utc_dt = _ensure_utc_datetime(value)
    return utc_dt.isoformat().replace("+00:00", "Z")


def _ensure_utc_datetime(value: datetime) -> datetime:
    """Require timezone awareness and normalize datetime to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include timezone information (ISO 8601 UTC or offset)")
    return value.astimezone(timezone.utc)


def _validate_http_url(value: str) -> str:
    """Validate that a URL string uses http/https with a valid host."""
    cleaned = value.strip()
    _HTTP_URL_ADAPTER.validate_python(cleaned)
    return cleaned


def _require_non_empty(value: str) -> str:
    """Ensure a string field is non-empty after trimming whitespace."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Field must not be empty or whitespace-only")
    return cleaned


def _normalize_domain(value: str) -> str:
    """Validate and lowercase a domain hostname."""
    return _require_non_empty(value).lower()


def _empty_to_none(value: str | None) -> str | None:
    """Convert empty or whitespace-only optional strings to None."""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


UtcDatetime = Annotated[
    datetime,
    AfterValidator(_ensure_utc_datetime),
    PlainSerializer(to_iso8601_utc, return_type=str, when_used="json"),
]
HttpUrlStr = Annotated[str, AfterValidator(_validate_http_url)]
NonEmptyStr = Annotated[str, AfterValidator(_require_non_empty)]
DomainStr = Annotated[str, AfterValidator(_normalize_domain)]
OptionalCleanStr = Annotated[str | None, AfterValidator(_empty_to_none)]


class FrontierStatus(StrEnum):
    """Lifecycle states for URLs in the crawl frontier."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    FETCHED = "fetched"
    FAILED = "failed"
    SKIPPED = "skipped"


class Feed(BaseModel):
    """RSS/Atom feed registry entry stored in ``rss_feeds``."""

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    id: int | None = Field(default=None, ge=1)
    name: NonEmptyStr
    source: NonEmptyStr
    category: NonEmptyStr
    url: HttpUrlStr
    etag: OptionalCleanStr = None
    last_modified: OptionalCleanStr = None
    content_hash: OptionalCleanStr = None
    last_polled_at: UtcDatetime | None = None
    is_active: bool = True


class FrontierItem(BaseModel):
    """URL queue record stored in ``crawl_frontier``."""

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    id: int | None = Field(default=None, ge=1)
    feed_id: int = Field(ge=1)
    url: HttpUrlStr
    guid: OptionalCleanStr = None
    domain: DomainStr
    status: FrontierStatus = FrontierStatus.PENDING
    retry_count: int = Field(default=0, ge=0)
    next_attempt_at: UtcDatetime | None = None
    last_http_status: int | None = Field(default=None, ge=100, le=599)
    discovered_at: UtcDatetime = Field(default_factory=utc_now)
    rss_title: OptionalCleanStr = None
    rss_summary: OptionalCleanStr = None
    rss_published_at: UtcDatetime | None = None


class Article(BaseModel):
    """Cleaned, validated article record stored in ``articles``."""

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    doc_id: NonEmptyStr
    feed_id: int = Field(ge=1)
    url: HttpUrlStr
    canonical_url: HttpUrlStr
    title: NonEmptyStr
    published_date: UtcDatetime | None = None
    author: OptionalCleanStr = None
    kicker: OptionalCleanStr = None
    body_text: NonEmptyStr
    content_hash: NonEmptyStr
    fetched_at: UtcDatetime = Field(default_factory=utc_now)
