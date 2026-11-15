"""Pydantic schemas and constants for the RSS feed poller."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.models.schemas import (
    DomainStr,
    HttpUrlStr,
    NonEmptyStr,
    OptionalCleanStr,
    UtcDatetime,
)

DEFAULT_USER_AGENT = "NewsSearchBot/1.0 (+http://localhost/bot; contact@example.com)"
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MIN_HOST_DELAY_SECONDS = 5.0


class ParsedFeedEntry(BaseModel):
    """Normalized entry extracted from an RSS 2.0 or Atom feed."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: NonEmptyStr
    link: HttpUrlStr
    domain: DomainStr
    guid: NonEmptyStr
    published_at: UtcDatetime | None = None
    summary: OptionalCleanStr = None


class FeedParseOutcome(BaseModel):
    """Structured result from parsing an XML feed payload."""

    entries: list[ParsedFeedEntry] = Field(default_factory=list)
    skipped_count: int = Field(default=0, ge=0)


class FeedPollResult(BaseModel):
    """Outcome summary for polling a single RSS/Atom feed."""

    feed_id: int = Field(ge=1)
    feed_url: HttpUrlStr
    status: Literal["updated", "not_modified", "unchanged_hash", "error"]
    http_status: int | None = Field(default=None, ge=100, le=599)
    items_parsed: int = Field(default=0, ge=0)
    items_inserted: int = Field(default=0, ge=0)
    items_skipped: int = Field(default=0, ge=0)
    error: OptionalCleanStr = None
