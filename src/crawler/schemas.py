"""Pydantic schemas and configuration constants for the polite web crawler."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.schemas import (
    DomainStr,
    FrontierStatus,
    HttpUrlStr,
    OptionalCleanStr,
    UtcDatetime,
)

DEFAULT_USER_AGENT = "NewsSearchBot/1.0 (+http://localhost/bot; contact@example.com)"
DEFAULT_MIN_DOMAIN_DELAY_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_BACKOFF_SECONDS = 60.0
DEFAULT_MAX_BACKOFF_SECONDS = 3600.0
DEFAULT_ROBOTS_TTL_SECONDS = 3600.0


class RobotsDecision(BaseModel):
    """Result of evaluating robots.txt for a candidate URL."""

    allowed: bool
    crawl_delay: float | None = Field(default=None, ge=0.0)
    reason: OptionalCleanStr = None


class CrawlResult(BaseModel):
    """Structured output produced by the crawler for a frontier item."""

    frontier_id: int = Field(ge=1)
    feed_id: int = Field(ge=1)
    url: HttpUrlStr
    final_url: HttpUrlStr
    domain: DomainStr
    status: FrontierStatus
    http_status: int | None = Field(default=None, ge=100, le=599)
    html: str | None = None
    fetched_at: UtcDatetime
    error: OptionalCleanStr = None
