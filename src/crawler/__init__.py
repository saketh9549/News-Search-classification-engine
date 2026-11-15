"""Crawler & Politeness package: robots.txt compliance, rate limiting, and HTML fetching."""

from src.crawler.crawler import PoliteCrawler, compute_backoff_seconds
from src.crawler.robots import RobotsCache, RobotsFetchError
from src.crawler.schemas import (
    DEFAULT_BASE_BACKOFF_SECONDS,
    DEFAULT_MAX_BACKOFF_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MIN_DOMAIN_DELAY_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    CrawlResult,
    RobotsDecision,
)

__all__ = [
    "CrawlResult",
    "DEFAULT_BASE_BACKOFF_SECONDS",
    "DEFAULT_MAX_BACKOFF_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MIN_DOMAIN_DELAY_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_USER_AGENT",
    "PoliteCrawler",
    "RobotsCache",
    "RobotsDecision",
    "RobotsFetchError",
    "compute_backoff_seconds",
]
