"""Feed Poller package: RSS/Atom monitoring, conditional HTTP, and frontier seeding."""

from src.poller.parser import FeedParseError, compute_content_hash, parse_feed_timestamp, parse_feed_xml
from src.poller.poller import FeedPoller, poll_all_feeds
from src.poller.schemas import (
    DEFAULT_MIN_HOST_DELAY_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    FeedParseOutcome,
    FeedPollResult,
    ParsedFeedEntry,
)

__all__ = [
    "DEFAULT_MIN_HOST_DELAY_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_USER_AGENT",
    "FeedParseError",
    "FeedParseOutcome",
    "FeedPollResult",
    "FeedPoller",
    "ParsedFeedEntry",
    "compute_content_hash",
    "parse_feed_timestamp",
    "parse_feed_xml",
    "poll_all_feeds",
]
