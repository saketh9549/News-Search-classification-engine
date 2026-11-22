"""Permanent SQLite storage package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.storage.init_db import (
        DEFAULT_DB_PATH,
        SEED_FEEDS,
        create_schema,
        get_connection,
        seed_feeds,
    )
    from src.storage.repository import (
        enqueue_frontier_items,
        find_matching_pending_frontier_items,
        get_active_feeds,
        get_article_by_doc_id,
        get_due_frontier_items,
        get_feed_by_id,
        reset_stale_in_progress_items,
        save_article,
        update_feed_poll_state,
        update_frontier_status,
    )

_INIT_EXPORTS = {
    "DEFAULT_DB_PATH",
    "SEED_FEEDS",
    "create_schema",
    "get_connection",
    "seed_feeds",
}
_REPO_EXPORTS = {
    "enqueue_frontier_items",
    "find_matching_pending_frontier_items",
    "get_active_feeds",
    "get_article_by_doc_id",
    "get_due_frontier_items",
    "get_feed_by_id",
    "reset_stale_in_progress_items",
    "save_article",
    "update_feed_poll_state",
    "update_frontier_status",
}

__all__ = sorted(_INIT_EXPORTS | _REPO_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _INIT_EXPORTS:
        from src.storage import init_db as init_db_module

        return getattr(init_db_module, name)
    if name in _REPO_EXPORTS:
        from src.storage import repository as repo_module

        return getattr(repo_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
