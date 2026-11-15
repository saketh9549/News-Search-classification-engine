"""Pydantic schemas and validation constants for article extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.schemas import Article, OptionalCleanStr

DEFAULT_MIN_WORD_COUNT = 80
DEFAULT_MAX_LINK_TO_TEXT_RATIO = 0.45


class ExtractionResult(BaseModel):
    """Outcome of extracting and validating an article from raw HTML."""

    accepted: bool
    article: Article | None = None
    persisted: bool = False
    rejection_reason: OptionalCleanStr = None
    word_count: int = Field(default=0, ge=0)
    link_to_text_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
