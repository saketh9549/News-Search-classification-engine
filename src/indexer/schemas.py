"""Pydantic schemas and configuration constants for multi-field BM25 indexing."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from src.models.schemas import HttpUrlStr, NonEmptyStr, OptionalCleanStr, UtcDatetime

DEFAULT_INDEX_DIR = Path("data/search_index")
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75


class IndexField(StrEnum):
    """Distinct searchable fields required for multi-field indexing."""

    TITLE = "title"
    KICKER = "kicker"
    AUTHOR = "author"
    PUBLISHED_DATE = "published_date"
    BODY = "body"


DEFAULT_FIELD_WEIGHTS: dict[IndexField, float] = {
    IndexField.TITLE: 3.0,
    IndexField.KICKER: 2.0,
    IndexField.AUTHOR: 1.5,
    IndexField.PUBLISHED_DATE: 1.0,
    IndexField.BODY: 1.0,
}


class IndexedDocument(BaseModel):
    """Document metadata persisted inside the standalone search index."""

    doc_id: NonEmptyStr
    feed_id: int = Field(ge=1)
    url: HttpUrlStr
    canonical_url: HttpUrlStr
    title: NonEmptyStr
    kicker: OptionalCleanStr = None
    author: OptionalCleanStr = None
    source: OptionalCleanStr = None
    published_date: UtcDatetime | None = None
    body_text: NonEmptyStr
    field_lengths: dict[IndexField, int] = Field(default_factory=dict)


class IndexSyncResult(BaseModel):
    """Summary returned after an incremental indexing run."""

    total_in_db: int = Field(default=0, ge=0)
    newly_indexed: int = Field(default=0, ge=0)
    already_indexed: int = Field(default=0, ge=0)
    total_indexed: int = Field(default=0, ge=0)
    index_dir: str


class SearchHit(BaseModel):
    """Ranked multi-field BM25 search result returned by the index."""

    doc_id: NonEmptyStr
    feed_id: int = Field(ge=1)
    url: HttpUrlStr
    canonical_url: HttpUrlStr
    title: NonEmptyStr
    kicker: OptionalCleanStr = None
    author: OptionalCleanStr = None
    source: OptionalCleanStr = None
    published_date: UtcDatetime | None = None
    body_text: NonEmptyStr
    score: float = Field(ge=0.0)
    matched_fields: list[IndexField] = Field(default_factory=list)
    field_scores: dict[IndexField, float] = Field(default_factory=dict)
