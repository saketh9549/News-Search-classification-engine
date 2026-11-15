"""Pydantic schemas for search results, NER entities, background links, and evaluation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from src.indexer.schemas import IndexField
from src.models.schemas import HttpUrlStr, NonEmptyStr, OptionalCleanStr, UtcDatetime


class EntityLabel(StrEnum):
    """TREC-style named entity categories extracted from news documents."""

    PERSON = "PERSON"
    ORG = "ORG"
    GPE = "GPE"
    EVENT = "EVENT"


class NamedEntity(BaseModel):
    """Named entity mention extracted from retrieved articles."""

    text: NonEmptyStr
    label: EntityLabel
    count: int = Field(default=1, ge=1)


class BackgroundLink(BaseModel):
    """Related background story suggested for an anchor article."""

    doc_id: NonEmptyStr
    title: NonEmptyStr
    source: OptionalCleanStr = None
    kicker: OptionalCleanStr = None
    published_date: UtcDatetime | None = None
    canonical_url: HttpUrlStr
    similarity_score: float = Field(ge=0.0)
    shared_entities: list[str] = Field(default_factory=list)


class FieldMatchDetail(BaseModel):
    """Per-field BM25 contribution breakdown showing multi-field weighting."""

    field: IndexField
    label: NonEmptyStr
    weight: float = Field(gt=0.0)
    score: float = Field(ge=0.0)


class FacetBucket(BaseModel):
    """Facet value and document count across the indexed corpus."""

    name: NonEmptyStr
    count: int = Field(ge=0)


class CorpusOverview(BaseModel):
    """Summary coverage statistics and live filter facets for the search index."""

    doc_count: int = Field(ge=0)
    date_min: UtcDatetime | None = None
    date_max: UtcDatetime | None = None
    categories: list[FacetBucket] = Field(default_factory=list)
    sources: list[FacetBucket] = Field(default_factory=list)
    sample_queries: list[str] = Field(default_factory=list)


class QueryDebugInfo(BaseModel):
    """Trace of query preprocessing, expansion, field weights, and latency."""

    raw_query: str
    raw_tokens: list[str] = Field(default_factory=list)
    removed_stopwords: list[str] = Field(default_factory=list)
    stemmed_tokens: list[str] = Field(default_factory=list)
    expanded_terms: dict[str, list[str]] = Field(default_factory=dict)
    final_terms: list[str] = Field(default_factory=list)
    field_weights: dict[str, float] = Field(default_factory=dict)
    execution_time_ms: float = Field(default=0.0, ge=0.0)


class SearchResultItem(BaseModel):
    """Search result item with highlighted text, field weights, NER, and background links."""

    doc_id: NonEmptyStr
    title: NonEmptyStr
    highlighted_title: str = ""
    source: OptionalCleanStr = None
    kicker: OptionalCleanStr = None
    author: OptionalCleanStr = None
    published_date: UtcDatetime | None = None
    canonical_url: HttpUrlStr
    score: float = Field(ge=0.0)
    matched_fields: list[IndexField] = Field(default_factory=list)
    field_matches: list[FieldMatchDetail] = Field(default_factory=list)
    snippet: str
    entities: list[NamedEntity] = Field(default_factory=list)
    predicted_category: NonEmptyStr = "world"
    category_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    background_links: list[BackgroundLink] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Top-level search response returned by the SearchService and FastAPI API."""

    query: str
    total_hits: int = Field(ge=0)
    filters: dict[str, str | None] = Field(default_factory=dict)
    results: list[SearchResultItem] = Field(default_factory=list)
    top_entities: list[NamedEntity] = Field(default_factory=list)
    facets: CorpusOverview = Field(default_factory=lambda: CorpusOverview(doc_count=0))
    query_debug: QueryDebugInfo | None = None
    unfiltered_total_hits: int | None = Field(default=None, ge=0)
    suggestions: list[str] = Field(default_factory=list)


class ArticleDetailResponse(BaseModel):
    """Full article payload paired with extracted entities and background links."""

    doc_id: NonEmptyStr
    title: NonEmptyStr
    source: OptionalCleanStr = None
    kicker: OptionalCleanStr = None
    author: OptionalCleanStr = None
    published_date: UtcDatetime | None = None
    canonical_url: HttpUrlStr
    body_text: NonEmptyStr
    entities: list[NamedEntity] = Field(default_factory=list)
    background_links: list[BackgroundLink] = Field(default_factory=list)


class QueryJudgment(BaseModel):
    """Relevance ground-truth judgment for evaluating a test query."""

    query_id: NonEmptyStr
    query: NonEmptyStr
    relevant_docs: dict[str, int] = Field(default_factory=dict)


class QueryEvalMetrics(BaseModel):
    """Per-query IR evaluation metrics."""

    query_id: NonEmptyStr
    query: NonEmptyStr
    precision_at_k: float = Field(ge=0.0, le=1.0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    average_precision: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)


class EvaluationReport(BaseModel):
    """Aggregated retrieval effectiveness report across benchmark queries."""

    k: int = Field(ge=1)
    num_queries: int = Field(ge=0)
    mean_precision_at_k: float = Field(ge=0.0, le=1.0)
    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_average_precision: float = Field(ge=0.0, le=1.0)
    mean_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    per_query: list[QueryEvalMetrics] = Field(default_factory=list)
