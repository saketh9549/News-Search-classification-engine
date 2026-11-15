"""Search & Ranking package: query execution, snippets, NER, classification, background linking, and API."""

from src.search.api import app, create_app
from src.search.classifier import predict_category
from src.search.context import aggregate_entities, extract_entities, suggest_background_links
from src.search.evaluation import (
    average_precision,
    evaluate_search,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.search.expansion import QueryAnalysis, inspect_and_expand_query
from src.search.schemas import (
    ArticleDetailResponse,
    BackgroundLink,
    CorpusOverview,
    EntityLabel,
    EvaluationReport,
    FacetBucket,
    FieldMatchDetail,
    NamedEntity,
    QueryDebugInfo,
    QueryEvalMetrics,
    QueryJudgment,
    SearchResponse,
    SearchResultItem,
)
from src.search.service import SearchService, parse_date_boundary
from src.search.snippets import build_highlighted_snippet, highlight_text

__all__ = [
    "ArticleDetailResponse",
    "BackgroundLink",
    "CorpusOverview",
    "EntityLabel",
    "EvaluationReport",
    "FacetBucket",
    "FieldMatchDetail",
    "NamedEntity",
    "QueryAnalysis",
    "QueryDebugInfo",
    "QueryEvalMetrics",
    "QueryJudgment",
    "SearchResponse",
    "SearchResultItem",
    "SearchService",
    "aggregate_entities",
    "app",
    "average_precision",
    "build_highlighted_snippet",
    "create_app",
    "evaluate_search",
    "extract_entities",
    "highlight_text",
    "inspect_and_expand_query",
    "ndcg_at_k",
    "parse_date_boundary",
    "precision_at_k",
    "predict_category",
    "recall_at_k",
    "reciprocal_rank",
    "suggest_background_links",
]
