"""Read-only search service orchestrating BM25 ranking, snippets, NER, and background links."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, time, timezone
from time import perf_counter

from src.indexer.bm25_index import BM25Index, _cached_field_stem_str
from src.indexer.preprocessing import preprocess_text
from src.indexer.schemas import IndexField
from src.models.schemas import to_iso8601_utc
from src.search.classifier import predict_category
from src.search.context import (
    aggregate_entities,
    extract_entities,
    suggest_background_links,
)
from src.search.evaluation import evaluate_search
from src.search.expansion import inspect_and_expand_query
from src.search.schemas import (
    ArticleDetailResponse,
    BackgroundLink,
    CorpusOverview,
    EvaluationReport,
    FacetBucket,
    FieldMatchDetail,
    NamedEntity,
    QueryDebugInfo,
    QueryJudgment,
    SearchResponse,
    SearchResultItem,
)
from src.search.snippets import (
    build_highlighted_snippet,
    highlight_text,
    prepare_body_sentences,
)

_FIELD_LABELS: dict[IndexField, str] = {
    IndexField.TITLE: "Headline",
    IndexField.KICKER: "Kicker",
    IndexField.AUTHOR: "Author",
    IndexField.PUBLISHED_DATE: "Date",
    IndexField.BODY: "Body",
}


def parse_date_boundary(raw: str | None, *, end_of_day: bool = False) -> datetime | None:
    """Parse YYYY-MM-DD or ISO 8601 timestamps into UTC boundary datetimes."""
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None

    if len(cleaned) == 10:
        parsed_date = datetime.strptime(cleaned, "%Y-%m-%d").date()
        boundary_time = time(23, 59, 59, 999999) if end_of_day else time(0, 0, 0)
        return datetime.combine(parsed_date, boundary_time, tzinfo=timezone.utc)

    normalized = cleaned[:-1] + "+00:00" if cleaned.endswith("Z") else cleaned
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class SearchService:
    """Facade over BM25Index providing highlighted snippets, NER, and background linking."""

    _MAX_CACHED_ENTITIES = 15
    _MAX_CACHED_BACKGROUND_LINKS = 5

    def __init__(self, index: BM25Index) -> None:
        self._index = index
        self._cached_doc_count: int = -1
        self._cached_doc_ids: frozenset[str] = frozenset()
        self._entities_cache: dict[str, list[NamedEntity]] = {}
        self._category_cache: dict[str, tuple[str, float]] = {}
        self._background_cache: dict[str, list[BackgroundLink]] = {}
        self._overview_cache: CorpusOverview | None = None
        self._ensure_caches()

    @property
    def index(self) -> BM25Index:
        return self._index

    def _build_corpus_overview(self) -> CorpusOverview:
        """Build live corpus coverage metrics, filter facet counts, and sample queries."""
        raw = self._index.corpus_overview()
        raw_categories = raw.get("categories") or []
        raw_sources = raw.get("sources") or []
        raw_samples = raw.get("sample_queries") or []
        categories = [
            FacetBucket(name=str(item["name"]), count=int(item["count"]))
            for item in raw_categories
            if isinstance(item, dict)
        ]
        sources = [
            FacetBucket(name=str(item["name"]), count=int(item["count"]))
            for item in raw_sources
            if isinstance(item, dict)
        ]
        return CorpusOverview(
            doc_count=int(raw.get("doc_count") or 0),
            date_min=raw.get("date_min"),
            date_max=raw.get("date_max"),
            categories=categories,
            sources=sources,
            sample_queries=[str(q) for q in raw_samples],
        )

    def _ensure_caches(self) -> None:
        """Precompute NER entities, categories, background links, and overview when index changes."""
        reloaded = self._index.refresh_if_changed()
        doc_count = self._index.doc_count
        doc_ids = self._index.indexed_doc_ids
        if (
            not reloaded
            and self._overview_cache is not None
            and doc_count == self._cached_doc_count
            and doc_ids == self._cached_doc_ids
        ):
            return

        entities_cache: dict[str, list[NamedEntity]] = {}
        category_cache: dict[str, tuple[str, float]] = {}
        body_entities_for_bg: dict[str, list[NamedEntity]] = {}
        token_sets_by_doc: dict[str, set[str]] = {}

        docs = [
            doc
            for doc_id in sorted(doc_ids)
            if (doc := self._index.get_document(doc_id)) is not None
        ]
        for doc in docs:
            entities_cache[doc.doc_id] = extract_entities(
                doc.body_text,
                author=doc.author,
                max_entities=self._MAX_CACHED_ENTITIES,
            )
            body_entities_for_bg[doc.doc_id] = extract_entities(
                doc.body_text,
                author=None,
                max_entities=12,
            )
            category_cache[doc.doc_id] = predict_category(
                doc.title,
                doc.body_text,
                kicker=doc.kicker,
            )
            token_sets_by_doc[doc.doc_id] = set(
                preprocess_text(f"{doc.title} {doc.body_text[:1200]}")
            )
            # Pre-warm stem, phrase, and snippet sentence caches across the full document at startup
            preprocess_text(doc.body_text)
            _cached_field_stem_str(doc.title)
            _cached_field_stem_str(doc.body_text)
            prepare_body_sentences(doc.body_text)

        background_cache: dict[str, list[BackgroundLink]] = {}
        for doc in docs:
            background_cache[doc.doc_id] = suggest_background_links(
                self._index,
                doc,
                limit=self._MAX_CACHED_BACKGROUND_LINKS,
                entities_by_doc=body_entities_for_bg,
                token_sets_by_doc=token_sets_by_doc,
            )

        self._entities_cache = entities_cache
        self._category_cache = category_cache
        self._background_cache = background_cache
        self._overview_cache = self._build_corpus_overview()
        self._cached_doc_count = doc_count
        self._cached_doc_ids = doc_ids

    def get_corpus_overview(self) -> CorpusOverview:
        """Return live corpus coverage metrics, filter facet counts, and sample queries."""
        self._ensure_caches()
        assert self._overview_cache is not None
        return self._overview_cache

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        category: str | None = None,
        source: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        snippet_max_chars: int = 240,
        max_entities_per_doc: int = 8,
        background_limit: int = 3,
    ) -> SearchResponse:
        """Execute read-only multi-field BM25 query with expansion, NER, classification, and background links."""
        self._ensure_caches()
        started = perf_counter()
        has_active_filters = bool(
            (category and category.strip())
            or (source and source.strip())
            or date_from is not None
            or date_to is not None
        )
        analysis = inspect_and_expand_query(query)
        extra_index_terms = analysis.extra_index_terms or None
        expanded_phrases = analysis.expanded_phrases or None
        concept_groups = analysis.concept_groups or None
        hits = self._index.search(
            query,
            top_k=top_k,
            category=category,
            source=source,
            date_from=date_from,
            date_to=date_to,
            extra_terms=extra_index_terms,
            expansion_phrases=expanded_phrases,
            concept_groups=concept_groups,
            allow_empty_with_filters=has_active_filters,
        )

        unfiltered_total_hits: int | None = None
        if not hits and has_active_filters:
            if query.strip():
                unfiltered_hits = self._index.search(
                    query,
                    top_k=max(top_k, 10),
                    extra_terms=extra_index_terms,
                    expansion_phrases=expanded_phrases,
                    concept_groups=concept_groups,
                )
                unfiltered_total_hits = len(unfiltered_hits)
            else:
                unfiltered_total_hits = self._index.doc_count

        field_weights = self._index.field_weights
        results: list[SearchResultItem] = []
        per_doc_entities = []
        for hit in hits:
            highlighted_title = highlight_text(
                hit.title,
                query,
                extra_terms=analysis.extra_highlight_terms,
            )
            snippet = build_highlighted_snippet(
                hit.body_text,
                query,
                extra_terms=analysis.extra_highlight_terms,
                max_chars=snippet_max_chars,
            )
            if hit.doc_id in self._entities_cache and max_entities_per_doc <= self._MAX_CACHED_ENTITIES:
                entities = list(self._entities_cache[hit.doc_id][:max_entities_per_doc])
            else:
                entities = extract_entities(
                    hit.body_text,
                    author=hit.author,
                    max_entities=max_entities_per_doc,
                )
            per_doc_entities.append(entities)

            field_matches = [
                FieldMatchDetail(
                    field=field,
                    label=_FIELD_LABELS.get(field, field.value.title()),
                    weight=field_weights.get(field, 1.0),
                    score=round(hit.field_scores.get(field, 0.0), 4),
                )
                for field in hit.matched_fields
            ]
            cached_category = self._category_cache.get(hit.doc_id)
            if cached_category is not None:
                predicted_category, category_confidence = cached_category
            else:
                predicted_category, category_confidence = predict_category(
                    hit.title,
                    hit.body_text,
                    kicker=hit.kicker,
                )

            background_links: list[BackgroundLink] = []
            if background_limit > 0:
                if (
                    hit.doc_id in self._background_cache
                    and background_limit <= self._MAX_CACHED_BACKGROUND_LINKS
                ):
                    background_links = list(
                        self._background_cache[hit.doc_id][:background_limit]
                    )
                else:
                    doc = self._index.get_document(hit.doc_id)
                    if doc is not None:
                        background_links = suggest_background_links(
                            self._index,
                            doc,
                            limit=background_limit,
                        )

            results.append(
                SearchResultItem(
                    doc_id=hit.doc_id,
                    title=hit.title,
                    highlighted_title=highlighted_title,
                    source=hit.source,
                    kicker=hit.kicker,
                    author=hit.author,
                    published_date=hit.published_date,
                    canonical_url=hit.canonical_url,
                    score=hit.score,
                    matched_fields=hit.matched_fields,
                    field_matches=field_matches,
                    snippet=snippet,
                    entities=entities,
                    predicted_category=predicted_category,
                    category_confidence=category_confidence,
                    background_links=background_links,
                )
            )

        elapsed_ms = round((perf_counter() - started) * 1000.0, 2)
        top_entities = aggregate_entities(per_doc_entities, max_entities=15)
        facets = self.get_corpus_overview()

        suggestions: list[str] = []
        if not results:
            seen_sugg: set[str] = set()
            for phrases in analysis.expanded_terms.values():
                for phrase in phrases:
                    lowered = phrase.lower()
                    if lowered not in seen_sugg:
                        seen_sugg.add(lowered)
                        suggestions.append(phrase)
            for sample in facets.sample_queries:
                if sample.lower() not in seen_sugg:
                    seen_sugg.add(sample.lower())
                    suggestions.append(sample)

        query_debug = QueryDebugInfo(
            raw_query=analysis.raw_query,
            raw_tokens=analysis.raw_tokens,
            removed_stopwords=analysis.removed_stopwords,
            stemmed_tokens=analysis.stemmed_tokens,
            expanded_terms=analysis.expanded_terms,
            final_terms=analysis.final_terms,
            field_weights={f.value: w for f, w in field_weights.items()},
            execution_time_ms=elapsed_ms,
        )

        return SearchResponse(
            query=query,
            total_hits=len(results),
            filters={
                "category": category.strip() if category else None,
                "source": source.strip() if source else None,
                "date_from": to_iso8601_utc(date_from) if date_from else None,
                "date_to": to_iso8601_utc(date_to) if date_to else None,
            },
            results=results,
            top_entities=top_entities,
            facets=facets,
            query_debug=query_debug,
            unfiltered_total_hits=unfiltered_total_hits,
            suggestions=suggestions[:6],
        )

    def get_article_detail(
        self,
        doc_id: str,
        *,
        background_limit: int = 3,
    ) -> ArticleDetailResponse | None:
        """Return full article metadata, entities, and 2-3 related background stories."""
        self._ensure_caches()
        doc = self._index.get_document(doc_id)
        if doc is None:
            return None
        if doc_id in self._entities_cache:
            entities = list(self._entities_cache[doc_id])
        else:
            entities = extract_entities(
                doc.body_text,
                author=doc.author,
                max_entities=15,
            )
        if (
            background_limit > 0
            and doc_id in self._background_cache
            and background_limit <= self._MAX_CACHED_BACKGROUND_LINKS
        ):
            background = list(self._background_cache[doc_id][:background_limit])
        else:
            background = suggest_background_links(
                self._index,
                doc,
                limit=background_limit,
            )
        return ArticleDetailResponse(
            doc_id=doc.doc_id,
            title=doc.title,
            source=doc.source,
            kicker=doc.kicker,
            author=doc.author,
            published_date=doc.published_date,
            canonical_url=doc.canonical_url,
            body_text=doc.body_text,
            entities=entities,
            background_links=background,
        )

    def get_background_links(
        self,
        doc_id: str,
        *,
        limit: int = 3,
    ) -> list[BackgroundLink] | None:
        """Return suggested background stories for an indexed doc_id, or None if missing."""
        self._ensure_caches()
        doc = self._index.get_document(doc_id)
        if doc is None:
            return None
        if (
            limit > 0
            and doc_id in self._background_cache
            and limit <= self._MAX_CACHED_BACKGROUND_LINKS
        ):
            return list(self._background_cache[doc_id][:limit])
        return suggest_background_links(self._index, doc, limit=limit)

    def evaluate(
        self,
        judgments: Sequence[QueryJudgment],
        *,
        k: int = 10,
    ) -> EvaluationReport:
        """Compute P@k, R@k, MAP, nDCG@k, and MRR for a set of query judgments."""
        return evaluate_search(self._index, judgments, k=k)
