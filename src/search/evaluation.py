"""Information Retrieval evaluation metrics (Precision@k, Recall@k, MAP, nDCG@k, MRR)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from src.indexer.bm25_index import BM25Index
from src.search.schemas import EvaluationReport, QueryEvalMetrics, QueryJudgment


def precision_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_docs: Mapping[str, int],
    k: int,
) -> float:
    """Compute Precision@k for binary relevance (grade > 0)."""
    if k <= 0:
        return 0.0
    top = ranked_doc_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if relevant_docs.get(doc_id, 0) > 0)
    return hits / len(top)


def recall_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_docs: Mapping[str, int],
    k: int,
) -> float:
    """Compute Recall@k across all relevant documents (grade > 0)."""
    total_relevant = sum(1 for grade in relevant_docs.values() if grade > 0)
    if total_relevant == 0 or k <= 0:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if relevant_docs.get(doc_id, 0) > 0)
    return hits / total_relevant


def average_precision(
    ranked_doc_ids: Sequence[str],
    relevant_docs: Mapping[str, int],
) -> float:
    """Compute Average Precision (AP) across a ranked list."""
    total_relevant = sum(1 for grade in relevant_docs.values() if grade > 0)
    if total_relevant == 0:
        return 0.0

    relevant_seen = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if relevant_docs.get(doc_id, 0) > 0:
            relevant_seen += 1
            precision_sum += relevant_seen / rank
    return precision_sum / total_relevant


def ndcg_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_docs: Mapping[str, int],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@k)."""
    if k <= 0:
        return 0.0

    def _dcg(grades: Sequence[int]) -> float:
        return sum(
            ((2**rel) - 1) / math.log2(rank + 1)
            for rank, rel in enumerate(grades, start=1)
            if rel > 0
        )

    actual_grades = [max(0, relevant_docs.get(doc_id, 0)) for doc_id in ranked_doc_ids[:k]]
    ideal_grades = sorted((max(0, g) for g in relevant_docs.values() if g > 0), reverse=True)[:k]
    idcg = _dcg(ideal_grades)
    if idcg <= 0.0:
        return 0.0
    return _dcg(actual_grades) / idcg


def reciprocal_rank(
    ranked_doc_ids: Sequence[str],
    relevant_docs: Mapping[str, int],
) -> float:
    """Compute Reciprocal Rank (1 / rank of first relevant document)."""
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if relevant_docs.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def evaluate_search(
    index: BM25Index,
    judgments: Sequence[QueryJudgment],
    *,
    k: int = 10,
) -> EvaluationReport:
    """Evaluate a BM25Index against ground-truth query judgments."""
    if not judgments:
        return EvaluationReport(
            k=k,
            num_queries=0,
            mean_precision_at_k=0.0,
            mean_recall_at_k=0.0,
            mean_average_precision=0.0,
            mean_ndcg_at_k=0.0,
            mean_reciprocal_rank=0.0,
            per_query=[],
        )

    per_query: list[QueryEvalMetrics] = []
    for item in judgments:
        hits = index.search(item.query, top_k=max(k, index.doc_count))
        ranked_ids = [hit.doc_id for hit in hits]
        per_query.append(
            QueryEvalMetrics(
                query_id=item.query_id,
                query=item.query,
                precision_at_k=round(precision_at_k(ranked_ids, item.relevant_docs, k), 6),
                recall_at_k=round(recall_at_k(ranked_ids, item.relevant_docs, k), 6),
                average_precision=round(average_precision(ranked_ids, item.relevant_docs), 6),
                ndcg_at_k=round(ndcg_at_k(ranked_ids, item.relevant_docs, k), 6),
                reciprocal_rank=round(reciprocal_rank(ranked_ids, item.relevant_docs), 6),
            )
        )

    n = len(per_query)
    return EvaluationReport(
        k=k,
        num_queries=n,
        mean_precision_at_k=round(sum(m.precision_at_k for m in per_query) / n, 6),
        mean_recall_at_k=round(sum(m.recall_at_k for m in per_query) / n, 6),
        mean_average_precision=round(sum(m.average_precision for m in per_query) / n, 6),
        mean_ndcg_at_k=round(sum(m.ndcg_at_k for m in per_query) / n, 6),
        mean_reciprocal_rank=round(sum(m.reciprocal_rank for m in per_query) / n, 6),
        per_query=per_query,
    )
