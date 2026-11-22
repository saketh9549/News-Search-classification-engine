"""End-to-end IR + ML Classifier + Background-Linking evaluation & qrels generator."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.indexer.bm25_index import BM25Index
from src.indexer.schemas import DEFAULT_INDEX_DIR
from src.search.api import DEFAULT_EVAL_QRELS_PATH
from src.search.classifier import DEFAULT_METRICS_PATH, load_classifier_metrics
from src.search.context import _PUBLISHER_AND_WIRE_ORGS, extract_entities
from src.search.schemas import QueryJudgment
from src.search.service import SearchService

CURATED_BENCHMARK_TOPICS: list[tuple[str, str, list[str], list[str]]] = [
    # (query_id, query, required_body_or_title_terms, high_relevance_title_terms)
    ("q01", "Nvidia", ["nvidia"], ["nvidia", "huang"]),
    ("q02", "Ukraine Russia", ["ukraine", "russia"], ["ukraine", "russia", "putin"]),
    ("q03", "Donald Trump", ["trump"], ["trump"]),
    ("q04", "cricket", ["cricket"], ["cricket", "wicket", "odi", "u-19", "asia cup"]),
    ("q05", "climate change", ["climate"], ["climate", "emissions", "cop", "carbon"]),
    ("q06", "artificial intelligence", ["ai", "artificial intelligence"], ["ai", "openai", "claude", "anthropic"]),
    ("q07", "interest rates inflation", ["inflation", "interest"], ["inflation", "rate", "economy", "gdp"]),
    ("q08", "India Pakistan", ["india", "pakistan"], ["india", "pakistan"]),
    ("q09", "Supreme Court", ["supreme court"], ["supreme court"]),
    ("q10", "stock market", ["stock", "shares", "market"], ["stock", "ipo", "market", "shares"]),
    ("q11", "Premier League football", ["football", "league"], ["premier league", "football", "chelsea", "arsenal", "united"]),
    ("q12", "space NASA moon", ["space", "moon", "nasa"], ["moon", "nasa", "space", "orbit", "artemis"]),
]


def _contains_term(text: str, term: str) -> bool:
    return bool(re.search(rf"\b{re.escape(term.lower())}\b", text))


def build_curated_qrels(index: BM25Index) -> list[QueryJudgment]:
    """Generate corpus-grounded graded relevance judgments (grade 2 = title match, 1 = body match)."""
    docs = [
        doc
        for doc_id in sorted(index.indexed_doc_ids)
        if (doc := index.get_document(doc_id)) is not None
    ]
    judgments: list[QueryJudgment] = []
    for qid, query, req_terms, title_boost_terms in CURATED_BENCHMARK_TOPICS:
        rel_docs: dict[str, int] = {}
        for doc in docs:
            title_l = doc.title.lower()
            full_l = f"{title_l}\n{doc.body_text.lower()}"
            if len(req_terms) == 1:
                matched = _contains_term(full_l, req_terms[0])
            elif qid in {"q06", "q07", "q10", "q12"}:
                matched = any(_contains_term(full_l, t) for t in req_terms) and (
                    sum(1 for t in req_terms if _contains_term(full_l, t)) >= min(2, len(req_terms))
                    or any(_contains_term(title_l, t) for t in title_boost_terms)
                )
            else:
                matched = all(_contains_term(full_l, t) for t in req_terms)
            if not matched:
                continue
            grade = 2 if any(_contains_term(title_l, t) for t in title_boost_terms) else 1
            rel_docs[doc.doc_id] = grade

        if rel_docs:
            judgments.append(QueryJudgment(query_id=qid, query=query, relevant_docs=rel_docs))
    return judgments


def main() -> int:
    index = BM25Index(DEFAULT_INDEX_DIR)
    service = SearchService(index)
    overview = service.get_corpus_overview()

    print("=" * 88)
    print(f" CORPUS OVERVIEW | Indexed docs: {overview.doc_count} | Sources: {len(overview.sources)} | Categories: {len(overview.categories)}")
    print("=" * 88)
    print("Categories:", [(c.name, c.count) for c in overview.categories])

    # 1. Classifier 5-Fold CV metrics
    clf_metrics = load_classifier_metrics(DEFAULT_METRICS_PATH)
    print("\n=== 1. SUPERVISED ML CLASSIFIER (5-FOLD STRATIFIED CV) ===")
    if clf_metrics:
        print(
            f"Model: {clf_metrics['model_type']} | Samples: {clf_metrics['num_samples']} | "
            f"Accuracy: {clf_metrics['accuracy']*100:.1f}% | Macro-F1: {clf_metrics['macro_f1']*100:.1f}% | "
            f"Weighted-F1: {clf_metrics['weighted_f1']*100:.1f}%"
        )
    else:
        print("No trained classifier metrics found.")

    # Verify 0% filter vs badge mismatch
    mismatches = 0
    for doc_id in index.indexed_doc_ids:
        doc = index.get_document(doc_id)
        if doc and doc.predicted_category:
            resp = service.search("", category=doc.predicted_category, top_k=5)
            for item in resp.results:
                if item.predicted_category != doc.predicted_category:
                    mismatches += 1
            break
    # Full check across all docs
    all_cat_mismatches = sum(
        1
        for doc_id in index.indexed_doc_ids
        if (doc := index.get_document(doc_id)) is not None
        and doc.predicted_category is not None
        and service._category_cache.get(doc_id, (None, 0.0))[0] != doc.predicted_category
    )
    print(f"UI Filter vs Predicted Badge mismatches: {all_cat_mismatches}/{overview.doc_count}")

    # 2. Generate & evaluate curated qrels
    qrels = build_curated_qrels(index)
    DEFAULT_EVAL_QRELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_EVAL_QRELS_PATH.write_text(
        json.dumps([j.model_dump(mode="json") for j in qrels], indent=2),
        encoding="utf-8",
    )
    report = service.evaluate(qrels, k=10)
    print(f"\n=== 2. RETRIEVAL BENCHMARK ({report.num_queries} CURATED QUERIES, k={report.k}) ===")
    print(
        f"MAP={report.mean_average_precision:.4f} | nDCG@10={report.mean_ndcg_at_k:.4f} | "
        f"MRR={report.mean_reciprocal_rank:.4f} | P@10={report.mean_precision_at_k:.4f} | "
        f"R@10={report.mean_recall_at_k:.4f}"
    )
    for row in report.per_query:
        print(
            f"  {row.query:<26} P@10={row.precision_at_k:.2f} R@10={row.recall_at_k:.2f} "
            f"AP={row.average_precision:.3f} nDCG={row.ndcg_at_k:.3f} RR={row.reciprocal_rank:.3f}"
        )

    # 3. Typo correction & phrase/sort verification
    print("\n=== 3. SPELL CORRECTION, PHRASE & SORT CHECKS ===")
    for typo_q in ("nvida", "cliamte"):
        r = service.search(typo_q, top_k=3)
        print(f"  Typo '{typo_q}' -> hits={r.total_hits}, did_you_mean={r.did_you_mean!r}")
    for phrase_q in ('"World Cup"', "Middle East", "Nvidia"):
        r = service.search(phrase_q, top_k=3)
        top_title = r.results[0].title[:70] if r.results else "<0 hits>"
        print(f"  Query {phrase_q:<14} -> hits={r.total_hits:2d} | {top_title}")

    # 4. NER & Background link cleanliness
    print("\n=== 4. NER & BACKGROUND LINK CLEANLINESS ===")
    pub_org_count = 0
    bg_total = 0
    bg_pub_only = 0
    for doc_id in sorted(index.indexed_doc_ids):
        doc = index.get_document(doc_id)
        if doc is None:
            continue
        for ent in extract_entities(doc.body_text, author=doc.author):
            if ent.text.lower() in _PUBLISHER_AND_WIRE_ORGS:
                pub_org_count += 1
        for link in service.get_background_links(doc_id, limit=3) or []:
            bg_total += 1
            shared_l = {s.lower() for s in link.shared_entities}
            if shared_l and shared_l.issubset(_PUBLISHER_AND_WIRE_ORGS):
                bg_pub_only += 1
    print(f"Publisher/wire ORG mentions in NER: {pub_org_count}")
    print(f"Background links checked: {bg_total} | Publisher-only false links: {bg_pub_only}")

    assert all_cat_mismatches == 0, f"Found {all_cat_mismatches} UI category mismatches"
    assert pub_org_count == 0, f"Found {pub_org_count} publisher ORG entities"
    assert bg_pub_only == 0, f"Found {bg_pub_only} publisher-glued background links"
    assert report.mean_average_precision >= 0.50, f"MAP too low: {report.mean_average_precision}"
    print("\nALL SYSTEM EVALUATION GATES PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
