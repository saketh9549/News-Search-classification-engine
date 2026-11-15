"""Benchmark and verification suite across 15 diverse queries against the live BM25 index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.indexer.bm25_index import BM25Index, DEFAULT_INDEX_DIR
from src.search.schemas import EntityLabel
from src.search.service import SearchService

BENCHMARK_QUERIES: tuple[tuple[str, str], ...] = (
    ("ICC", "acronym"),
    ("UN", "acronym"),
    ("NATO", "acronym"),
    ("AI", "acronym"),
    ("WHO", "acronym"),
    ("ind vs pak u19", "multi-word-event"),
    ("climate change", "multi-word-topic"),
    ("World Cup", "multi-word-event"),
    ("interest rates", "multi-word-topic"),
    ("Middle East", "multi-word-entity"),
    ("Prime Minister", "multi-word-entity"),
    ("technology", "section"),
    ("India", "entity-geo"),
    ("mars colony 2099", "out-of-corpus"),
    ("xyzzy nonexistent topic", "out-of-corpus"),
)

BANNED_PERSON_FALSE_POSITIVES: frozenset[str] = frozenset(
    {
        "Air Missions",
        "Congo Basin",
        "Race Course",
        "Chris Vallance Produced",
        "Chris Vallance Producer",
        "Andy Rankin Many",
        "Joseph Vijay Questions",
        "Rachita Prasad Related",
        "Henry Moore Get",
        "Asset Recovery",
        "Conservation Education",
        "Comparative Zoology",
        "Super Heavy",
        "Super Heavy Booster",
        "Grand Theft Auto",
        "Blue Origin",
        "Bluest Sky",
        "Delta Air Lines",
        "Royal Mail",
        "Royal Aeronautical Society",
        "Rockstar Games",
        "White House",
        "Rajya Sabha",
        "Bhagavad Gita",
        "Peruvian Pope",
        "Peruvian Catholic",
        "Many Ugandans",
        "Others Democrats",
        "Unknown Soldier",
        "Veterans Day",
        "Weightlifting Championships",
        "Winter Paralympics",
        "Microsoft Outlook",
        "Microsoft Teams",
        "Exchange Online",
        "Cool Calc",
        "Cruise Lines",
        "Small Business Charter",
        "Aviation Administration",
        "Homeland Security",
        "Forces Command",
        "Protection Office",
        "Resistance Army",
        "Resistance Movement",
        "Operation Purgatory",
        "Individual Neutral Athlete",
        "Lloyd Law College",
        "Rajaratnam School",
        "Grace Center",
        "Chiclayo Cathedral",
        "Phoenix Publishing House",
        "Standard Book Numbers",
        "Tech Life",
        "Christian Life",
        "Fairy Maiden",
        "One Part Woman",
        "Moon Building",
        "Mississippi River",
        "Boca Chica",
        "Latin America",
        "Jantar Mantar",
        "Ramlila Maidan",
        "Getty Images",
    }
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 15-query precision and latency verification.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--max-warm-ms", type=float, default=25.0)
    args = parser.parse_args()

    t0 = perf_counter()
    index = BM25Index(args.index_dir)
    service = SearchService(index)
    startup_ms = round((perf_counter() - t0) * 1000.0, 2)

    overview = service.get_corpus_overview()
    print(
        f"Startup warmup: {startup_ms:.2f} ms | Indexed docs: {overview.doc_count} | "
        f"Window: {overview.date_min} .. {overview.date_max}"
    )
    print("Categories:", [(c.name, c.count) for c in overview.categories])
    print("-" * 100)
    print(f"{'Query':<26} {'Kind':<18} {'Hits':>4} {'Time(ms)':>9}  Top Hit")
    print("-" * 100)

    latencies: list[float] = []
    icc_titles: list[str] = []
    out_of_corpus_hits = 0

    for query, kind in BENCHMARK_QUERIES:
        q_start = perf_counter()
        resp = service.search(query, top_k=5)
        wall_ms = round((perf_counter() - q_start) * 1000.0, 2)
        latencies.append(wall_ms)

        if resp.results:
            top = resp.results[0]
            top_desc = f"{top.title[:48]} (score={top.score:.2f}, cat={top.predicted_category})"
        else:
            top_desc = "<no hits>"
        print(f"{query:<26} {kind:<18} {resp.total_hits:>4} {wall_ms:>9.2f}  {top_desc}")

        if query == "ICC":
            icc_titles = [r.title for r in resp.results]
            for hit in resp.results:
                doc = index.get_document(hit.doc_id)
                assert doc is not None
                combined = f"{doc.title} {doc.body_text} {doc.kicker or ''}".lower()
                assert (
                    "icc" in combined or "international cricket council" in combined
                ), f"Unexpected polluted ICC hit: {hit.title}"
        if kind == "out-of-corpus":
            out_of_corpus_hits += resp.total_hits

    print("-" * 100)
    avg_ms = sum(latencies) / len(latencies)
    max_ms = max(latencies)
    print(f"Warm query latency: avg={avg_ms:.2f} ms, max={max_ms:.2f} ms (limit={args.max_warm_ms:.2f} ms)")

    # Verify normalized categories (no mixed-case duplicates)
    cat_names = [c.name for c in overview.categories]
    assert cat_names == [c.lower() for c in cat_names], f"Non-lowercase categories found: {cat_names}"
    assert len(cat_names) == len(set(cat_names)), f"Duplicate category names found: {cat_names}"

    # Verify corpus expanded beyond original 42 articles
    assert overview.doc_count > 42, f"Expected > 42 indexed docs, got {overview.doc_count}"

    # Verify zero banned PERSON false positives across the full indexed corpus
    bad_hits: list[tuple[str, str]] = []
    for doc_id in sorted(index.indexed_doc_ids):
        detail = service.get_article_detail(doc_id)
        assert detail is not None
        for ent in detail.entities:
            if ent.label == EntityLabel.PERSON and ent.text in BANNED_PERSON_FALSE_POSITIVES:
                bad_hits.append((detail.title, ent.text))
    assert not bad_hits, f"False-positive PERSON entities detected: {bad_hits}"

    assert avg_ms < args.max_warm_ms, f"Average latency {avg_ms:.2f} ms exceeds {args.max_warm_ms:.2f} ms"
    assert max_ms < args.max_warm_ms, f"Max latency {max_ms:.2f} ms exceeds {args.max_warm_ms:.2f} ms"
    assert out_of_corpus_hits == 0, f"Out-of-corpus queries returned {out_of_corpus_hits} hits"

    print(
        f"ALL CHECKS PASSED: docs={overview.doc_count}, icc_hits={len(icc_titles)}, "
        f"bad_person_tags=0, avg_ms={avg_ms:.2f}, max_ms={max_ms:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
