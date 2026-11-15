"""Phase 5 tests: preprocessing, multi-field BM25 ranking, incremental persistence, and CLI."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import respx

from src.indexer import (
    BM25Index,
    IndexField,
    preprocess_date,
    preprocess_text,
    tokenize_article_fields,
)
from src.models import Article
from src.pipeline import main as pipeline_main
from src.storage import get_connection, save_article
from src.storage.init_db import init_db


def _make_article(
    *,
    canonical_url: str,
    title: str,
    body_text: str,
    kicker: str | None = "world",
    author: str | None = "Staff Reporter",
    published_date: datetime | None = None,
    feed_id: int = 1,
) -> Article:
    doc_id = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
    content_hash = hashlib.sha256(body_text.encode("utf-8")).hexdigest()
    return Article(
        doc_id=doc_id,
        feed_id=feed_id,
        url=canonical_url,
        canonical_url=canonical_url,
        title=title,
        published_date=published_date or datetime(2026, 11, 14, 16, 0, 0, tzinfo=timezone.utc),
        author=author,
        kicker=kicker,
        body_text=body_text,
        content_hash=content_hash,
        fetched_at=datetime(2026, 11, 14, 17, 0, 0, tzinfo=timezone.utc),
    )


def test_text_preprocessing_stopwords_stemming_and_fields() -> None:
    tokens = preprocess_text("The economies are running renewable initiatives in the cities.")
    assert "the" not in tokens
    assert "are" not in tokens
    assert "run" in tokens
    assert "economi" in tokens
    assert "renew" in tokens

    date_tokens = preprocess_date(datetime(2026, 11, 14, 22, 0, tzinfo=timezone(timezone.utc.utcoffset(None))))
    assert date_tokens == ["2026-11-14", "2026-11", "2026"]

    article = _make_article(
        canonical_url="https://www.bbc.co.uk/news/solar-grid",
        title="Solar Grids Expanding",
        kicker="technology",
        author="Meera Nair",
        body_text="Engineers built new solar transmission networks across multiple cities.",
    )
    fields = tokenize_article_fields(article)
    assert set(fields.keys()) == set(IndexField)
    assert "solar" in fields[IndexField.TITLE]
    assert "technolog" in fields[IndexField.KICKER]
    assert "meera" in fields[IndexField.AUTHOR]
    assert "2026-11-14" in fields[IndexField.PUBLISHED_DATE]
    assert "engin" in fields[IndexField.BODY]


def test_bm25_weights_title_and_kicker_above_body(tmp_path: Path) -> None:
    index_dir = tmp_path / "search_index"
    index = BM25Index(index_dir)

    doc_title_match = _make_article(
        canonical_url="https://example.com/1",
        title="Quantum Computing Breakthrough Announced",
        kicker="science",
        body_text="Researchers published new experimental lab benchmarks for superconducting circuits.",
    )
    doc_kicker_match = _make_article(
        canonical_url="https://example.com/2",
        title="Annual Industry Roundup",
        kicker="quantum",
        body_text="Hardware startups reported steady quarterly shipments and enterprise adoption.",
    )
    doc_body_only = _make_article(
        canonical_url="https://example.com/3",
        title="Annual Industry Roundup",
        kicker="business",
        body_text="One brief mention of quantum sensors appeared near the closing conference remarks.",
    )
    doc_unrelated = _make_article(
        canonical_url="https://example.com/4",
        title="Cricket Championship Schedule",
        kicker="sport",
        body_text="National teams announced training camps ahead of the upcoming winter series.",
    )

    inserted = index.index_articles(
        [doc_title_match, doc_kicker_match, doc_body_only, doc_unrelated]
    )
    assert inserted == 4

    hits = index.search("quantum", top_k=10)
    assert [h.canonical_url for h in hits] == [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]
    assert hits[0].score > hits[1].score > hits[2].score
    assert IndexField.TITLE in hits[0].matched_fields
    assert IndexField.KICKER in hits[1].matched_fields
    assert IndexField.BODY in hits[2].matched_fields

    # Author and date field search + filters
    author_hits = index.search("Staff Reporter", category="science")
    assert len(author_hits) == 1
    assert author_hits[0].canonical_url == "https://example.com/1"

    date_hits = index.search("2026-11-14", top_k=10)
    assert len(date_hits) == 4


def test_incremental_sync_from_db_persists_without_clearing(tmp_path: Path) -> None:
    db_path = tmp_path / "news.db"
    index_dir = tmp_path / "search_index"
    init_db(db_path)

    conn = get_connection(db_path)
    try:
        art1 = _make_article(
            canonical_url="https://www.thehindu.com/news/national/a1.ece",
            title="Monsoon Forecast Released",
            kicker="national",
            author="Ananya Sen",
            published_date=datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc),
            body_text="Meteorological department analysts expect steady rainfall across coastal districts.",
            feed_id=4,
        )
        art2 = _make_article(
            canonical_url="https://www.bbc.co.uk/news/tech-a2",
            title="Open Source AI Governance",
            kicker="technology",
            author="Rohan Verma",
            published_date=datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc),
            body_text="Software maintainers debated licensing norms and evaluation transparency standards.",
            feed_id=2,
        )
        assert save_article(conn, art1) is True
        assert save_article(conn, art2) is True

        index = BM25Index(index_dir)
        first_sync = index.sync_from_db(conn)
        assert first_sync.total_in_db == 2
        assert first_sync.newly_indexed == 2
        assert first_sync.already_indexed == 0
        assert first_sync.total_indexed == 2

        # Separate index files exist on disk outside SQLite
        assert (index_dir / "manifest.json").is_file()
        assert (index_dir / "documents.json").is_file()
        assert (index_dir / "postings.json").is_file()

        # Add a third article and reload index from disk
        art3 = _make_article(
            canonical_url="https://www.thehindu.com/business/a3.ece",
            title="Port Freight Volume Surges",
            kicker="business",
            author="Priya Iyer",
            published_date=datetime(2026, 11, 14, 18, 0, tzinfo=timezone.utc),
            body_text="Container terminals recorded higher export throughput during October.",
            feed_id=5,
        )
        assert save_article(conn, art3) is True

        reloaded_index = BM25Index(index_dir)
        assert reloaded_index.doc_count == 2
        second_sync = reloaded_index.sync_from_db(conn)
        assert second_sync.total_in_db == 3
        assert second_sync.newly_indexed == 1
        assert second_sync.already_indexed == 2
        assert second_sync.total_indexed == 3

        # Search is read-only and preserves all 3 documents; source filter works via rss_feeds join
        hits = reloaded_index.search("monsoon", source="The Hindu")
        assert len(hits) == 1
        assert hits[0].source == "The Hindu"
        assert reloaded_index.doc_count == 3

        date_filtered = reloaded_index.search(
            "transparency",
            date_from=datetime(2026, 11, 13, 0, 0, tzinfo=timezone.utc),
            date_to=datetime(2026, 11, 15, 0, 0, tzinfo=timezone.utc),
        )
        assert len(date_filtered) == 1
        assert date_filtered[0].title == "Open Source AI Governance"
    finally:
        conn.close()


def test_cli_pipeline_subcommands_and_e2e_run(
    tmp_path: Path,
    capsys: object,
    monkeypatch: object,
) -> None:
    db_path = tmp_path / "cli_news.db"
    index_dir = tmp_path / "cli_index"

    assert pipeline_main(["init", "--db", str(db_path)]) == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert json.loads(out)["rss_feeds"] == 15

    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("UPDATE rss_feeds SET is_active = CASE WHEN id = 1 THEN 1 ELSE 0 END;")
        feed_url = conn.execute("SELECT url FROM rss_feeds WHERE id = 1;").fetchone()["url"]
    finally:
        conn.close()

    rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>BBC World</title>
    <item>
      <title>Arctic Shipping Corridor Opens</title>
      <link>https://www.bbc.co.uk/news/arctic-corridor</link>
      <guid>bbc-arctic-1</guid>
      <pubDate>Sat, 14 Nov 2026 16:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""
    long_body = " ".join(
        [
            "Maritime regulators and commercial shipping lines evaluated seasonal navigation safety "
            "protocols along northern routes following updated satellite ice-charting briefings. "
            "Port authorities coordinated emergency response vessels, meteorological telemetry, "
            "and crew certification standards ahead of winter cargo operations across regional hubs."
        ]
        * 3
    )
    article_html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Arctic Shipping Corridor Opens - BBC News</title>
  <meta property="og:title" content="Arctic Shipping Corridor Opens" />
  <meta property="article:published_time" content="2026-11-14T16:00:00Z" />
  <meta name="author" content="Liam Carter" />
</head>
<body>
  <article>
    <h1>Arctic Shipping Corridor Opens</h1>
    <p>{long_body}</p>
  </article>
</body>
</html>"""

    # Advance fake monotonic clock cleanly without actually sleeping 5s
    mono = [100.0]

    def fake_sleep(sec: float) -> None:
        mono[0] += sec

    monkeypatch.setattr("src.crawler.crawler.time.monotonic", lambda: mono[0])  # type: ignore[attr-defined]
    monkeypatch.setattr("src.crawler.crawler.time.sleep", fake_sleep)  # type: ignore[attr-defined]

    with respx.mock(assert_all_called=True) as mock_http:
        mock_http.get(feed_url).mock(return_value=httpx.Response(200, content=rss_xml))
        mock_http.get("https://www.bbc.co.uk/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nAllow: /\n")
        )
        mock_http.get("https://www.bbc.co.uk/news/arctic-corridor").mock(
            return_value=httpx.Response(
                200,
                text=article_html,
                headers={"content-type": "text/html"},
            )
        )

        assert (
            pipeline_main(
                [
                    "run",
                    "--db",
                    str(db_path),
                    "--index-dir",
                    str(index_dir),
                    "--limit",
                    "5",
                ]
            )
            == 0
        )

    run_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert run_output["poll"]["items_inserted"] == 1
    assert run_output["crawl_and_extract"]["extracted_persisted"] == 1
    assert run_output["index"]["total_indexed"] == 1

    assert pipeline_main(["status", "--db", str(db_path), "--index-dir", str(index_dir)]) == 0
    status_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert status_output["database"]["articles"] == 1
    assert status_output["index"]["doc_count"] == 1

    assert (
        pipeline_main(["search", "arctic shipping", "--index-dir", str(index_dir), "--k", "5"])
        == 0
    )
    search_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert len(search_output) == 1
    assert search_output[0]["title"] == "Arctic Shipping Corridor Opens"
    assert search_output[0]["source"] == "BBC News"
