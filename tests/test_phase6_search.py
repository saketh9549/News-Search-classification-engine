"""Phase 6 tests: snippets, TREC NER & background linking, IR evaluation, and FastAPI UI/API."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute

from src.indexer import BM25Index
from src.models import Article
from src.search import (
    EntityLabel,
    QueryJudgment,
    SearchService,
    average_precision,
    build_highlighted_snippet,
    create_app,
    evaluate_search,
    extract_entities,
    highlight_text,
    inspect_and_expand_query,
    ndcg_at_k,
    precision_at_k,
    predict_category,
    recall_at_k,
    reciprocal_rank,
    suggest_background_links,
)


def _article(
    *,
    url: str,
    title: str,
    body_text: str,
    kicker: str = "world",
    author: str = "Meera Nair",
    published_date: datetime | None = None,
    feed_id: int = 1,
) -> Article:
    return Article(
        doc_id=hashlib.sha256(url.encode("utf-8")).hexdigest(),
        feed_id=feed_id,
        url=url,
        canonical_url=url,
        title=title,
        published_date=published_date or datetime(2026, 11, 14, 16, 0, 0, tzinfo=timezone.utc),
        author=author,
        kicker=kicker,
        body_text=body_text,
        content_hash=hashlib.sha256(body_text.encode("utf-8")).hexdigest(),
        fetched_at=datetime(2026, 11, 14, 17, 0, 0, tzinfo=timezone.utc),
    )


def _build_corpus_index(tmp_path: Path) -> tuple[BM25Index, dict[str, Article]]:
    index = BM25Index(tmp_path / "search_index")
    articles = {
        "grid_main": _article(
            url="https://www.thehindu.com/news/national/grid-main.ece",
            title="South Asia Clean Energy Summit Approves Regional Grid Accord",
            kicker="national",
            author="Meera Nair",
            published_date=datetime(2026, 11, 14, 16, 0, 0, tzinfo=timezone.utc),
            feed_id=4,
            body_text=(
                "Ministers meeting at the South Asia Clean Energy Summit in New Delhi signed "
                "a regional transmission pact backed by the World Bank and Power Grid Corporation. "
                "Prime Minister Arjun Rao stated that solar interconnections across India will "
                "stabilize peak electricity supply throughout summer."
            ),
        ),
        "grid_bg1": _article(
            url="https://www.thehindu.com/business/grid-bg1.ece",
            title="World Bank Backs Solar Transmission Corridors in India",
            kicker="business",
            author="Rohan Deshmukh",
            published_date=datetime(2026, 11, 10, 10, 0, 0, tzinfo=timezone.utc),
            feed_id=5,
            body_text=(
                "Earlier feasibility reports prepared in New Delhi showed that World Bank loans "
                "could finance high-voltage solar transmission lines linking western grids in India."
            ),
        ),
        "grid_bg2": _article(
            url="https://www.bbc.co.uk/news/world-grid-bg2",
            title="Power Grid Corporation Plans Cross-Border Substation Upgrades",
            kicker="world",
            author="Ananya Sen",
            published_date=datetime(2026, 11, 12, 12, 0, 0, tzinfo=timezone.utc),
            feed_id=1,
            body_text=(
                "Engineers at Power Grid Corporation reviewed regional grid accord drafts ahead of "
                "the South Asia Clean Energy Summit to align tariff rules and transformer safety."
            ),
        ),
        "tech_ai": _article(
            url="https://www.bbc.co.uk/news/technology-ai-1",
            title="OpenAI and European Commission Debate AI Safety Standards",
            kicker="technology",
            author="Liam Carter",
            published_date=datetime(2026, 11, 13, 9, 0, 0, tzinfo=timezone.utc),
            feed_id=2,
            body_text=(
                "Regulators from the European Commission met OpenAI researchers in London to review "
                "model auditing rules and transparency benchmarks for enterprise software."
            ),
        ),
        "sport_cricket": _article(
            url="https://www.thehindu.com/sport/cricket-1.ece",
            title="India Wins Test Series Opener in Mumbai",
            kicker="sport",
            author="Vikram Joshi",
            published_date=datetime(2026, 11, 11, 14, 0, 0, tzinfo=timezone.utc),
            feed_id=6,
            body_text=(
                "Fast bowlers claimed eight wickets at Wankhede Stadium in Mumbai during the "
                "opening match of the Cricket Championship."
            ),
        ),
    }
    sources = {
        1: "BBC News",
        2: "BBC News",
        4: "The Hindu",
        5: "The Hindu",
        6: "The Hindu",
    }
    index.index_articles(list(articles.values()), source_by_feed_id=sources)
    return index, articles


def test_snippet_highlighting_selects_best_passage_and_escapes_html() -> None:
    body = (
        "First introductory sentence with <script>alert(1)</script> unrelated text. "
        "Second sentence explains how solar grids expand renewable transmission capacity. "
        "Closing remarks summarize quarterly budgets."
    )
    snippet = build_highlighted_snippet(body, "solar expanding", max_chars=120)
    assert "<script>" not in snippet
    assert "<mark>solar</mark>" in snippet
    assert "<mark>expand</mark>" in snippet


def test_ner_extracts_person_org_gpe_and_event() -> None:
    text = (
        "At the South Asia Clean Energy Summit in New Delhi, Prime Minister Arjun Rao "
        "and the World Bank announced new transmission funding for India."
    )
    entities = extract_entities(text, author="Meera Nair")
    by_label = {e.label: {x.text for x in entities if x.label == e.label} for e in entities}

    assert "Meera Nair" in by_label[EntityLabel.PERSON]
    assert "Arjun Rao" in by_label[EntityLabel.PERSON]
    assert "World Bank" in by_label[EntityLabel.ORG]
    assert "New Delhi" in by_label[EntityLabel.GPE]
    assert "India" in by_label[EntityLabel.GPE]
    assert "South Asia Clean Energy Summit" in by_label[EntityLabel.EVENT]


def test_background_linking_returns_related_stories_excluding_anchor(tmp_path: Path) -> None:
    index, articles = _build_corpus_index(tmp_path)
    anchor_doc = index.get_document(articles["grid_main"].doc_id)
    assert anchor_doc is not None

    links = suggest_background_links(index, anchor_doc, limit=3)
    assert 2 <= len(links) <= 3
    link_ids = [link.doc_id for link in links]
    assert anchor_doc.doc_id not in link_ids
    # Top 2 background links should be the two solar/grid background stories rather than cricket
    assert set(link_ids[:2]) == {
        articles["grid_bg1"].doc_id,
        articles["grid_bg2"].doc_id,
    }
    assert any(link.shared_entities for link in links[:2])


def test_ir_evaluation_metrics_and_report(tmp_path: Path) -> None:
    ranked = ["d1", "d2", "d3", "d4"]
    judgments_map = {"d1": 2, "d3": 1, "d5": 1}

    assert precision_at_k(ranked, judgments_map, 2) == pytest.approx(0.5)
    assert recall_at_k(ranked, judgments_map, 3) == pytest.approx(2 / 3)
    # Relevant hits at rank 1 (prec=1.0) and rank 3 (prec=2/3); 3 total relevant -> (1 + 2/3)/3
    assert average_precision(ranked, judgments_map) == pytest.approx((1.0 + 2 / 3) / 3)
    assert reciprocal_rank(ranked, judgments_map) == pytest.approx(1.0)
    assert 0.0 < ndcg_at_k(ranked, judgments_map, 3) <= 1.0

    index, articles = _build_corpus_index(tmp_path)
    report = evaluate_search(
        index,
        [
            QueryJudgment(
                query_id="q1",
                query="solar transmission grid",
                relevant_docs={
                    articles["grid_main"].doc_id: 2,
                    articles["grid_bg1"].doc_id: 1,
                    articles["grid_bg2"].doc_id: 1,
                },
            ),
            QueryJudgment(
                query_id="q2",
                query="OpenAI European Commission",
                relevant_docs={articles["tech_ai"].doc_id: 2},
            ),
        ],
        k=3,
    )
    assert report.num_queries == 2
    assert report.mean_reciprocal_rank == 1.0
    assert report.mean_average_precision > 0.8
    assert report.mean_ndcg_at_k > 0.8


def test_fastapi_routes_search_filters_ui_and_evaluation(tmp_path: Path) -> None:
    index, articles = _build_corpus_index(tmp_path)
    app = create_app(index=index)

    openapi = app.openapi()
    assert set(openapi["paths"].keys()) == {
        "/",
        "/robots.txt",
        "/health",
        "/api/corpus",
        "/api/stats",
        "/api/classifier/metrics",
        "/api/search",
        "/api/search/live",
        "/api/crawl-for-query",
        "/api/articles/{doc_id}",
        "/api/articles/{doc_id}/background",
        "/api/evaluate",
        "/api/evaluate/default",
    }

    endpoints = {
        route.path: route.endpoint
        for route in app.routes
        if isinstance(route, APIRoute)
    }

    robots_response = endpoints["/robots.txt"]()
    assert robots_response.status_code == 200
    assert robots_response.media_type == "text/plain"
    robots_body = robots_response.body.decode("utf-8")
    assert "User-agent: *" in robots_body
    assert "Allow: /" in robots_body
    assert "Disallow: /api/" in robots_body

    ui_response = endpoints["/"]()
    assert ui_response.status_code == 200
    ui_html = ui_response.body.decode("utf-8")
    for expected_marker in (
        "News Search Engine",
        "Show IR Debug Info",
        "Related / Background Stories",
        '<select id="category"',
        '<select id="source"',
        "corpus-stats",
        "try-pills",
        "&quot;",
        "&#39;",
        "safeHttpUrl",
    ):
        assert expected_marker in ui_html

    corpus_data = endpoints["/api/corpus"]()
    assert corpus_data.doc_count == 5
    assert corpus_data.date_min is not None
    assert corpus_data.date_max is not None
    assert {c.name for c in corpus_data.categories} == {"national", "business", "world", "technology", "sport"}
    assert len(corpus_data.sample_queries) >= 1

    health_data = endpoints["/health"]()
    assert health_data["status"] == "ok"
    assert health_data["doc_count"] == 5

    # Search with category & date range filters
    search_resp = endpoints["/api/search"](
        q="solar transmission",
        k=5,
        category="business",
        source="The Hindu",
        date_from="2026-11-09",
        date_to="2026-11-11",
    )
    assert search_resp.total_hits == 1
    hit = search_resp.results[0]
    assert hit.doc_id == articles["grid_bg1"].doc_id
    assert hit.source == "The Hindu"
    assert "<mark>" in hit.snippet
    assert len(search_resp.top_entities) >= 1

    # Blank query or inverted date range raises HTTPException(422)
    with pytest.raises(HTTPException) as exc_blank:
        endpoints["/api/search"](q="   ")
    assert exc_blank.value.status_code == 422

    with pytest.raises(HTTPException) as exc_dates:
        endpoints["/api/search"](q="solar", date_from="2026-11-15", date_to="2026-11-10")
    assert exc_dates.value.status_code == 422

    # Article detail & background links endpoints
    detail = endpoints["/api/articles/{doc_id}"](articles["grid_main"].doc_id)
    assert detail.doc_id == articles["grid_main"].doc_id
    assert len(detail.background_links) >= 2

    bg = endpoints["/api/articles/{doc_id}/background"](articles["grid_main"].doc_id, limit=2)
    assert len(bg) == 2

    with pytest.raises(HTTPException) as exc_missing:
        endpoints["/api/articles/{doc_id}"]("missing-doc-id")
    assert exc_missing.value.status_code == 404

    # Evaluation endpoint
    eval_report = endpoints["/api/evaluate"](
        [
            QueryJudgment(
                query_id="q1",
                query="cricket Mumbai",
                relevant_docs={articles["sport_cricket"].doc_id: 1},
            )
        ],
        k=3,
    )
    assert eval_report.mean_reciprocal_rank == 1.0
    assert eval_report.mean_ndcg_at_k == 1.0


def test_query_expansion_and_zero_result_filter_recovery(tmp_path: Path) -> None:
    index = BM25Index(tmp_path / "exp_index")
    u19_article = _article(
        url="https://www.thehindu.com/sport/cricket/india-pakistan-u19.ece",
        title="India vs Pakistan Under-19 Cricket Semifinal Set for Sharjah",
        kicker="sport",
        feed_id=6,
        body_text=(
            "India Under-19 captain led the batting lineup against Pakistan bowlers in Sharjah "
            "ahead of the Youth World Cup cricket semifinal clash."
        ),
    )
    index.index_articles([u19_article], source_by_feed_id={6: "The Hindu"})
    service = SearchService(index)

    analysis = inspect_and_expand_query("ind vs pak u19")
    assert "vs" in analysis.removed_stopwords
    assert "ind" in analysis.expanded_terms
    assert "pak" in analysis.expanded_terms
    assert "u19" in analysis.expanded_terms
    assert "india" in analysis.extra_index_terms
    assert "pakistan" in analysis.extra_index_terms
    assert "19" not in analysis.extra_index_terms

    # Abbreviated query matches full article headline and keeps compound Under-19 intact
    resp = service.search("ind vs pak u19")
    assert resp.total_hits == 1
    hit = resp.results[0]
    assert hit.doc_id == u19_article.doc_id
    assert "<mark>India</mark>" in hit.highlighted_title
    assert "<mark>Pakistan</mark>" in hit.highlighted_title
    assert "<mark>Under-19</mark>" in hit.highlighted_title
    assert hit.predicted_category == "sport"
    assert hit.category_confidence > 0.6
    assert hit.field_matches
    assert pytest.approx(sum(fm.score for fm in hit.field_matches), rel=1e-3) == hit.score
    assert resp.query_debug is not None
    assert "ind" in resp.query_debug.expanded_terms
    assert resp.query_debug.execution_time_ms >= 0.0

    # Filtering to an unmatched category triggers unfiltered recovery count & suggestions
    filtered = service.search("ind vs pak u19", category="business")
    assert filtered.total_hits == 0
    assert filtered.unfiltered_total_hits == 1
    assert len(filtered.suggestions) >= 1


def test_classifier_and_inline_background_links(tmp_path: Path) -> None:
    cat_tech, conf_tech = predict_category(
        "Semiconductor Foundry Unveils New AI Silicon Chip",
        "Major cloud technology firms ordered processor wafers for generative artificial intelligence models.",
        kicker="technology",
    )
    assert cat_tech == "technology"
    assert 0.5 <= conf_tech <= 1.0

    cat_biz, conf_biz = predict_category(
        "RBI Holds Interest Rate Steady Amid GDP Growth Forecast",
        "Stock markets rallied as banking shares and inflation data boosted investor sentiment.",
        kicker="business",
    )
    assert cat_biz == "business"
    assert 0.5 <= conf_biz <= 1.0

    index, articles = _build_corpus_index(tmp_path)
    service = SearchService(index)
    resp = service.search("South Asia Clean Energy Summit")
    assert resp.total_hits >= 1
    top = resp.results[0]
    assert top.doc_id == articles["grid_main"].doc_id
    assert len(top.background_links) >= 2
    assert any(
        fm.field.value == "title" and fm.weight == 3.0 and fm.score > 0.0
        for fm in top.field_matches
    )
    assert "<mark>Energy</mark>" in highlight_text("South Asia Clean Energy Summit", "energy")


def test_compound_tokens_quorum_and_stopword_filtering(tmp_path: Path) -> None:
    index = BM25Index(tmp_path / "quorum_index")
    docs = [
        _article(
            url="https://example.com/u19-full",
            title="India vs Pakistan Under-19 Semifinal Thriller",
            kicker="sport",
            body_text="India Under-19 batters defeated Pakistan in a close cricket match.",
        ),
        _article(
            url="https://example.com/ind-pak-senior",
            title="India and Pakistan Discuss Trade",
            kicker="world",
            body_text="Diplomats from India and Pakistan reviewed senior bilateral customs accords.",
        ),
        _article(
            url="https://example.com/noise-age-19",
            title="Youth Festival Held in New Delhi",
            kicker="national",
            body_text="A 19 year old student won the essay contest vs other schools.",
        ),
        _article(
            url="https://example.com/noise-only-india",
            title="Monsoon Flood Relief in India",
            kicker="india",
            body_text="Emergency teams coordinated shelter supplies across coastal districts.",
        ),
    ]
    index.index_articles(docs, source_by_feed_id={1: "Example News"})
    service = SearchService(index)

    # "vs" is stripped as stopword; "19" noise & 1/3 single-concept India doc are filtered out by quorum
    resp = service.search("ind vs pak u19")
    returned_urls = [r.canonical_url for r in resp.results]
    assert "https://example.com/u19-full" in returned_urls
    assert "https://example.com/noise-age-19" not in returned_urls
    assert "https://example.com/noise-only-india" not in returned_urls
    # Full 3/3 match outscores 2/3 partial match via Soft-AND coordination
    assert returned_urls[0] == "https://example.com/u19-full"


def test_ner_filters_who_pronoun_title_case_and_unrelated_background_links(
    tmp_path: Path,
) -> None:
    # Lowercase "who" and sentence-start "Who" must not trigger ORG: WHO
    prose_with_who_pronoun = (
        "Who will lead the delegation remains unclear for reporters who attended the briefing.\n"
        "Historic Delhi Architecture Exhibition\n"
        "Contemporary World Scholars Meet\n"
        "The Chief Minister greeted conservationists working for For Padam Singh and local artisans."
    )
    entities = extract_entities(prose_with_who_pronoun)
    names_by_label = {(e.text, e.label) for e in entities}
    assert ("WHO", EntityLabel.ORG) not in names_by_label
    assert ("Historic Delhi", EntityLabel.PERSON) not in names_by_label
    assert ("Chief Minister", EntityLabel.PERSON) not in names_by_label
    assert ("Contemporary World", EntityLabel.PERSON) not in names_by_label
    assert ("For Padam Singh", EntityLabel.PERSON) not in names_by_label
    assert ("Padam Singh", EntityLabel.PERSON) in names_by_label

    # Unrelated stories sharing only pronoun "who" should not produce bogus high-similarity links
    index = BM25Index(tmp_path / "clean_bg_index")
    t20_doc = _article(
        url="https://example.com/t20-cricket",
        title="T20 Bowlers Dominate Evening Match",
        kicker="sport",
        body_text="Fans who watched the stadium match cheered bowlers who bowled yorkers.",
    )
    mosque_doc = _article(
        url="https://example.com/heritage-mosque",
        title="Historic Delhi Mosque Restoration Completed",
        kicker="national",
        body_text="Architects who led masonry repairs consulted historians who studied stonework.",
    )
    index.index_articles([t20_doc, mosque_doc], source_by_feed_id={1: "Example News"})
    t20_indexed = index.get_document(t20_doc.doc_id)
    assert t20_indexed is not None
    links = suggest_background_links(index, t20_indexed, limit=3)
    assert links == []


def test_phrase_expansion_cooccurrence_and_low_weight(tmp_path: Path) -> None:
    index = BM25Index(tmp_path / "phrase_exp_index")
    direct_doc = _article(
        url="https://example.com/icc-direct",
        title="ICC Finalizes Tournament Schedule",
        kicker="sport",
        body_text="The ICC announced venues and dates for the upcoming tournament series.",
    )
    phrase_doc = _article(
        url="https://example.com/icc-phrase",
        title="Cricket Governance Update",
        kicker="sport",
        body_text="Delegates from the International Cricket Council met to review match regulations.",
    )
    isolated_doc = _article(
        url="https://example.com/icc-isolated",
        title="Council Policy Debate",
        kicker="world",
        body_text="The city council reviewed international trade tariffs and zoning bylaws.",
    )
    index.index_articles(
        [direct_doc, phrase_doc, isolated_doc],
        source_by_feed_id={1: "Example News"},
    )
    service = SearchService(index)

    resp = service.search("ICC")
    urls = [r.canonical_url for r in resp.results]
    assert "https://example.com/icc-direct" in urls
    assert "https://example.com/icc-phrase" in urls
    assert "https://example.com/icc-isolated" not in urls
    assert urls[0] == "https://example.com/icc-direct"


def test_category_normalization_and_index_rebuild(tmp_path: Path) -> None:
    from src.models import normalize_category
    from src.storage.init_db import get_connection, init_db
    from src.storage.repository import save_article

    assert normalize_category("Technology") == "technology"
    assert normalize_category("Sport Front Page") == "sport"
    assert normalize_category("World") == "world"
    assert normalize_category("Business & Economy") == "business"

    docs = [
        _article(
            url="https://example.com/tech-1",
            title="Cloud Chip Update",
            kicker="Technology",
            body_text="Semiconductor chip makers shared new datacenter benchmarks today.",
        ),
        _article(
            url="https://example.com/tech-2",
            title="AI Hardware Update",
            kicker="technology",
            body_text="Hardware startups released new inference accelerators this week.",
        ),
        _article(
            url="https://example.com/sport-1",
            title="Test Series Update",
            kicker="Sport Front Page",
            body_text="Fast bowlers dominated the opening session at the stadium today.",
        ),
    ]
    index = BM25Index(tmp_path / "cat_norm_index")
    index.index_articles(docs, source_by_feed_id={1: "Example News"})
    service = SearchService(index)

    cat_counts = {c.name: c.count for c in service.get_corpus_overview().categories}
    assert cat_counts == {"technology": 2, "sport": 1}

    filtered = service.search("update", category="Technology")
    assert filtered.total_hits == 2

    db_path = tmp_path / "rebuild.db"
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        for doc in docs:
            save_article(conn, doc)
        rebuilt = index.rebuild_from_db(conn)
        assert rebuilt.newly_indexed == 3
        assert rebuilt.total_indexed == 3
    finally:
        conn.close()


def test_ner_suppresses_false_positive_person_and_preserves_real_people() -> None:
    prose = (
        "Air Missions announced upgrades near the Congo Basin and Race Course.\n"
        "Engineers at Blue Origin reviewed the Super Heavy Booster after Rockstar Games streamed Grand Theft Auto footage.\n"
        "White House aides used Microsoft Outlook before Weightlifting Championships on Veterans Day.\n"
        "Reporting by Chris Vallance\n"
        "Produced by studio teams in London after Arjun Rao announced new findings."
    )
    entities = extract_entities(
        prose,
        author="Rachel Clun, Bernd Debusmann, Jr",
        max_entities=25,
    )
    by_label = {(e.text, e.label) for e in entities}
    person_names = {e.text for e in entities if e.label == EntityLabel.PERSON}

    for bad in (
        "Air Missions",
        "Congo Basin",
        "Race Course",
        "Chris Vallance Produced",
        "Super Heavy",
        "Super Heavy Booster",
        "Grand Theft Auto",
        "Blue Origin",
        "Rockstar Games",
        "White House",
        "Microsoft Outlook",
        "Weightlifting Championships",
        "Veterans Day",
    ):
        assert bad not in person_names

    assert ("Congo Basin", EntityLabel.GPE) in by_label
    assert ("Blue Origin", EntityLabel.ORG) in by_label
    assert ("White House", EntityLabel.ORG) in by_label
    assert ("Rockstar Games", EntityLabel.ORG) in by_label
    assert "Rachel Clun" in person_names
    assert "Bernd Debusmann" in person_names
    assert "Chris Vallance" in person_names
    assert "Arjun Rao" in person_names


def test_startup_cache_warm_search_latency(tmp_path: Path) -> None:
    docs = [
        _article(
            url=f"https://example.com/perf-{i}",
            title=f"Global Climate and Technology Briefing {i}",
            kicker="technology" if i % 2 == 0 else "world",
            author="Jane Doe",
            body_text=(
                f"Renewable energy grid report {i}. Engineers discussed solar battery storage "
                "and semiconductor supply chains alongside World Bank delegates."
            ),
        )
        for i in range(25)
    ]
    index = BM25Index(tmp_path / "perf_index")
    index.index_articles(docs, source_by_feed_id={1: "Example News"})
    service = SearchService(index)

    resp = service.search("solar battery semiconductor")
    assert resp.total_hits > 0
    assert resp.query_debug.execution_time_ms < 50.0


def test_non_article_url_filter_and_at_tracking_strip() -> None:
    from src.models import normalize_url
    from src.poller.parser import is_article_url, parse_feed_xml

    assert is_article_url("https://www.bbc.co.uk/news/articles/c123456") is True
    assert is_article_url("https://www.bbc.co.uk/news/videos/c999999") is False
    assert is_article_url("https://www.bbc.co.uk/iplayer/episode/m0012345") is False
    assert (
        normalize_url(
            "https://www.bbc.co.uk/news/world-123?at_medium=RSS&at_campaign=rss&utm_source=x"
        )
        == "https://www.bbc.co.uk/news/world-123"
    )

    rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>BBC Feed</title>
        <item>
          <title>Valid Story</title>
          <link>https://www.bbc.co.uk/news/articles/c123456?at_medium=RSS</link>
          <guid>c123456</guid>
        </item>
        <item>
          <title>Video Story</title>
          <link>https://www.bbc.co.uk/news/videos/c999999</link>
          <guid>c999999</guid>
        </item>
      </channel>
    </rss>"""
    outcome = parse_feed_xml(rss_xml)
    assert len(outcome.entries) == 1
    assert outcome.entries[0].link == "https://www.bbc.co.uk/news/articles/c123456"
    assert outcome.skipped_count == 1


def test_filter_browse_mode_index_auto_refresh_and_url_cleanup(tmp_path: Path) -> None:
    from src.models import normalize_url
    from src.poller.parser import is_article_url

    assert (
        normalize_url("https://www.aljazeera.com/news/2026/11/14/story?traffic_source=rss")
        == "https://www.aljazeera.com/news/2026/11/14/story"
    )
    assert is_article_url("https://www.aajtak.in/bihar/story/example") is False

    idx_dir = tmp_path / "live_index"
    writer_idx = BM25Index(idx_dir)
    doc1 = _article(
        url="https://www.thehindu.com/sport/story-1.ece",
        title="India Cricket Team Prepares for Test Series",
        kicker="sport",
        feed_id=6,
        body_text="Selectors met in Mumbai ahead of the international cricket tour.",
    )
    writer_idx.index_articles([doc1], source_by_feed_id={6: "The Hindu"})

    app = create_app(idx_dir)
    endpoints = {
        route.path: route.endpoint
        for route in app.routes
        if isinstance(route, APIRoute)
    }

    # Filter-only browse works with empty q when category/source is provided
    browse = endpoints["/api/search"](q="", category="sport")
    assert browse.total_hits == 1
    assert browse.results[0].doc_id == doc1.doc_id

    # Blank q with no filters still rejects with 422
    with pytest.raises(HTTPException) as exc_empty:
        endpoints["/api/search"](q="   ")
    assert exc_empty.value.status_code == 422

    # External index update on disk is automatically picked up by live SearchService
    doc2 = _article(
        url="https://www.theguardian.com/world/story-2",
        title="Global Climate Accord Signed in Geneva",
        kicker="Science & Environment",
        feed_id=13,
        body_text="Delegates adopted binding clean grid targets at the climate summit.",
    )
    writer_idx.index_articles([doc2], source_by_feed_id={13: "The Guardian"})
    # Ensure filesystem mtime advances or differs on fast filesystems by touching size/content (already larger!)
    corpus = endpoints["/api/corpus"]()
    assert corpus.doc_count == 2
    assert {c.name for c in corpus.categories} == {"sport", "science"}
    assert {s.name for s in corpus.sources} == {"The Hindu", "The Guardian"}


def test_ml_classifier_phrase_boost_spell_correction_sort_and_publisher_ner_filter(
    tmp_path: Path,
) -> None:
    """Verify ML classifier CV/persistence, phrase proximity, spell correction, sort, and publisher ORG filter."""
    from src.search.classifier import NewsCategoryClassifier

    # 1. Supervised ML classifier training, CV evaluation, persistence, and reload
    samples = [
        (f"Cricket World Cup Match {i}", f"Bowlers took wickets and batsmen scored a century in the stadium match {i}.", "sport")
        for i in range(6)
    ] + [
        (f"Semiconductor AI Chip Launch {i}", f"Software engineers built cloud GPU silicon hardware and neural models {i}.", "technology")
        for i in range(6)
    ] + [
        (f"Central Bank Inflation GDP Report {i}", f"Stock market investors reacted to quarterly revenue, banking tariffs, and earnings {i}.", "business")
        for i in range(6)
    ]
    clf = NewsCategoryClassifier.train_from_samples(samples, min_samples_per_class=4)
    assert clf.metrics["num_classes"] == 3
    assert clf.metrics["accuracy"] >= 0.8
    assert clf.metrics["macro_f1"] >= 0.8
    assert len(clf.metrics["confusion_matrix"]) == 3

    model_path = tmp_path / "clf.joblib"
    metrics_path = tmp_path / "clf_metrics.json"
    clf.save(model_path=model_path, metrics_path=metrics_path)
    reloaded = NewsCategoryClassifier.load(model_path=model_path, metrics_path=metrics_path)
    assert reloaded is not None
    pred_cat, pred_conf = reloaded.predict(
        "New Silicon AI Processor Released",
        "Cloud hardware firms ordered GPU chips for neural software models.",
    )
    assert pred_cat == "technology"
    assert 0.3 <= pred_conf <= 1.0

    # 2. Phrase proximity, quoted phrase filter, typo spell correction, latest sort, and publisher ORG filtering
    idx = BM25Index(tmp_path / "idx_v2")
    a_phrase = _article(
        url="https://example.com/world-cup-final",
        title="World Cup Final Thriller Delights Stadium Crowd",
        kicker="sport",
        feed_id=6,
        published_date=datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc),
        body_text=(
            "Fans gathered across London for the World Cup championship match. "
            "BBC and Reuters reported Nvidia sponsored broadcast analytics."
        ),
    )
    a_scattered = _article(
        url="https://example.com/weather-world-coffee-cup",
        title="Global Weather Shifts Around the World",
        kicker="world",
        feed_id=1,
        published_date=datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc),
        body_text=(
            "Extreme weather affected cities around the world while farmers held a cup of coffee. "
            "BBC and Getty Images documented the climate storm."
        ),
    )
    idx.index_articles([a_phrase, a_scattered], source_by_feed_id={1: "BBC News", 6: "The Hindu"})
    svc = SearchService(idx)

    # Exact adjacent phrase "World Cup" must rank #1 and quoted query filters out scattered match
    unquoted = svc.search("World Cup")
    assert unquoted.results[0].doc_id == a_phrase.doc_id
    quoted = svc.search('"World Cup"')
    assert [r.doc_id for r in quoted.results] == [a_phrase.doc_id]

    # Typo spell correction ("nvida" -> "nvidia")
    typo_resp = svc.search("nvida")
    assert typo_resp.total_hits == 0
    assert typo_resp.did_you_mean == "nvidia"

    # Sort by latest puts newer doc first
    latest_resp = svc.search("world", sort="latest")
    assert [r.doc_id for r in latest_resp.results] == [a_scattered.doc_id, a_phrase.doc_id]

    # Publisher/wire-agency ORGs (BBC, Reuters, Getty Images) must NEVER be emitted as ORG entities
    ents = extract_entities(a_phrase.body_text)
    ent_names_lower = {e.text.lower() for e in ents}
    assert "nvidia" in ent_names_lower
    assert "bbc" not in ent_names_lower
    assert "reuters" not in ent_names_lower
    assert "getty images" not in ent_names_lower

    # Unrelated stories sharing only BBC must NOT link as background stories
    links = svc.get_background_links(a_phrase.doc_id, limit=3) or []
    assert all(l.doc_id != a_scattered.doc_id for l in links)

    # New API endpoints work cleanly
    app = create_app(
        index=idx,
        db_path=tmp_path / "nonexistent.db",
        metrics_path=metrics_path,
        qrels_path=tmp_path / "nonexistent_qrels.json",
    )
    endpoints = {r.path: r.endpoint for r in app.routes if isinstance(r, APIRoute)}
    stats_payload = endpoints["/api/stats"]()
    assert stats_payload["indexed_doc_count"] == 2
    clf_payload = endpoints["/api/classifier/metrics"]()
    assert clf_payload["trained"] is True
    assert clf_payload["accuracy"] >= 0.8
    eval_payload = endpoints["/api/evaluate/default"](k=5)
    assert eval_payload.num_queries >= 1


def test_query_driven_live_retrieval_from_rss_and_frontier(tmp_path: Path) -> None:
    """Verify query-driven live retrieval polls RSS, matches pending links, politely crawls, and indexes."""
    import asyncio

    import httpx
    import respx

    from src.search.live_retriever import LiveQueryRetriever
    from src.storage.init_db import get_connection, init_db
    from src.storage.repository import find_matching_pending_frontier_items

    db_path = tmp_path / "live_query.db"
    idx_dir = tmp_path / "live_index"
    init_db(db_path)

    # Keep only BBC World active in this isolated test DB
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
      <title>Quantum Battery Breakthrough Announced in Tokyo</title>
      <link>https://www.bbc.co.uk/news/articles/quantum-battery-tokyo</link>
      <guid>bbc-qb-1</guid>
      <description>Physicists unveil rapid-charging solid-state quantum battery prototype.</description>
      <pubDate>Sun, 22 Nov 2026 06:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Local Gardening Fair Draws Visitors</title>
      <link>https://www.bbc.co.uk/news/articles/local-gardening-fair</link>
      <guid>bbc-garden-2</guid>
      <description>Community flower show opened over the weekend.</description>
      <pubDate>Sun, 22 Nov 2026 05:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""
    long_article_body = " ".join(
        [
            "Physicists in Tokyo announced a quantum battery breakthrough enabling rapid solid-state "
            "energy storage for electric aviation and renewable grid balancing across Asia and Europe. "
            "Independent laboratory benchmarks confirmed stable charge cycles, thermal resilience, "
            "and commercial pilot production schedules beginning early next year."
        ]
        * 3
    )
    article_html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Quantum Battery Breakthrough Announced in Tokyo - BBC News</title>
  <meta property="og:title" content="Quantum Battery Breakthrough Announced in Tokyo" />
  <meta property="article:published_time" content="2026-11-22T06:00:00Z" />
  <meta name="author" content="Kenji Sato" />
</head>
<body>
  <article>
    <h1>Quantum Battery Breakthrough Announced in Tokyo</h1>
    <p>{long_article_body}</p>
  </article>
</body>
</html>"""

    index = BM25Index(idx_dir)
    service = SearchService(index)
    assert service.search("quantum battery").total_hits == 0

    retriever = LiveQueryRetriever(service, db_path=db_path, min_domain_delay=5.0)
    mono = [500.0]
    sleeps: list[float] = []
    stages: list[str] = []

    async def fake_asleep(sec: float) -> None:
        sleeps.append(sec)
        mono[0] += sec

    async def on_progress(evt: dict[str, object]) -> None:
        stages.append(str(evt["stage"]))

    with respx.mock(assert_all_called=False) as mock_http:
        mock_http.get(feed_url).mock(return_value=httpx.Response(200, content=rss_xml))
        mock_http.get("https://www.bbc.co.uk/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nAllow: /\n")
        )
        matching_route = mock_http.get("https://www.bbc.co.uk/news/articles/quantum-battery-tokyo").mock(
            return_value=httpx.Response(200, text=article_html, headers={"content-type": "text/html"})
        )
        unrelated_route = mock_http.get("https://www.bbc.co.uk/news/articles/local-gardening-fair").mock(
            return_value=httpx.Response(200, text="<html><body>unused</body></html>", headers={"content-type": "text/html"})
        )

        async def _run() -> dict[str, object]:
            async with httpx.AsyncClient() as client:
                return await retriever.retrieve_for_query(
                    "quantum battery",
                    max_crawl=4,
                    force_poll=True,
                    enable_discovery=False,
                    client=client,
                    sleep_fn=fake_asleep,
                    monotonic_fn=lambda: mono[0],
                    progress_cb=on_progress,
                )

        summary = asyncio.run(_run())

    # Only the query-matching URL was crawled; unrelated RSS item remained untouched in pending queue!
    assert matching_route.call_count == 1
    assert unrelated_route.call_count == 0
    assert summary["feeds_polled"] == 1
    assert summary["rss_new_links"] == 2
    assert summary["matched_pending"] == 1
    assert summary["extracted_persisted"] == 1
    assert summary["newly_indexed"] == 1
    assert any(s >= 5.0 - 1e-6 for s in sleeps)
    assert "polling" in stages and "crawling" in stages and "indexing" in stages

    # Immediate search now returns the freshly crawled & permanently indexed article
    after = service.search("quantum battery")
    assert after.total_hits == 1
    assert after.results[0].title == "Quantum Battery Breakthrough Announced in Tokyo"

    # Unrelated gardening item still lives in pending frontier and matches a "gardening" query!
    conn = get_connection(db_path)
    try:
        garden_matches = find_matching_pending_frontier_items(conn, "gardening fair", limit=5)
        assert len(garden_matches) == 1
        assert garden_matches[0].rss_title == "Local Gardening Fair Draws Visitors"
    finally:
        conn.close()


def test_ner_gazetteer_financial_tech_orgs_and_byline_exclusion(tmp_path: Path) -> None:
    prose = (
        "Updated - November 20, 2026 09:29 am IST - Chennai\n"
        "By Balasubramanyam C.P.\n"
        "Reporting by Rachel Clun\n"
        "Morgan Stanley and Charles Schwab advised on the listing while Hugging Face "
        "released open weights alongside Tata Sons.\n"
        "Philanthropic funder Coefficient Giving backed safety research, CEO Dario Amodei said."
    )
    direct = extract_entities(prose, author=None, exclude_authors="Balasubramanyam C.P.", strip_bylines=True, max_entities=20)
    by_label = {(e.text, e.label) for e in direct}
    persons = {e.text for e in direct if e.label == EntityLabel.PERSON}

    for org_name in ("Morgan Stanley", "Charles Schwab", "Hugging Face", "Coefficient Giving", "Tata Sons"):
        assert (org_name, EntityLabel.ORG) in by_label
        assert org_name not in persons

    assert "Balasubramanyam C.P" not in persons
    assert "Rachel Clun" not in persons
    assert "Dario Amodei" in persons

    # SearchService must also keep journalist metadata bylines out of card entity chips
    art = _article(
        url="https://example.com/anthropic-ipo-org-test",
        title="Anthropic raises $81 billion in IPO",
        kicker="business",
        author="Balasubramanyam C.P.",
        body_text=prose,
    )
    idx = BM25Index(tmp_path / "ner_idx")
    idx.index_articles([art], source_by_feed_id={1: "The Hindu"})
    svc = SearchService(idx)
    resp = svc.search("Anthropic Morgan Stanley")
    assert resp.total_hits == 1
    card_entities = {(e.text, e.label) for e in resp.results[0].entities}
    card_persons = {e.text for e in resp.results[0].entities if e.label == EntityLabel.PERSON}
    assert ("Morgan Stanley", EntityLabel.ORG) in card_entities
    assert "Balasubramanyam C.P" not in card_persons
    assert "Balasubramanyam C.P." not in card_persons


def test_borderline_classifier_returns_top2_categories(tmp_path: Path) -> None:
    from src.search.classifier import filter_borderline_secondary, predict_category_top2

    # High-confidence prediction suppresses secondary category
    assert filter_borderline_secondary(0.72, "technology", 0.21) == (None, None)
    # Borderline (< 0.50) with substantial runner-up (>= 0.15) keeps secondary category
    assert filter_borderline_secondary(0.37, "business", 0.34) == ("business", 0.34)
    # Borderline (< 0.50) with tiny noise runner-up (< 0.15) suppresses secondary category
    assert filter_borderline_secondary(0.42, "world", 0.08) == (None, None)

    # Lexical borderline test on an IPO / AI crossover article
    cat1, conf1, cat2, conf2 = predict_category_top2(
        "Going public on AI software IPO",
        "AI software chips startup filed an IPO on public stock markets attracting investors.",
        use_ml=False,
    )
    assert cat1 in {"technology", "business"}
    if conf1 < 0.50:
        assert cat2 in {"technology", "business"}
        assert cat2 != cat1
        assert conf2 is not None and conf2 >= 0.15

    # End-to-end persistence of borderline secondary_category through BM25Index -> SearchService
    art = _article(
        url="https://example.com/borderline-ipo",
        title="Going public on AI: On Anthropic IPO",
        kicker="opinion",
        author="Editorial Desk",
        body_text="The Initial Public Offering (IPO) of Anthropic, PBC marks a milestone for AI technology and stock markets.",
    )
    idx = BM25Index(tmp_path / "borderline_idx")
    idx.index_articles(
        [art],
        source_by_feed_id={1: "The Hindu"},
        predicted_categories={art.doc_id: ("technology", 0.3713, "business", 0.3379)},
    )
    svc = SearchService(idx)
    resp = svc.search("Anthropic IPO")
    assert resp.total_hits == 1
    hit = resp.results[0]
    assert hit.predicted_category == "technology"
    assert hit.category_confidence == 0.3713
    assert hit.secondary_category == "business"
    assert hit.secondary_confidence == 0.3379


def test_snippet_normalization_cleans_datelines_and_punctuation_spacing() -> None:
    body = (
        "Updated - November 21, 2026 11:17 am IST - Chennai\n\n"
        "The Initial Public Offering ( IPO ) of Anthropic , PBC , , is a mile marker in AI , . | Photo Credit: Reuters\n\n"
        "Security officers inspected a .303 rifle near the venue ."
    )
    snippet = build_highlighted_snippet(body, "Anthropic IPO", max_chars=260)
    assert "Updated -" not in snippet
    assert "Photo Credit" not in snippet
    assert "(<mark>IPO</mark>) of <mark>Anthropic</mark>, PBC, is a mile marker in AI." in snippet
    assert "a .303 rifle near the venue." in snippet






