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
        "/api/search",
        "/api/articles/{doc_id}",
        "/api/articles/{doc_id}/background",
        "/api/evaluate",
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



