"""Phase 4 tests: HTML boilerplate removal, metadata parsing, quality gates, and DB storage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import respx

from src.crawler import CrawlResult, PoliteCrawler
from src.extractor import extract_and_store, extract_article
from src.models import FrontierStatus
from src.poller import FeedPoller
from src.storage import get_article_by_doc_id, get_connection, get_feed_by_id
from src.storage.init_db import init_db

LONG_PARAGRAPH_1 = (
    "Officials from more than forty countries met in New Delhi on Saturday to finalize "
    "a regional grid interconnection framework aimed at balancing renewable power generation "
    "across South Asia during peak seasonal demand. Energy ministry engineers explained that "
    "cross-border high-voltage transmission corridors can cut curtailment of solar and wind "
    "electricity while reducing backup fossil fuel dispatch across metropolitan utilities."
)

LONG_PARAGRAPH_2 = (
    "Independent policy analysts noted that tariff harmonization and transparent scheduling "
    "protocols remain essential before commercial utilities commit capital to new substation "
    "upgrades. Working groups will publish draft technical standards early next quarter for "
    "public review by regulators, grid operators, and industrial consumers."
)


def _make_crawl_result(
    *,
    html: str,
    url: str = "https://www.thehindu.com/news/national/grid-pact.ece?utm_source=rss",
    final_url: str = "https://www.thehindu.com/news/national/grid-pact.ece",
    feed_id: int = 4,
) -> CrawlResult:
    return CrawlResult(
        frontier_id=1,
        feed_id=feed_id,
        url=url,
        final_url=final_url,
        domain="www.thehindu.com",
        status=FrontierStatus.FETCHED,
        http_status=200,
        html=html,
        fetched_at=datetime(2026, 11, 14, 17, 15, 0, tzinfo=timezone.utc),
    )


def test_extract_and_store_strips_boilerplate_and_persists_article(tmp_path: Path) -> None:
    db_path = tmp_path / "articles.db"
    init_db(db_path)

    raw_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <title>South Asia Grid Pact Advances | The Hindu</title>
  <link rel="canonical" href="https://www.thehindu.com/news/national/grid-pact.ece?utm_campaign=canon" />
  <meta property="og:title" content="South Asia Grid Pact Advances" />
  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "NewsArticle",
    "headline": "South Asia Grid Pact Advances",
    "datePublished": "2026-11-14T22:00:00+05:30",
    "articleSection": "national",
    "author": {{"@type": "Person", "name": "Meera Nair"}}
  }}
  </script>
</head>
<body>
  <header><nav><a href="/home">Home</a> <a href="/sport">Sport</a></nav></header>
  <div class="ad-banner">Subscribe now for unlimited access!</div>
  <article>
    <h1>South Asia Grid Pact Advances</h1>
    <div class="social-share"><a href="#share">Share on X</a></div>
    <p>{LONG_PARAGRAPH_1}</p>
    <blockquote><p>{LONG_PARAGRAPH_2}</p></blockquote>
    <div class="related-articles"><a href="/other">Related Story Here</a></div>
  </article>
  <aside class="sidebar">Most Read Widgets Here</aside>
  <footer>Copyright 2026 The Hindu</footer>
</body>
</html>
"""

    conn = get_connection(db_path)
    try:
        crawl_result = _make_crawl_result(html=raw_html)
        first = extract_and_store(conn, crawl_result)

        assert first.accepted is True
        assert first.persisted is True
        assert first.article is not None
        assert first.word_count >= 80
        assert first.link_to_text_ratio < 0.1

        article = first.article
        assert article.title == "South Asia Grid Pact Advances"
        assert article.canonical_url == "https://www.thehindu.com/news/national/grid-pact.ece"
        assert article.author == "Meera Nair"
        assert article.kicker == "national"
        assert article.published_date == datetime(2026, 11, 14, 16, 30, 0, tzinfo=timezone.utc)
        assert "Subscribe now" not in article.body_text
        assert "Copyright 2026" not in article.body_text
        assert "Related Story Here" not in article.body_text
        assert "Share on X" not in article.body_text
        assert LONG_PARAGRAPH_1 in article.body_text
        assert article.body_text.count(LONG_PARAGRAPH_2) == 1

        stored = get_article_by_doc_id(conn, article.doc_id)
        assert stored is not None
        assert stored.canonical_url == article.canonical_url
        assert stored.published_date == datetime(2026, 11, 14, 16, 30, 0, tzinfo=timezone.utc)

        # Second storage attempt with identical canonical URL is idempotent
        second = extract_and_store(conn, crawl_result)
        assert second.accepted is True
        assert second.persisted is False
        assert conn.execute("SELECT COUNT(*) FROM articles;").fetchone()[0] == 1
    finally:
        conn.close()


def test_rejects_short_article_below_80_words(tmp_path: Path) -> None:
    db_path = tmp_path / "short.db"
    init_db(db_path)

    short_html = """<html>
<head><title>Brief Update</title></head>
<body>
  <article>
    <h1>Brief Update</h1>
    <p>This short wire alert contains far fewer than eighty words and must be rejected before storage.</p>
  </article>
</body>
</html>"""

    conn = get_connection(db_path)
    try:
        res = extract_and_store(conn, _make_crawl_result(html=short_html))
        assert res.accepted is False
        assert res.persisted is False
        assert res.article is None
        assert "too short" in (res.rejection_reason or "").lower()
        assert conn.execute("SELECT COUNT(*) FROM articles;").fetchone()[0] == 0
    finally:
        conn.close()


def test_rejects_high_link_to_text_landing_page(tmp_path: Path) -> None:
    db_path = tmp_path / "links.db"
    init_db(db_path)

    link_items = " ".join(
        f'<p><a href="/story-{i}">Breaking national politics technology economy update story item {i}</a></p>'
        for i in range(15)
    )
    landing_html = f"""<html>
<head><title>News Index Front</title></head>
<body>
  <main>
    <h1>Latest Headlines</h1>
    {link_items}
  </main>
</body>
</html>"""

    conn = get_connection(db_path)
    try:
        res = extract_and_store(conn, _make_crawl_result(html=landing_html))
        assert res.accepted is False
        assert res.persisted is False
        assert res.link_to_text_ratio > 0.45
        assert "link-to-text" in (res.rejection_reason or "").lower()
        assert conn.execute("SELECT COUNT(*) FROM articles;").fetchone()[0] == 0
    finally:
        conn.close()


def test_rejects_missing_headline_or_body() -> None:
    no_title_html = f"<html><body><article><p>{LONG_PARAGRAPH_1} {LONG_PARAGRAPH_2}</p></article></body></html>"
    res_no_title = extract_article(_make_crawl_result(html=no_title_html))
    assert res_no_title.accepted is False
    assert "headline" in (res_no_title.rejection_reason or "").lower()

    no_body_html = "<html><head><title>Only Headline</title></head><body><h1>Only Headline</h1></body></html>"
    res_no_body = extract_article(_make_crawl_result(html=no_body_html))
    assert res_no_body.accepted is False
    assert "body" in (res_no_body.rejection_reason or "").lower()


def test_meta_and_feed_category_fallback_without_json_ld(tmp_path: Path) -> None:
    db_path = tmp_path / "fallback.db"
    init_db(db_path)

    meta_only_html = f"""<html>
<head>
  <title>Quarterly Trade Report - The Hindu</title>
  <meta name="author" content="By Rohan Deshmukh" />
  <meta property="article:published_time" content="2026-11-14T11:15:00Z" />
</head>
<body>
  <article>
    <h1>Quarterly Trade Report</h1>
    <p>{LONG_PARAGRAPH_1}</p>
    <p>{LONG_PARAGRAPH_2}</p>
  </article>
</body>
</html>"""

    conn = get_connection(db_path)
    try:
        # Feed 5 is The Hindu - Business (category='business')
        crawl_res = _make_crawl_result(
            html=meta_only_html,
            url="https://www.thehindu.com/business/trade-report.ece",
            final_url="https://www.thehindu.com/business/trade-report.ece",
            feed_id=5,
        )
        res = extract_and_store(conn, crawl_res)
        assert res.accepted is True
        assert res.persisted is True
        assert res.article is not None
        assert res.article.author == "Rohan Deshmukh"
        assert res.article.kicker == "business"
        assert res.article.published_date == datetime(2026, 11, 14, 11, 15, 0, tzinfo=timezone.utc)
    finally:
        conn.close()


def test_end_to_end_poller_crawler_extractor_pipeline(tmp_path: Path) -> None:
    db_path = tmp_path / "pipeline.db"
    init_db(db_path)

    rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>BBC World</title>
    <item>
      <title>Regional Grid Pact Advances</title>
      <link>https://www.bbc.co.uk/news/world-555?utm_source=rss</link>
      <guid>bbc-world-555</guid>
      <pubDate>Sat, 14 Nov 2026 16:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""
    article_html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Regional Grid Pact Advances - BBC News</title>
  <meta property="og:title" content="Regional Grid Pact Advances" />
  <meta property="article:published_time" content="2026-11-14T16:00:00Z" />
  <meta name="author" content="Ananya Sen" />
</head>
<body>
  <nav><a href="/">Home</a></nav>
  <article>
    <h1>Regional Grid Pact Advances</h1>
    <p>{LONG_PARAGRAPH_1}</p>
    <p>{LONG_PARAGRAPH_2}</p>
  </article>
  <footer>BBC Copyright</footer>
</body>
</html>"""

    mono = [100.0]

    def fake_sleep(sec: float) -> None:
        mono[0] += sec

    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("UPDATE rss_feeds SET is_active = CASE WHEN id = 1 THEN 1 ELSE 0 END;")
        feed = get_feed_by_id(conn, 1)
        assert feed is not None

        with respx.mock(assert_all_called=True) as mock_http:
            mock_http.get(feed.url).mock(
                return_value=httpx.Response(200, content=rss_xml, headers={"ETag": '"e1"'})
            )
            mock_http.get("https://www.bbc.co.uk/robots.txt").mock(
                return_value=httpx.Response(200, text="User-agent: *\nAllow: /\n")
            )
            mock_http.get("https://www.bbc.co.uk/news/world-555").mock(
                return_value=httpx.Response(
                    200,
                    text=article_html,
                    headers={"content-type": "text/html"},
                )
            )

            with httpx.Client() as client:
                poller = FeedPoller(conn, client=client, min_host_delay=0.0)
                poll_results = poller.poll_all_feeds()
                assert poll_results[0].items_inserted == 1

                crawler = PoliteCrawler(
                    conn,
                    client=client,
                    monotonic_fn=lambda: mono[0],
                    sleep_fn=fake_sleep,
                )
                crawl_results = crawler.crawl_pending(limit=10)
                assert len(crawl_results) == 1
                assert crawl_results[0].status == FrontierStatus.FETCHED

                extract_res = extract_and_store(conn, crawl_results[0])
                assert extract_res.accepted is True
                assert extract_res.persisted is True

        assert conn.execute("SELECT COUNT(*) FROM articles;").fetchone()[0] == 1
        row = conn.execute("SELECT title, kicker, author, canonical_url FROM articles;").fetchone()
        assert row["title"] == "Regional Grid Pact Advances"
        assert row["kicker"] == "world"
        assert row["author"] == "Ananya Sen"
        assert row["canonical_url"] == "https://www.bbc.co.uk/news/world-555"
    finally:
        conn.close()


def test_extract_uses_json_ld_article_body_when_ssr_dom_is_short() -> None:
    ssr_html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Aviation Operations Resume After Router Outage</title>
  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "WebPage",
    "name": "Aviation Operations Resume After Router Outage"
  }}
  </script>
  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "articleBody": "{LONG_PARAGRAPH_1} {LONG_PARAGRAPH_2}"
  }}
  </script>
</head>
<body>
  <main>
    <h1>Aviation Operations Resume After Router Outage</h1>
    <p>Brief teaser summary shown before client hydration completes on portal pages.</p>
  </main>
</body>
</html>"""
    res = extract_article(
        _make_crawl_result(
            html=ssr_html,
            url="https://timesofindia.indiatimes.com/world/us/test/articleshow/1.cms",
            final_url="https://timesofindia.indiatimes.com/world/us/test/articleshow/1.cms",
            feed_id=7,
        )
    )
    assert res.accepted is True
    assert res.word_count >= 80
    assert res.article is not None
    assert "regional grid interconnection framework" in res.article.body_text


def test_extractor_strips_boilerplate_paragraphs_and_rejects_bbc_sounds() -> None:
    """Verify promo/newsletter paragraphs are stripped and BBC Sounds podcast pages are rejected."""
    html_with_promos = f"""<!DOCTYPE html>
<html>
<head><title>Clean Energy Summit Concludes in Geneva</title></head>
<body>
  <article>
    <h1>Clean Energy Summit Concludes in Geneva</h1>
    <p>Sign up for our morning newsletter to get daily updates.</p>
    <p>{LONG_PARAGRAPH_1}</p>
    <p>Read more: Click here for ten things to watch today.</p>
    <p>Download TOI app for breaking alerts.</p>
    <p>{LONG_PARAGRAPH_2}</p>
    <p>Image source, Getty Images</p>
  </article>
</body>
</html>"""
    res = extract_article(
        _make_crawl_result(
            html=html_with_promos,
            url="https://www.theguardian.com/world/2026/nov/21/clean-energy-summit",
            final_url="https://www.theguardian.com/world/2026/nov/21/clean-energy-summit",
            feed_id=13,
        )
    )
    assert res.accepted is True
    assert res.article is not None
    body_l = res.article.body_text.lower()
    assert "sign up for our morning newsletter" not in body_l
    assert "download toi app" not in body_l
    assert "getty images" not in body_l
    assert "regional grid interconnection framework" in body_l

    podcast_html = f"""<!DOCTYPE html>
<html>
<head><title>Tech Life - Beyond silicon chips - BBC Sounds</title></head>
<body><article><p>{LONG_PARAGRAPH_1} {LONG_PARAGRAPH_2}</p></article></body>
</html>"""
    pod_res = extract_article(
        _make_crawl_result(
            html=podcast_html,
            url="https://www.bbc.co.uk/sounds/play/w3ct8jyw",
            final_url="https://www.bbc.co.uk/sounds/play/w3ct8jyw",
            feed_id=2,
        )
    )
    assert pod_res.accepted is False
    assert pod_res.rejection_reason is not None and "podcast" in pod_res.rejection_reason.lower()
