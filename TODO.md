# Project Plan: News Crawler & Search Engine

This plan follows the strict constraints in [rules.txt](rules%20and%20phases/rules.txt) and module boundaries in [Agents.md](rules%20and%20phases/Agents.md).

## Phase 1: Data Modeling & Storage Setup *(Approved)*
- [x] Create `pyproject.toml` (Python 3.11+, Pydantic v2, pytest) and decoupled `src/` module layout (`poller`, `crawler`, `extractor`, `indexer`, `search`, `models`, `storage`)
- [x] Implement Pydantic v2 schemas (`Feed`, `FrontierItem`, `Article`) in `src/models` with ISO 8601 UTC timestamp standardization
- [x] Implement idempotent SQLite schema creation (`rss_feeds`, `crawl_frontier`, `articles`) in `src/storage/init_db.py` using `CREATE TABLE IF NOT EXISTS` only (no `DROP` or `DELETE`)
- [x] Seed the 15 target RSS feeds into `rss_feeds` using `INSERT OR IGNORE`
- [x] Add pytest verification for idempotent initialization, 15-feed seeding, unique/foreign-key constraints, and UTC schema validation
- [x] Review and approve Phase 1 schemas (including adding optional `guid` indexed with `domain` on `crawl_frontier`)

## Phase 2: RSS Feed Ingestion & Change Detection *(Completed)*
- [x] Implement conditional HTTP feed fetching in `src/poller` using `If-None-Match` (`ETag`), `If-Modified-Since`, and `"NewsSearchBot/1.0 (+http://localhost/bot; contact@example.com)"`
- [x] Handle HTTP `304 Not Modified` without re-parsing XML payloads (updating `last_polled_at`)
- [x] Gracefully handle network timeouts, DNS/connection errors, HTTP `429`, HTTP `5xx`, and malformed XML without halting the batch
- [x] Parse RSS 2.0 / Atom entries (`title`, `link`, `pubDate`, `guid`, `summary`) with `defusedxml` + `feedparser` and strip tracking query parameters
- [x] Deduplicate feeds using SHA-256 XML `content_hash` digests and `(domain, guid)` / `url` `INSERT OR IGNORE` into `crawl_frontier`
- [ ] (Optional) Adaptive poll-interval tracking for high-volume vs. low-volume feeds
- [x] Add pytest suite with mocked HTTP responses (`tests/test_phase2_poller.py`)

## Phase 3: Web Crawler & Politeness Engine *(Completed)*
- [x] Implement `robots.txt` fetcher, parser, and per-origin TTL cache in `src/crawler` honoring `Disallow` and `Crawl-delay` directives
- [x] Enforce mandatory domain-level politeness delay (minimum 5.0 seconds between requests to the same host)
- [x] Build polite `httpx` downloader with identifiable `User-Agent`, request timeouts, HTTP `429`/5xx/timeout exponential backoff, and `crawl_frontier` status transitions (`pending`, `in_progress`, `fetched`, `failed`, `skipped`)
- [x] Add pytest suite verifying per-domain delay enforcement, `robots.txt` disallow compliance, and backoff scheduling (`tests/test_phase3_crawler.py`)

## Phase 4: Article Extraction & Content Cleaning Pipeline *(Completed)*
- [x] Implement HTML boilerplate removal in `src/extractor` (`lxml.html`) stripping navigation, ads, social widgets, and footers
- [x] Extract headline, body text, author/byline, kicker/category, canonical URL, and publication timestamp (JSON-LD, OpenGraph, meta tags, `<time>`) standardized to ISO 8601 UTC
- [x] Enforce quality gates before storage: reject items missing headline/body, under 80 words, or exceeding link-to-text density threshold (`0.45`)
- [x] Persist validated `Article` records permanently and idempotently into SQLite `articles` via `src/storage`
- [x] Add pytest suite covering news HTML extraction, boilerplate/landing-page rejection, and end-to-end poller-to-extractor integration (`tests/test_phase4_extractor.py`)

## Phase 5: Search Indexing & Multi-Field BM25 Engine *(Completed)*
- [x] Implement text preprocessing in `src/indexer` (tokenization, stop-word removal, Porter stemming, UTC date token extraction)
- [x] Build a persistent inverted search index (`BM25Index` in `data/search_index/`) strictly decoupled from SQLite persistence
- [x] Index `title`, `kicker`, `author`, `published_date`, and `body` as distinct fields
- [x] Configure Okapi BM25 ranking with higher field weights on `title` (`3.0`) and `kicker` (`2.0`) over `body` (`1.0`)
- [x] Ensure incremental indexing never drops or clears existing index collections
- [x] Add lightweight CLI runner (`src/pipeline.py`) covering Phases 1–6 (`init`, `poll`, `crawl`, `index`, `run`, `status`, `search`, `serve`)
- [x] Add pytest suite verifying multi-field BM25 scoring, incremental persistence, and CLI orchestration (`tests/test_phase5_indexer.py`)

## Phase 6: Search Interface & Query Evaluation *(Completed)*
- [x] Build REST API & interactive web UI (`FastAPI` in `src/search/api.py` + `serve` CLI command) supporting keyword queries, `published_date` range filters, and `category`/`source` filters
- [x] Return ranked results with article title, publication date, source outlet, and HTML-safe highlighted matching snippets (`<mark>`)
- [x] Implement TREC-style contextual features (`src/search/context.py`): Named Entity Recognition (`PERSON`, `ORG`, `GPE`, `EVENT`) and automatic background linking (2–3 related background stories)
- [x] Implement retrieval evaluation (`Precision@k`, `Recall@k`, `MAP`, `nDCG@k`, `MRR` in `src/search/evaluation.py` + `/api/evaluate`) and end-to-end tests (`tests/test_phase6_search.py`)
