# Agent Architecture: News Crawler & Search Engine

## Overview

The application follows a decoupled pipeline where distinct agents handle feed monitoring, rate-limited crawling, boilerplate extraction, search indexing, and query ranking.

```
[RSS Feeds] --> (1. Feed Poller Agent)
                        |
                        | new article URLs
                        v
                  [URL Frontier]
                        |
                        v
          (2. Crawler & Politeness Agent)
                        |
                        | raw HTML
                        v
          (3. Article Extractor Agent)
                        |
                        | clean structured JSON
                        v
[Permanent Storage] --> (4. Indexer Agent) --> [Search Index]
                                                    ^
                                                    | query
                                      (5. Search & Ranking Agent)
```

---

### 1. Feed Poller Agent

- **Role**: Continuously monitor the 15 RSS feeds for newly published articles.
- **Responsibilities**:
  - Fetch XML feeds using `ETag` and `Last-Modified` headers.
  - Handle HTTP 304 (Not Modified) without re-parsing.
  - Parse entries and extract GUIDs and canonical URLs.
  - Deduplicate using XML hash digests and GUIDs before inserting into the URL frontier.
  - Keep all frontier queue operations idempotent.

### 2. Crawler & Politeness Agent

- **Role**: Fetch raw HTML for URLs in the frontier without overloading any host.
- **Responsibilities**:
  - Fetch and respect `robots.txt` before crawling any URL.
  - Enforce a delay of at least 5-10 seconds between requests to the same host.
  - Send an explicit, identifiable User-Agent header.
  - Handle timeouts, HTTP 429, and connection resets with exponential backoff.

### 3. Article Extractor Agent

- **Role**: Turn raw HTML into clean, validated article records.
- **Responsibilities**:
  - Strip boilerplate (navigation, ads, social widgets, footers) using `trafilatura` or `readability`.
  - Extract headline, body, author, section tags, and publication timestamp.
  - Standardize timestamps to ISO 8601 UTC.
  - Reject records missing a headline or body, under 80 words, or with a high link-to-text ratio.

### 4. Indexer Agent

- **Role**: Persist validated records and keep the search index in sync.
- **Responsibilities**:
  - Write records to the database (SQLite/PostgreSQL) permanently.
  - Index `title`, `kicker/category`, `author`, `published_date`, and `body` as separate fields.
  - Never drop or clear collections.
  - Keep database persistence separate from the search index (Elasticsearch / PyTerrier).

### 5. Search & Ranking Agent

- **Role**: Answer user queries with ranked results.
- **Responsibilities**:
  - Rank with Okapi BM25 as the baseline, weighting title and kicker above body.
  - Support keyword queries with date range and category filters.
  - Return snippets with title, date, source, and highlighted context.
  - Expose results through a FastAPI/Flask API.