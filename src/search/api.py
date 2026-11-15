"""FastAPI search interface and lightweight HTML UI for the News Search Engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

from src.indexer.bm25_index import BM25Index
from src.indexer.schemas import DEFAULT_INDEX_DIR
from src.search.schemas import (
    ArticleDetailResponse,
    BackgroundLink,
    CorpusOverview,
    EvaluationReport,
    QueryJudgment,
    SearchResponse,
)
from src.search.service import SearchService, parse_date_boundary

_ROBOTS_TXT_PATH = Path(__file__).resolve().parents[2] / "robots.txt"
_DEFAULT_ROBOTS_TXT = "User-agent: *\nAllow: /\nDisallow: /api/\n"


def _load_robots_txt() -> str:
    if _ROBOTS_TXT_PATH.is_file():
        return _ROBOTS_TXT_PATH.read_text(encoding="utf-8")
    return _DEFAULT_ROBOTS_TXT

_SEARCH_UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>News Search Engine</title>
  <style>
    :root {
      --bg: #f4f6f9;
      --card: #ffffff;
      --ink: #17202a;
      --muted: #566370;
      --accent: #0b57d0;
      --accent-soft: #e8f0fe;
      --border: #dce2ea;
      --mark: #fff0a0;
      --success-bg: #e6f4ea;
      --success-ink: #137333;
      --warn-bg: #fef7e0;
      --warn-border: #f6d57a;
      --code-bg: #0f172a;
      --code-ink: #e2e8f0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.5;
    }
    .container { max-width: 1180px; margin: 0 auto; padding: 24px 18px 64px; }
    header h1 { margin: 0 0 4px; font-size: 1.7rem; }
    header p { margin: 0 0 12px; color: var(--muted); font-size: 0.95rem; }
    .coverage-bar {
      display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px;
      padding: 10px 14px; margin-bottom: 18px;
      background: var(--card); border: 1px solid var(--border); border-radius: 10px;
      font-size: 0.86rem; color: var(--muted);
    }
    .coverage-bar strong { color: var(--ink); }
    .search-panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 18px;
    }
    .query-row { display: flex; gap: 10px; margin-bottom: 10px; }
    .query-row input {
      flex: 1; padding: 10px 12px; font-size: 1rem;
      border: 1px solid var(--border); border-radius: 8px;
    }
    .query-row button {
      padding: 10px 18px; font-size: 1rem; font-weight: 600;
      border: 0; border-radius: 8px; background: var(--accent); color: #fff; cursor: pointer;
    }
    .try-row {
      display: flex; flex-wrap: wrap; align-items: center; gap: 7px;
      margin-bottom: 12px; font-size: 0.84rem; color: var(--muted);
    }
    .pill {
      display: inline-flex; align-items: center; gap: 4px;
      padding: 3px 10px; border-radius: 999px; font-size: 0.8rem;
      border: 1px solid var(--border); background: #f8fafc; color: var(--ink);
      cursor: pointer; text-decoration: none;
    }
    .pill:hover { border-color: var(--accent); background: var(--accent-soft); color: var(--accent); }
    .filters {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px; align-items: end;
    }
    .filters label { display: block; font-size: 0.78rem; color: var(--muted); margin-bottom: 3px; }
    .filters input, .filters select {
      width: 100%; padding: 7px 9px; border: 1px solid var(--border); border-radius: 6px;
      font-size: 0.9rem; background: #fff; color: var(--ink);
    }
    .controls-footer {
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
      gap: 10px; margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border);
      font-size: 0.85rem;
    }
    .debug-toggle { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; font-weight: 500; }
    .reset-btn {
      padding: 4px 10px; border: 1px solid var(--border); border-radius: 6px;
      background: #fff; color: var(--muted); cursor: pointer; font-size: 0.82rem;
    }
    .reset-btn:hover { color: var(--ink); border-color: var(--muted); }
    .layout { display: grid; grid-template-columns: 245px minmax(0, 1fr); gap: 18px; align-items: start; }
    @media (max-width: 820px) { .layout { grid-template-columns: 1fr; } }
    .sidebar {
      background: var(--card); border: 1px solid var(--border); border-radius: 10px;
      padding: 14px;
    }
    .sidebar h3 {
      margin: 0 0 8px; font-size: 0.82rem; text-transform: uppercase;
      letter-spacing: 0.04em; color: var(--muted);
    }
    .facet-group { margin-bottom: 16px; }
    .facet-group:last-child { margin-bottom: 0; }
    .facet-list { display: flex; flex-direction: column; gap: 4px; max-height: 240px; overflow-y: auto; }
    .facet-btn {
      display: flex; justify-content: space-between; align-items: center; width: 100%;
      padding: 5px 8px; border: 1px solid transparent; border-radius: 6px;
      background: transparent; color: var(--ink); font-size: 0.85rem; cursor: pointer; text-align: left;
    }
    .facet-btn:hover { background: #f1f5f9; }
    .facet-btn.active {
      background: var(--accent-soft); border-color: #b7d0fb; color: #0b57d0; font-weight: 600;
    }
    .facet-count { font-size: 0.78rem; color: var(--muted); }
    .debug-panel {
      display: none;
      background: var(--code-bg); color: var(--code-ink);
      border-radius: 10px; padding: 14px 16px; margin-bottom: 14px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.8rem;
    }
    .debug-panel.visible { display: block; }
    .debug-grid { display: grid; grid-template-columns: 165px 1fr; gap: 6px 12px; }
    .debug-key { color: #94a3b8; }
    .entity-bar { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
    .badge {
      display: inline-flex; align-items: center; gap: 4px;
      padding: 3px 9px; border-radius: 999px; font-size: 0.77rem;
      background: #e8eefc; color: #16408b; border: 1px solid #d0dcf8;
      cursor: pointer;
    }
    .badge:hover { background: #d6e3fc; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 14px;
    }
    .card-top {
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start;
      gap: 8px; margin-bottom: 6px;
    }
    .card h2 { margin: 0; font-size: 1.15rem; flex: 1; min-width: 220px; }
    .card h2 a { color: var(--accent); text-decoration: none; }
    .card h2 a:hover { text-decoration: underline; }
    .card h2 mark, .snippet mark {
      background: var(--mark); color: inherit; padding: 0 2px; border-radius: 2px; font-weight: 600;
    }
    .score-group { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
    .score-pill {
      padding: 3px 9px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
      background: #eef2ff; color: #312e81; border: 1px solid #c7d2fe;
    }
    .cat-pill {
      padding: 3px 9px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
      background: var(--success-bg); color: var(--success-ink); border: 1px solid #cce8d4;
    }
    .meta { font-size: 0.84rem; color: var(--muted); margin-bottom: 8px; }
    .field-breakdown { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
    .field-chip {
      padding: 2px 8px; border-radius: 6px; font-size: 0.76rem;
      background: #f8fafc; color: #334155; border: 1px solid var(--border);
    }
    .snippet { margin-bottom: 10px; }
    .card-entities { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
    .entity-chip {
      padding: 2px 8px; border-radius: 999px; font-size: 0.75rem;
      background: #f1f5f9; color: #1e293b; border: 1px solid #cbd5e1; cursor: pointer;
    }
    .entity-chip:hover { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
    .bg-drawer {
      margin-top: 8px; padding: 8px 12px; border-radius: 8px;
      background: #f8fafc; border: 1px solid var(--border); font-size: 0.87rem;
    }
    .bg-drawer summary { cursor: pointer; font-weight: 600; color: #1e293b; }
    .bg-list { margin: 8px 0 2px; padding-left: 18px; color: var(--muted); }
    .bg-list li { margin-bottom: 4px; }
    .bg-list a { color: var(--accent); text-decoration: none; }
    .bg-list a:hover { text-decoration: underline; }
    .empty-state {
      background: var(--warn-bg); border: 1px solid var(--warn-border);
      border-radius: 10px; padding: 16px;
    }
    .empty-state h3 { margin: 0 0 6px; font-size: 1rem; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>News Search Engine</h1>
      <p>Multi-field Okapi BM25 ranking with classification, named entity recognition, and background story linking.</p>
    </header>

    <div id="corpus-stats" class="coverage-bar">Loading corpus statistics...</div>

    <form class="search-panel" id="search-form">
      <div class="query-row">
        <input id="q" name="q" type="search" placeholder="Search news headlines, topics, people, or abbreviations (e.g. ind vs pak u19)..." required />
        <button type="submit">Search</button>
      </div>
      <div class="try-row" id="try-pills">
        <span>Try:</span>
      </div>
      <div class="filters">
        <div>
          <label for="category">Category / Kicker</label>
          <select id="category"><option value="">All Categories</option></select>
        </div>
        <div>
          <label for="source">Source Outlet</label>
          <select id="source"><option value="">All Sources</option></select>
        </div>
        <div>
          <label for="date_from">From (UTC Date)</label>
          <input id="date_from" type="date" />
        </div>
        <div>
          <label for="date_to">To (UTC Date)</label>
          <input id="date_to" type="date" />
        </div>
      </div>
      <div class="controls-footer">
        <label class="debug-toggle">
          <input type="checkbox" id="debug-toggle" />
          Show IR Debug Info
        </label>
        <button type="button" id="reset-filters" class="reset-btn">Reset Filters</button>
      </div>
    </form>

    <div class="layout">
      <aside class="sidebar" id="facet-sidebar">
        <div class="facet-group">
          <h3>Categories</h3>
          <div class="facet-list" id="category-facets"><span class="meta">Loading...</span></div>
        </div>
        <div class="facet-group">
          <h3>Sources</h3>
          <div class="facet-list" id="source-facets"><span class="meta">Loading...</span></div>
        </div>
      </aside>

      <main>
        <div id="ir-debug-panel" class="debug-panel"></div>
        <div id="entities" class="entity-bar"></div>
        <div id="status" class="meta"></div>
        <div id="results"></div>
      </main>
    </div>
  </div>

  <script>
    const form = document.getElementById("search-form");
    const qInput = document.getElementById("q");
    const categorySelect = document.getElementById("category");
    const sourceSelect = document.getElementById("source");
    const dateFromInput = document.getElementById("date_from");
    const dateToInput = document.getElementById("date_to");
    const debugToggle = document.getElementById("debug-toggle");
    const debugPanel = document.getElementById("ir-debug-panel");
    const resultsEl = document.getElementById("results");
    const entitiesEl = document.getElementById("entities");
    const statusEl = document.getElementById("status");
    const corpusStatsEl = document.getElementById("corpus-stats");
    const tryPillsEl = document.getElementById("try-pills");
    const categoryFacetsEl = document.getElementById("category-facets");
    const sourceFacetsEl = document.getElementById("source-facets");
    const resetBtn = document.getElementById("reset-filters");

    let lastDebug = null;
    const FALLBACK_TRY = ["climate summit", "semiconductor", "cricket"];

    function escapeHtml(text) {
      const div = document.createElement("div");
      div.textContent = String(text ?? "");
      return div.innerHTML;
    }

    function capitalizeLabel(label) {
      if (!label) return "General";
      return label.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    }

    function formatDateShort(iso) {
      if (!iso) return "";
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return iso.slice(0, 10);
      return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
    }

    function populateSelect(selectEl, defaultLabel, buckets, currentVal) {
      const options = [`<option value="">${escapeHtml(defaultLabel)}</option>`];
      for (const b of buckets) {
        const selected = b.name === currentVal ? " selected" : "";
        options.push(`<option value="${escapeHtml(b.name)}"${selected}>${escapeHtml(b.name)} (${b.count})</option>`);
      }
      selectEl.innerHTML = options.join("");
    }

    function renderFacetList(container, buckets, activeVal, selectEl) {
      if (!buckets.length) {
        container.innerHTML = `<span class="meta">No facets indexed yet</span>`;
        return;
      }
      const allBtn = `<button type="button" class="facet-btn${!activeVal ? " active" : ""}" data-val=""><span>All</span></button>`;
      const items = buckets.map(b => {
        const isActive = b.name === activeVal ? " active" : "";
        return `<button type="button" class="facet-btn${isActive}" data-val="${escapeHtml(b.name)}"><span>${escapeHtml(b.name)}</span><span class="facet-count">${b.count}</span></button>`;
      });
      container.innerHTML = allBtn + items.join("");
      for (const btn of container.querySelectorAll(".facet-btn")) {
        btn.addEventListener("click", () => {
          const val = btn.dataset.val || "";
          selectEl.value = (val && selectEl.value === val) ? "" : val;
          if (qInput.value.trim() || hasActiveFilters()) {
            runSearch();
          } else {
            renderOverviewSync();
            clearToIdle();
            qInput.focus();
          }
        });
      }
    }

    function hasActiveFilters() {
      return Boolean(
        categorySelect.value.trim()
        || sourceSelect.value.trim()
        || dateFromInput.value.trim()
        || dateToInput.value.trim()
      );
    }

    function clearToIdle() {
      statusEl.textContent = "Enter keywords or select a category, source, or date filter.";
      resultsEl.innerHTML = "";
      entitiesEl.innerHTML = "";
      renderDebugPanel(null);
    }

    let cachedOverview = null;
    function renderOverviewSync() {
      if (!cachedOverview) return;
      renderOverview(cachedOverview);
    }

    function renderOverview(overview) {
      cachedOverview = overview;
      const windowText = (overview.date_min && overview.date_max)
        ? `${formatDateShort(overview.date_min)} – ${formatDateShort(overview.date_max)}`
        : "No dates indexed";
      corpusStatsEl.innerHTML =
        `<span><strong>Indexed:</strong> ${overview.doc_count} article(s)</span>` +
        `<span><strong>Sources:</strong> ${overview.sources.length}</span>` +
        `<span><strong>Categories:</strong> ${overview.categories.length}</span>` +
        `<span><strong>Crawl Window (UTC):</strong> ${escapeHtml(windowText)}</span>`;

      if (overview.date_min) dateFromInput.min = overview.date_min.slice(0, 10);
      if (overview.date_max) dateToInput.max = overview.date_max.slice(0, 10);

      const curCat = categorySelect.value;
      const curSrc = sourceSelect.value;
      populateSelect(categorySelect, "All Categories", overview.categories, curCat);
      populateSelect(sourceSelect, "All Sources", overview.sources, curSrc);
      renderFacetList(categoryFacetsEl, overview.categories, categorySelect.value, categorySelect);
      renderFacetList(sourceFacetsEl, overview.sources, sourceSelect.value, sourceSelect);

      const samples = (overview.sample_queries && overview.sample_queries.length)
        ? overview.sample_queries
        : FALLBACK_TRY;
      tryPillsEl.innerHTML = `<span>Try:</span>` + samples.map(q =>
        `<button type="button" class="pill try-pill" data-query="${escapeHtml(q)}">${escapeHtml(q)}</button>`
      ).join("");
      for (const pill of tryPillsEl.querySelectorAll(".try-pill")) {
        pill.addEventListener("click", () => {
          qInput.value = pill.dataset.query || "";
          runSearch();
        });
      }
    }

    function renderDebugPanel(debug) {
      lastDebug = debug;
      if (!debugToggle.checked || !debug) {
        debugPanel.classList.remove("visible");
        debugPanel.innerHTML = "";
        return;
      }
      const expansions = Object.entries(debug.expanded_terms || {})
        .map(([k, v]) => `${k} -> [${v.join(", ")}]`)
        .join("; ") || "none";
      const weights = Object.entries(debug.field_weights || {})
        .map(([f, w]) => `${f}: ${w}x`)
        .join(" · ");
      debugPanel.classList.add("visible");
      debugPanel.innerHTML = `
        <div class="debug-grid">
          <div class="debug-key">Execution Time:</div><div>${debug.execution_time_ms.toFixed(2)} ms</div>
          <div class="debug-key">Raw Tokens:</div><div>[${escapeHtml((debug.raw_tokens || []).join(", "))}]</div>
          <div class="debug-key">Stopwords Removed:</div><div>[${escapeHtml((debug.removed_stopwords || []).join(", ")) || "none"}]</div>
          <div class="debug-key">Porter Stems:</div><div>[${escapeHtml((debug.stemmed_tokens || []).join(", "))}]</div>
          <div class="debug-key">Expanded Terms:</div><div>${escapeHtml(expansions)}</div>
          <div class="debug-key">Final BM25 Terms:</div><div>[${escapeHtml((debug.final_terms || []).join(", "))}]</div>
          <div class="debug-key">Field Weights:</div><div>${escapeHtml(weights)}</div>
        </div>`;
    }

    function triggerQuery(queryText) {
      qInput.value = queryText;
      runSearch();
    }

    function renderBackgroundList(links) {
      if (!links || !links.length) {
        return `<p class="meta" style="margin:6px 0 0;">No related background stories found in current index.</p>`;
      }
      const items = links.map(l => {
        const shared = l.shared_entities && l.shared_entities.length
          ? ` — Shared: ${escapeHtml(l.shared_entities.join(", "))}`
          : "";
        return `<li><a href="${escapeHtml(l.canonical_url)}" target="_blank" rel="noopener">${escapeHtml(l.title)}</a>` +
          ` (${escapeHtml(l.source || "Unknown")}${l.published_date ? ", " + escapeHtml(formatDateShort(l.published_date)) : ""} · sim ${Number(l.similarity_score).toFixed(2)}${shared})</li>`;
      }).join("");
      return `<ul class="bg-list">${items}</ul>`;
    }

    async function runSearch() {
      const qVal = qInput.value.trim();
      if (!qVal && !hasActiveFilters()) {
        clearToIdle();
        return;
      }

      const params = new URLSearchParams();
      if (qVal) params.set("q", qVal);
      for (const [key, el] of [
        ["category", categorySelect],
        ["source", sourceSelect],
        ["date_from", dateFromInput],
        ["date_to", dateToInput],
      ]) {
        const val = el.value.trim();
        if (val) params.set(key, val);
      }

      statusEl.textContent = "Searching...";
      let res;
      let data = {};
      try {
        res = await fetch(`/api/search?${params.toString()}`);
        try {
          data = await res.json();
        } catch (_parseErr) {
          data = {};
        }
      } catch (_netErr) {
        statusEl.textContent = "Network error while contacting search API.";
        resultsEl.innerHTML = "";
        return;
      }

      if (!res.ok) {
        statusEl.textContent = data.detail || `Search error (HTTP ${res.status})`;
        resultsEl.innerHTML = "";
        return;
      }

      if (data.facets) renderOverview(data.facets);
      renderDebugPanel(data.query_debug);

      const execTime = data.query_debug ? ` in ${data.query_debug.execution_time_ms.toFixed(1)} ms` : "";
      statusEl.textContent = data.query.trim()
        ? `${data.total_hits} result(s) for "${data.query}"${execTime}`
        : `Showing ${data.total_hits} filtered article(s)${execTime}`;

      entitiesEl.innerHTML = data.top_entities.map(ent =>
        `<button type="button" class="badge entity-trigger" data-entity="${escapeHtml(ent.text)}">${escapeHtml(ent.label)}: ${escapeHtml(ent.text)} (${ent.count})</button>`
      ).join("");

      if (!data.results.length) {
        const retryTarget = data.query.trim() ? ` &amp; retry "${escapeHtml(data.query)}"` : "";
        const filterHint = (data.unfiltered_total_hits && data.unfiltered_total_hits > 0)
          ? `<p>Active filters hid <strong>${data.unfiltered_total_hits}</strong> matching article(s). <button type="button" class="pill" id="clear-retry-btn">Clear filters${retryTarget}</button></p>`
          : `<p>No documents matched all terms. Try broadening abbreviations, clearing date filters, or picking a sample query below:</p>`;
        const sugPills = (data.suggestions || []).map(s =>
          `<button type="button" class="pill sug-pill" data-query="${escapeHtml(s)}">${escapeHtml(s)}</button>`
        ).join(" ");
        resultsEl.innerHTML = `<div class="empty-state"><h3>No matching results found</h3>${filterHint}<div class="try-row">${sugPills}</div></div>`;
        const clearRetry = document.getElementById("clear-retry-btn");
        if (clearRetry) {
          clearRetry.addEventListener("click", () => {
            categorySelect.value = "";
            sourceSelect.value = "";
            dateFromInput.value = "";
            dateToInput.value = "";
            runSearch();
          });
        }
        for (const p of resultsEl.querySelectorAll(".sug-pill")) {
          p.addEventListener("click", () => triggerQuery(p.dataset.query || ""));
        }
        return;
      }

      resultsEl.innerHTML = data.results.map(item => {
        const titleHtml = item.highlighted_title || escapeHtml(item.title);
        const catConf = `${escapeHtml(capitalizeLabel(item.predicted_category))} · ${Math.round(Number(item.category_confidence) * 100)}%`;
        const fieldChips = (item.field_matches || []).map(fm =>
          `<span class="field-chip">${escapeHtml(fm.label)} (${Number(fm.weight).toFixed(1)}x: +${Number(fm.score).toFixed(2)})</span>`
        ).join("");
        const entChips = (item.entities || []).map(ent =>
          `<button type="button" class="entity-chip entity-trigger" data-entity="${escapeHtml(ent.text)}">${escapeHtml(ent.label)}: ${escapeHtml(ent.text)}</button>`
        ).join("");
        const bgCount = (item.background_links || []).length;
        return `
          <article class="card">
            <div class="card-top">
              <h2><a href="${escapeHtml(item.canonical_url)}" target="_blank" rel="noopener">${titleHtml}</a></h2>
              <div class="score-group">
                <span class="cat-pill">${catConf}</span>
                <span class="score-pill">${data.query.trim() ? `Score: ${Number(item.score).toFixed(2)}` : "Filter match"}</span>
              </div>
            </div>
            <div class="meta">
              ${escapeHtml(item.source || "Unknown Source")}
              ${item.kicker ? " · Kicker: " + escapeHtml(item.kicker) : ""}
              ${item.author ? " · By " + escapeHtml(item.author) : ""}
              ${item.published_date ? " · " + escapeHtml(formatDateShort(item.published_date)) : ""}
            </div>
            <div class="field-breakdown">${fieldChips}</div>
            <div class="snippet">${item.snippet}</div>
            ${entChips ? `<div class="card-entities">${entChips}</div>` : ""}
            <details class="bg-drawer">
              <summary>Related / Background Stories (${bgCount})</summary>
              ${renderBackgroundList(item.background_links)}
            </details>
          </article>`;
      }).join("");

      for (const trigger of document.querySelectorAll(".entity-trigger")) {
        trigger.addEventListener("click", () => triggerQuery(trigger.dataset.entity || ""));
      }
    }

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      runSearch();
    });

    for (const el of [categorySelect, sourceSelect, dateFromInput, dateToInput]) {
      el.addEventListener("change", () => {
        if (qInput.value.trim() || hasActiveFilters()) runSearch();
        else {
          renderOverviewSync();
          clearToIdle();
        }
      });
    }

    debugToggle.addEventListener("change", () => renderDebugPanel(lastDebug));

    resetBtn.addEventListener("click", () => {
      categorySelect.value = "";
      sourceSelect.value = "";
      dateFromInput.value = "";
      dateToInput.value = "";
      renderOverviewSync();
      if (qInput.value.trim()) runSearch();
      else clearToIdle();
    });

    (async () => {
      try {
        const res = await fetch("/api/corpus");
        if (res.ok) {
          renderOverview(await res.json());
        }
      } catch (_err) {
        corpusStatsEl.textContent = "Corpus statistics unavailable.";
      }
    })();
  </script>
</body>
</html>
"""


def create_app(
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    *,
    index: BM25Index | None = None,
) -> FastAPI:
    """Create a configured FastAPI application for news search and evaluation."""
    resolved_index = index if index is not None else BM25Index(index_dir)
    service = SearchService(resolved_index)

    application = FastAPI(
        title="News Crawler & Search Engine API",
        version="0.1.0",
        description="Multi-field Okapi BM25 search with classification, NER, and background story linking.",
    )
    application.state.search_service = service

    @application.get("/", response_class=HTMLResponse)
    def search_ui() -> HTMLResponse:
        return HTMLResponse(content=_SEARCH_UI_HTML)

    @application.get("/robots.txt", response_class=PlainTextResponse)
    def robots_txt() -> PlainTextResponse:
        return PlainTextResponse(content=_load_robots_txt())

    @application.get("/health")
    def health() -> dict[str, object]:
        service.index.refresh_if_changed()
        return {
            "status": "ok",
            "doc_count": service.index.doc_count,
            "index_dir": str(service.index.index_dir),
        }

    @application.get("/api/corpus", response_model=CorpusOverview)
    def get_corpus() -> CorpusOverview:
        return service.get_corpus_overview()

    @application.get("/api/search", response_model=SearchResponse)
    def search_articles(
        q: Annotated[
            str,
            Query(description="Keyword search query (optional when filters are active)"),
        ] = "",
        k: Annotated[int, Query(ge=1, le=100, description="Maximum number of results")] = 10,
        category: Annotated[str | None, Query(description="Filter by kicker/category")] = None,
        source: Annotated[str | None, Query(description="Filter by publisher source name")] = None,
        date_from: Annotated[
            str | None, Query(description="Inclusive start UTC date/ISO timestamp")
        ] = None,
        date_to: Annotated[
            str | None, Query(description="Inclusive end UTC date/ISO timestamp")
        ] = None,
    ) -> SearchResponse:
        has_filters = bool(
            (category and category.strip())
            or (source and source.strip())
            or (date_from and date_from.strip())
            or (date_to and date_to.strip())
        )
        if not q.strip() and not has_filters:
            raise HTTPException(status_code=422, detail="Query string 'q' must not be blank.")
        try:
            start_dt = parse_date_boundary(date_from, end_of_day=False)
            end_dt = parse_date_boundary(date_to, end_of_day=True)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid date filter: {exc}") from exc

        if start_dt and end_dt and start_dt > end_dt:
            raise HTTPException(
                status_code=422,
                detail="'date_from' must be earlier than or equal to 'date_to'.",
            )

        return service.search(
            q,
            top_k=k,
            category=category,
            source=source,
            date_from=start_dt,
            date_to=end_dt,
        )

    @application.get("/api/articles/{doc_id}", response_model=ArticleDetailResponse)
    def get_article(
        doc_id: str,
        background_limit: Annotated[int, Query(ge=1, le=10)] = 3,
    ) -> ArticleDetailResponse:
        detail = service.get_article_detail(doc_id, background_limit=background_limit)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Article {doc_id!r} not found.")
        return detail

    @application.get("/api/articles/{doc_id}/background", response_model=list[BackgroundLink])
    def get_article_background(
        doc_id: str,
        limit: Annotated[int, Query(ge=1, le=10)] = 3,
    ) -> list[BackgroundLink]:
        links = service.get_background_links(doc_id, limit=limit)
        if links is None:
            raise HTTPException(status_code=404, detail=f"Article {doc_id!r} not found.")
        return links

    @application.post("/api/evaluate", response_model=EvaluationReport)
    def evaluate_queries(
        judgments: list[QueryJudgment],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
    ) -> EvaluationReport:
        return service.evaluate(judgments, k=k)

    return application


app = create_app(Path(os.environ.get("NEWS_INDEX_DIR", str(DEFAULT_INDEX_DIR))))
