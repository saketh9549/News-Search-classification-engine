"""Snippet extraction and query-term highlighting for search results."""

from __future__ import annotations

import functools
import html
import re
from collections.abc import Iterable, Sequence

from src.indexer.preprocessing import (
    ENGLISH_STOPWORDS,
    normalize_compound_tokens,
    preprocess_text,
    stem_token,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_SPAN_RE = re.compile(
    r"\b(?:under-\d{2}|[A-Za-z]{1,6}-\d{1,4})\b|[A-Za-z0-9]+",
    re.IGNORECASE,
)
_DATELINE_LINE_RE = re.compile(
    r"^(?:updated|published)\s*[-–—:]\s*.*$",
    re.IGNORECASE,
)
_INLINE_PHOTO_CREDIT_RE = re.compile(
    r"\s*\|\s*(?:photo\s+credit|file\s+photo|image\s+credit|credit)\b[^\n]*$",
    re.IGNORECASE,
)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;!?)\]]|\.(?!\d))")
_SPACE_BEFORE_COLON_RE = re.compile(r"(?<=[A-Za-z])\s+:")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[])\s+")
_DUP_COMMA_RE = re.compile(r",\s*,+")
_COMMA_PERIOD_RE = re.compile(r",\s*\.(?!\d)")


def _clean_single_block(block: str) -> str:
    cleaned = _INLINE_PHOTO_CREDIT_RE.sub("", block)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _SPACE_BEFORE_COLON_RE.sub(":", cleaned)
    cleaned = _SPACE_AFTER_OPEN_RE.sub(r"\1", cleaned)
    cleaned = _DUP_COMMA_RE.sub(",", cleaned)
    cleaned = _COMMA_PERIOD_RE.sub(".", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def normalize_snippet_text(text: str) -> str:
    """Clean stray whitespace, dateline banners, photo credits, and broken punctuation spacing."""
    if not text:
        return ""
    kept_blocks: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or _DATELINE_LINE_RE.match(stripped):
            continue
        cleaned = _clean_single_block(stripped)
        if cleaned:
            kept_blocks.append(cleaned)
    return " ".join(kept_blocks)


@functools.lru_cache(maxsize=2048)
def _cached_query_match_sets(
    query: str,
    extra_tuple: tuple[str, ...],
) -> tuple[frozenset[str], frozenset[str], tuple[re.Pattern[str], ...]]:
    raw_terms = {
        tok.lower()
        for tok in _WORD_SPAN_RE.findall(query)
        if len(tok) > 1 and tok.lower() not in ENGLISH_STOPWORDS
    }
    raw_terms.update(tok for tok in preprocess_text(query, apply_stemming=False))
    stems = set(preprocess_text(query))
    phrase_patterns: list[re.Pattern[str]] = []

    for term in extra_tuple:
        cleaned = term.strip().lower()
        if not cleaned or cleaned in ENGLISH_STOPWORDS:
            continue
        if " " in cleaned:
            words = [re.escape(w) for w in cleaned.split() if w]
            if words:
                phrase_patterns.append(re.compile(rf"\b{'\\s+'.join(words)}\b", re.IGNORECASE))
        elif "-" in cleaned:
            raw_terms.add(cleaned)
            stems.update(preprocess_text(cleaned))
        else:
            for tok in _WORD_SPAN_RE.findall(cleaned):
                if len(tok) > 1 and tok not in ENGLISH_STOPWORDS:
                    raw_terms.add(tok)
                    stems.add(stem_token(tok))

    return frozenset(raw_terms), frozenset(stems), tuple(phrase_patterns)


def _query_match_sets(
    query: str,
    extra_terms: Iterable[str] = (),
) -> tuple[frozenset[str], frozenset[str], tuple[re.Pattern[str], ...]]:
    return _cached_query_match_sets(query, tuple(extra_terms))


def _token_matches(
    token: str,
    raw_terms: frozenset[str] | set[str],
    stems: frozenset[str] | set[str],
) -> bool:
    lowered = token.lower()
    if lowered in raw_terms:
        return True
    if lowered in ENGLISH_STOPWORDS:
        return False
    normalized = normalize_compound_tokens(lowered) if "-" in lowered else lowered
    if normalized in raw_terms or normalized in stems:
        return True
    if stems and stem_token(normalized) in stems:
        return True
    return False


def _find_highlight_spans(
    text: str,
    raw_terms: frozenset[str] | set[str],
    stems: frozenset[str] | set[str],
    phrase_patterns: Sequence[re.Pattern[str]],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in phrase_patterns:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))

    for match in _WORD_SPAN_RE.finditer(text):
        if _token_matches(match.group(0), raw_terms, stems):
            spans.append((match.start(), match.end()))

    if not spans:
        return []

    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            if end > prev_end:
                merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


@functools.lru_cache(maxsize=2048)
def prepare_body_sentences(
    body_text: str,
) -> tuple[str, tuple[str, ...], tuple[tuple[tuple[str, str], ...], ...]]:
    """Normalize paragraphs, split sentences on boundaries, and precompute stem pairs."""
    if not body_text or not body_text.strip():
        return "", (), ()
    sentence_list: list[str] = []
    for raw_line in body_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or _DATELINE_LINE_RE.match(stripped):
            continue
        cleaned_block = _clean_single_block(stripped)
        if not cleaned_block:
            continue
        for part in _SENTENCE_SPLIT_RE.split(cleaned_block):
            s = part.strip()
            if s:
                sentence_list.append(s)
    if not sentence_list:
        return "", (), ()
    sentences = tuple(sentence_list)
    collapsed = " ".join(sentences)
    sent_tokens: list[tuple[tuple[str, str], ...]] = []
    for s in sentences:
        pairs: list[tuple[str, str]] = []
        for tok in _WORD_SPAN_RE.findall(s):
            lowered = tok.lower()
            if lowered in ENGLISH_STOPWORDS:
                continue
            norm = normalize_compound_tokens(lowered) if "-" in lowered else lowered
            pairs.append((norm, stem_token(norm)))
        sent_tokens.append(tuple(pairs))
    return collapsed, sentences, tuple(sent_tokens)


def _select_best_window(
    body_text: str,
    raw_terms: frozenset[str] | set[str],
    stems: frozenset[str] | set[str],
    phrase_patterns: Sequence[re.Pattern[str]],
    *,
    max_chars: int,
) -> str:
    collapsed, sentences, sent_tokens = prepare_body_sentences(body_text)
    if not collapsed:
        return ""
    if len(collapsed) <= max_chars:
        return collapsed

    sentence_scores = [
        sum(1 for norm, st in pairs if norm in raw_terms or st in stems)
        + (sum(2 * len(p.findall(s)) for p in phrase_patterns) if phrase_patterns else 0)
        for s, pairs in zip(sentences, sent_tokens)
    ]

    best_segment = ""
    best_score = -1

    for idx, sentence in enumerate(sentences):
        window = sentence
        matches = sentence_scores[idx]
        if len(window) < max_chars * 0.6 and idx + 1 < len(sentences):
            sep = " " if window[-1] in ".!?:;—–-" else " — "
            window = f"{window}{sep}{sentences[idx + 1]}"
            matches += sentence_scores[idx + 1]
        if matches > best_score:
            best_score = matches
            best_segment = window

    if best_score <= 0 or not best_segment:
        best_segment = collapsed

    if len(best_segment) <= max_chars:
        return best_segment

    spans = _find_highlight_spans(best_segment, raw_terms, stems, phrase_patterns)
    first_match_pos = spans[0][0] if spans else 0

    start = max(0, first_match_pos - max_chars // 3)
    if start > 0:
        next_space = best_segment.find(" ", start)
        if next_space != -1 and next_space < start + 25:
            start = next_space + 1

    end = min(len(best_segment), start + max_chars)
    if end < len(best_segment):
        prev_space = best_segment.rfind(" ", start, end)
        if prev_space > start + max_chars // 2:
            end = prev_space

    snippet = best_segment[start:end].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(best_segment):
        snippet = f"{snippet}..."
    return snippet


def _render_highlighted(
    text: str,
    raw_terms: frozenset[str] | set[str],
    stems: frozenset[str] | set[str],
    phrase_patterns: Sequence[re.Pattern[str]],
    *,
    highlight_open: str = "<mark>",
    highlight_close: str = "</mark>",
) -> str:
    if not text:
        return ""
    if not raw_terms and not stems and not phrase_patterns:
        return html.escape(text)

    spans = _find_highlight_spans(text, raw_terms, stems, phrase_patterns)
    if not spans:
        return html.escape(text)

    parts: list[str] = []
    last_end = 0
    for start, end in spans:
        parts.append(html.escape(text[last_end:start]))
        parts.append(f"{highlight_open}{html.escape(text[start:end])}{highlight_close}")
        last_end = end

    parts.append(html.escape(text[last_end:]))
    return "".join(parts)


def highlight_text(
    text: str,
    query: str,
    *,
    extra_terms: Iterable[str] = (),
    highlight_open: str = "<mark>",
    highlight_close: str = "</mark>",
) -> str:
    """Safely HTML-escape text and wrap matching query/expanded terms in highlight tags."""
    if not text:
        return ""
    cleaned = normalize_snippet_text(text)
    raw_terms, stems, phrase_patterns = _query_match_sets(query, extra_terms=extra_terms)
    return _render_highlighted(
        cleaned,
        raw_terms,
        stems,
        phrase_patterns,
        highlight_open=highlight_open,
        highlight_close=highlight_close,
    )


def build_highlighted_snippet(
    body_text: str,
    query: str,
    *,
    extra_terms: Iterable[str] = (),
    max_chars: int = 240,
    highlight_open: str = "<mark>",
    highlight_close: str = "</mark>",
) -> str:
    """Select the highest-matching passage and highlight query terms safely."""
    raw_terms, stems, phrase_patterns = _query_match_sets(query, extra_terms=extra_terms)
    window = _select_best_window(
        body_text,
        raw_terms,
        stems,
        phrase_patterns,
        max_chars=max_chars,
    )
    return _render_highlighted(
        window,
        raw_terms,
        stems,
        phrase_patterns,
        highlight_open=highlight_open,
        highlight_close=highlight_close,
    )
