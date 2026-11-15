"""Query preprocessing inspection, concept grouping, and phrase-aware query expansion."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.indexer.preprocessing import (
    ENGLISH_STOPWORDS,
    normalize_compound_tokens,
    preprocess_text,
    stem_token,
)

_COMPOUND_SPAN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
_ISO_DATE_RE = re.compile(r"\b\d{4}(?:-\d{2}(?:-\d{2})?)?\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_UNDER_AGE_RE = re.compile(r"^u(\d{2})$")

EXPANSION_WEIGHT: float = 0.25

NEWS_SYNONYM_MAP: dict[str, tuple[str, ...]] = {
    "ind": ("india", "indian"),
    "pak": ("pakistan", "pakistani"),
    "aus": ("australia", "australian"),
    "eng": ("england", "english"),
    "nz": ("new zealand",),
    "sa": ("south africa",),
    "sl": ("sri lanka",),
    "ban": ("bangladesh",),
    "afg": ("afghanistan",),
    "wi": ("west indies",),
    "u19": ("under-19", "u-19"),
    "odi": ("one-day", "cricket"),
    "t20": ("twenty20", "cricket"),
    "wc": ("world cup",),
    "ipl": ("indian premier league",),
    "bcci": ("board of control for cricket in india",),
    "icc": ("international cricket council",),
    "us": ("united states", "america", "american"),
    "usa": ("united states", "america"),
    "uk": ("britain", "british", "united kingdom"),
    "un": ("united nations",),
    "eu": ("european union", "europe"),
    "uae": ("emirates", "dubai"),
    "pm": ("prime minister",),
    "fm": ("finance minister", "foreign minister"),
    "sc": ("supreme court",),
    "hc": ("high court",),
    "ec": ("election commission",),
    "rbi": ("reserve bank", "central bank"),
    "fed": ("federal reserve", "interest rate"),
    "imf": ("international monetary fund",),
    "isro": ("space", "rocket", "satellite"),
    "nasa": ("space", "orbit", "mission"),
    "ai": ("artificial intelligence", "machine learning"),
    "ml": ("machine learning", "artificial intelligence"),
    "llm": ("large language model", "artificial intelligence"),
    "ev": ("electric vehicle", "battery"),
    "evs": ("electric vehicles", "battery"),
    "gdp": ("economic growth", "economy"),
    "ipo": ("stock market", "shares", "listing"),
    "crypto": ("bitcoin", "cryptocurrency"),
    "btc": ("bitcoin", "cryptocurrency"),
    "tech": ("technology",),
    "biz": ("business",),
    "govt": ("government",),
    "dept": ("department",),
    "intl": ("international",),
    "covid": ("coronavirus", "pandemic"),
    "covid19": ("coronavirus", "pandemic"),
    "cop": ("climate summit",),
    "semicon": ("semiconductor", "chip"),
    "chips": ("semiconductor",),
    "chip": ("semiconductor",),
}


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    """Detailed trace of tokenization, stop-word removal, stemming, expansion, and concepts."""

    raw_query: str
    raw_tokens: list[str]
    removed_stopwords: list[str]
    stemmed_tokens: list[str]
    expanded_terms: dict[str, list[str]]
    extra_index_terms: dict[str, float]
    expanded_phrases: list[tuple[str, tuple[str, ...], float]]
    extra_highlight_terms: list[str]
    final_terms: list[str]
    concept_groups: list[list[str]]


def _lookup_expansions(token: str) -> list[str]:
    phrases = list(NEWS_SYNONYM_MAP.get(token, ()))
    if not phrases:
        age_match = _UNDER_AGE_RE.match(token)
        if age_match:
            age = age_match.group(1)
            phrases.extend([f"under-{age}", f"u-{age}"])
    return phrases


def inspect_and_expand_query(
    query: str,
    *,
    expansion_weight: float = EXPANSION_WEIGHT,
) -> QueryAnalysis:
    """Inspect preprocessing stages, group concepts, and separate single-term vs phrase expansions."""
    raw_tokens = [tok.lower() for tok in _COMPOUND_SPAN_RE.findall(query)]
    date_tokens = list(dict.fromkeys(_ISO_DATE_RE.findall(query)))
    non_date_query = _ISO_DATE_RE.sub(" ", query)
    normalized_tokens = _TOKEN_RE.findall(normalize_compound_tokens(non_date_query))

    removed_stopwords: list[str] = []
    content_tokens: list[str] = []
    stemmed_tokens: list[str] = []

    for tok in normalized_tokens:
        if tok in ENGLISH_STOPWORDS:
            removed_stopwords.append(tok)
            continue
        content_tokens.append(tok)
        stemmed = stem_token(tok)
        if stemmed not in stemmed_tokens:
            stemmed_tokens.append(stemmed)

    seen_stems = set(stemmed_tokens)
    expanded_terms: dict[str, list[str]] = {}
    extra_index_terms: dict[str, float] = {}
    expanded_phrases: list[tuple[str, tuple[str, ...], float]] = []
    seen_phrase_keys: set[tuple[str, ...]] = set()
    extra_highlight_terms: list[str] = []
    concept_groups_by_stem: dict[str, list[str]] = {stem: [stem] for stem in stemmed_tokens}

    for tok in content_tokens:
        base_stem = stem_token(tok)
        group = concept_groups_by_stem[base_stem]
        phrases = _lookup_expansions(tok)
        if not phrases:
            continue
        expanded_terms[tok] = phrases
        for phrase in phrases:
            lowered_phrase = phrase.lower()
            phrase_stems = tuple(preprocess_text(phrase))
            if not phrase_stems:
                continue

            if len(phrase_stems) == 1:
                exp_stem = phrase_stems[0]
                if "-" in lowered_phrase or _UNDER_AGE_RE.match(tok):
                    if lowered_phrase not in extra_highlight_terms:
                        extra_highlight_terms.append(lowered_phrase)
                elif lowered_phrase not in extra_highlight_terms:
                    extra_highlight_terms.append(lowered_phrase)

                if exp_stem not in group:
                    group.append(exp_stem)
                if exp_stem not in seen_stems:
                    extra_index_terms[exp_stem] = expansion_weight
            else:
                if lowered_phrase not in extra_highlight_terms:
                    extra_highlight_terms.append(lowered_phrase)
                phrase_marker = f"__phrase__:{'_'.join(phrase_stems)}"
                if phrase_marker not in group:
                    group.append(phrase_marker)
                if phrase_stems not in seen_phrase_keys:
                    seen_phrase_keys.add(phrase_stems)
                    expanded_phrases.append((lowered_phrase, phrase_stems, expansion_weight))

    final_terms = list(stemmed_tokens)
    for exp_stem in extra_index_terms:
        if exp_stem not in final_terms:
            final_terms.append(exp_stem)
    for _, phrase_stems, _ in expanded_phrases:
        phrase_marker = f"__phrase__:{'_'.join(phrase_stems)}"
        if phrase_marker not in final_terms:
            final_terms.append(phrase_marker)

    concept_groups: list[list[str]] = [[dt] for dt in date_tokens]
    concept_groups.extend(concept_groups_by_stem.values())

    return QueryAnalysis(
        raw_query=query,
        raw_tokens=raw_tokens,
        removed_stopwords=removed_stopwords,
        stemmed_tokens=stemmed_tokens,
        expanded_terms=expanded_terms,
        extra_index_terms=extra_index_terms,
        expanded_phrases=expanded_phrases,
        extra_highlight_terms=extra_highlight_terms,
        final_terms=final_terms,
        concept_groups=concept_groups,
    )
