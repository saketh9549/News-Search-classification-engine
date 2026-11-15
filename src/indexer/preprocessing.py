"""Text preprocessing: tokenization, stop-word removal, and Porter stemming."""

from __future__ import annotations

import functools
import re
from datetime import datetime, timezone

from nltk.stem import PorterStemmer

from src.indexer.schemas import IndexField
from src.models.schemas import Article

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_UNDER_AGE_COMPOUND_RE = re.compile(r"\b(?:under|u)[\s-]+(\d{2})\b")
_HYPHEN_ALPHA_NUM_RE = re.compile(r"\b([a-z]{1,6})-(\d{1,4})\b")
_PORTER = PorterStemmer()

# Self-contained standard English stop-words so preprocessing works offline
# without requiring external nltk_data corpus downloads.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "ain", "all", "am",
        "an", "and", "any", "are", "aren", "as", "at", "be", "because", "been",
        "before", "being", "below", "between", "both", "but", "by", "can",
        "couldn", "d", "did", "didn", "do", "does", "doesn", "doing", "don",
        "down", "during", "each", "few", "for", "from", "further", "had",
        "hadn", "has", "hasn", "have", "haven", "having", "he", "her", "here",
        "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
        "into", "is", "isn", "it", "its", "itself", "just", "ll", "m", "ma",
        "me", "mightn", "more", "most", "mustn", "my", "myself", "needn", "no",
        "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or",
        "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s",
        "same", "shan", "she", "should", "shouldn", "so", "some", "such", "t",
        "than", "that", "the", "their", "theirs", "them", "themselves", "then",
        "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up", "v", "ve", "versus", "very", "via", "vs", "was",
        "wasn", "we", "were", "weren", "what", "when", "where", "which", "while",
        "who", "whom", "why", "will", "with", "won", "wouldn", "y", "you",
        "your", "yours", "yourself", "yourselves",
    }
)


def normalize_compound_tokens(text: str) -> str:
    """Keep alphanumeric compound terms (e.g. U19, Under-19, COVID-19, T-20) intact."""
    lowered = text.lower()
    if "-" not in lowered and " " not in lowered and "\t" not in lowered and "\n" not in lowered:
        return lowered
    normalized = _UNDER_AGE_COMPOUND_RE.sub(r"u\1", lowered)
    normalized = _HYPHEN_ALPHA_NUM_RE.sub(r"\1\2", normalized)
    return normalized


@functools.lru_cache(maxsize=65536)
def stem_token(token: str) -> str:
    """Apply Porter stemming to a normalized lowercase token."""
    return _PORTER.stem(token)


def preprocess_text(
    text: str | None,
    *,
    remove_stopwords: bool = True,
    apply_stemming: bool = True,
) -> list[str]:
    """Tokenize, preserve compound tokens, remove stop-words, and Porter-stem text."""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(normalize_compound_tokens(text))
    processed: list[str] = []
    for token in tokens:
        if remove_stopwords and token in ENGLISH_STOPWORDS:
            continue
        processed.append(stem_token(token) if apply_stemming else token)
    return processed


def preprocess_date(published_date: datetime | str | None) -> list[str]:
    """Extract searchable UTC date tokens (YYYY, YYYY-MM, YYYY-MM-DD)."""
    if published_date is None:
        return []
    if isinstance(published_date, datetime):
        utc_dt = published_date.astimezone(timezone.utc)
        date_str = utc_dt.strftime("%Y-%m-%d")
    else:
        raw = published_date.strip()
        if len(raw) < 10:
            return _TOKEN_RE.findall(raw.lower())
        date_str = raw[:10]

    parts = date_str.split("-")
    tokens = [date_str]
    if len(parts) == 3:
        tokens.extend([f"{parts[0]}-{parts[1]}", parts[0]])
    return tokens


def preprocess_query_for_field(query: str, field: IndexField) -> list[str]:
    """Apply field-matched preprocessing to a search query string."""
    if field == IndexField.PUBLISHED_DATE:
        return [tok for tok in re.findall(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", query)]
    return preprocess_text(query)


def tokenize_article_fields(article: Article) -> dict[IndexField, list[str]]:
    """Preprocess an Article into separate token lists for all 5 indexed fields."""
    return {
        IndexField.TITLE: preprocess_text(article.title),
        IndexField.KICKER: preprocess_text(article.kicker),
        IndexField.AUTHOR: preprocess_text(article.author),
        IndexField.PUBLISHED_DATE: preprocess_date(article.published_date),
        IndexField.BODY: preprocess_text(article.body_text),
    }
