"""Indexer package: text preprocessing and persistent multi-field BM25 indexing."""

from src.indexer.bm25_index import BM25Index
from src.indexer.preprocessing import (
    ENGLISH_STOPWORDS,
    preprocess_date,
    preprocess_query_for_field,
    preprocess_text,
    stem_token,
    tokenize_article_fields,
)
from src.indexer.schemas import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_FIELD_WEIGHTS,
    DEFAULT_INDEX_DIR,
    IndexedDocument,
    IndexField,
    IndexSyncResult,
    SearchHit,
)

__all__ = [
    "BM25Index",
    "DEFAULT_BM25_B",
    "DEFAULT_BM25_K1",
    "DEFAULT_FIELD_WEIGHTS",
    "DEFAULT_INDEX_DIR",
    "ENGLISH_STOPWORDS",
    "IndexField",
    "IndexSyncResult",
    "IndexedDocument",
    "SearchHit",
    "preprocess_date",
    "preprocess_query_for_field",
    "preprocess_text",
    "stem_token",
    "tokenize_article_fields",
]
