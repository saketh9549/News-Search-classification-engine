"""Persistent, incremental multi-field Okapi BM25 inverted index decoupled from SQLite."""

from __future__ import annotations

import functools
import json
import math
import os
import re
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from src.indexer.preprocessing import (
    ENGLISH_STOPWORDS,
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
    IndexField,
    IndexedDocument,
    IndexSyncResult,
    SearchHit,
)
from src.models.schemas import (
    Article,
    map_to_canonical_category,
    normalize_category,
    to_iso8601_utc,
    utc_now,
)

_MANIFEST_FILE = "manifest.json"
_DOCUMENTS_FILE = "documents.json"
_POSTINGS_FILE = "postings.json"
_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"')
_RAW_WORD_RE = re.compile(r"[a-z]{3,}")


def _edit_distance_le2(a: str, b: str) -> int:
    """Compute Levenshtein edit distance, short-circuiting early when > 2."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 2:
        return 3
    if a == b:
        return 0
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * lb
        row_min = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            curr[j] = val
            if val < row_min:
                row_min = val
        if row_min > 2:
            return 3
        prev = curr
    return prev[lb]


@functools.lru_cache(maxsize=4096)
def _cached_field_stem_str(field_text: str) -> str:
    if not field_text:
        return ""
    return " " + " ".join(preprocess_text(field_text)) + " "


def _atomic_write_json(target: Path, payload: object) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.stem}_",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


class BM25Index:
    """Standalone multi-field Okapi BM25 index persisted separately from SQLite."""

    def __init__(
        self,
        index_dir: Path | str = DEFAULT_INDEX_DIR,
        *,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
        field_weights: Mapping[IndexField, float] | None = None,
    ) -> None:
        self._index_dir = Path(index_dir)
        self._k1 = float(k1)
        self._b = float(b)
        self._field_weights: dict[IndexField, float] = dict(
            field_weights if field_weights is not None else DEFAULT_FIELD_WEIGHTS
        )
        self._documents: dict[str, IndexedDocument] = {}
        self._postings: dict[IndexField, dict[str, dict[str, int]]] = {
            field: {} for field in IndexField
        }
        self._field_total_lengths: dict[IndexField, int] = {
            field: 0 for field in IndexField
        }
        self._manifest_sig: tuple[int, int] | None = None
        self._raw_vocab_df: Counter[str] | None = None
        self._raw_vocab_buckets: dict[tuple[str, int], list[tuple[str, int]]] | None = None
        self._load_if_exists()

    @property
    def index_dir(self) -> Path:
        return self._index_dir

    @property
    def doc_count(self) -> int:
        return len(self._documents)

    @property
    def indexed_doc_ids(self) -> frozenset[str]:
        return frozenset(self._documents.keys())

    @property
    def field_weights(self) -> dict[IndexField, float]:
        return dict(self._field_weights)

    def _load_if_exists(self) -> None:
        manifest_path = self._index_dir / _MANIFEST_FILE
        docs_path = self._index_dir / _DOCUMENTS_FILE
        postings_path = self._index_dir / _POSTINGS_FILE

        if not (manifest_path.is_file() and docs_path.is_file() and postings_path.is_file()):
            return

        self._documents.clear()
        self._postings = {field: {} for field in IndexField}
        self._field_total_lengths = {field: 0 for field in IndexField}

        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        for field_name, total_len in manifest.get("field_total_lengths", {}).items():
            self._field_total_lengths[IndexField(field_name)] = int(total_len)

        with docs_path.open("r", encoding="utf-8") as fh:
            raw_docs = json.load(fh)
        for doc_id, doc_data in raw_docs.items():
            doc = IndexedDocument.model_validate(doc_data)
            norm_kicker = normalize_category(doc.kicker)
            if norm_kicker != doc.kicker:
                doc = doc.model_copy(update={"kicker": norm_kicker})
            self._documents[doc_id] = doc

        with postings_path.open("r", encoding="utf-8") as fh:
            raw_postings = json.load(fh)
        for field_name, term_map in raw_postings.items():
            field = IndexField(field_name)
            self._postings[field] = {
                term: {doc_id: int(tf) for doc_id, tf in doc_map.items()}
                for term, doc_map in term_map.items()
            }
        st = manifest_path.stat()
        self._manifest_sig = (st.st_mtime_ns, st.st_size)
        self._raw_vocab_df = None
        self._raw_vocab_buckets = None

    def refresh_if_changed(self) -> bool:
        """Reload index files from disk if manifest.json was modified externally."""
        manifest_path = self._index_dir / _MANIFEST_FILE
        docs_path = self._index_dir / _DOCUMENTS_FILE
        postings_path = self._index_dir / _POSTINGS_FILE
        if not (manifest_path.is_file() and docs_path.is_file() and postings_path.is_file()):
            return False
        st = manifest_path.stat()
        sig = (st.st_mtime_ns, st.st_size)
        if self._manifest_sig == sig:
            return False
        self._load_if_exists()
        return True

    def _save(self) -> None:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": 1,
            "k1": self._k1,
            "b": self._b,
            "doc_count": len(self._documents),
            "field_weights": {f.value: w for f, w in self._field_weights.items()},
            "field_total_lengths": {
                f.value: length for f, length in self._field_total_lengths.items()
            },
        }
        docs_payload = {
            doc_id: doc.model_dump(mode="json")
            for doc_id, doc in self._documents.items()
        }
        postings_payload = {
            field.value: term_map for field, term_map in self._postings.items()
        }
        _atomic_write_json(self._index_dir / _MANIFEST_FILE, manifest)
        _atomic_write_json(self._index_dir / _DOCUMENTS_FILE, docs_payload)
        _atomic_write_json(self._index_dir / _POSTINGS_FILE, postings_payload)
        st = (self._index_dir / _MANIFEST_FILE).stat()
        self._manifest_sig = (st.st_mtime_ns, st.st_size)

    def ensure_persisted(self) -> None:
        """Create on-disk index files if they do not exist without clearing existing data."""
        manifest_path = self._index_dir / _MANIFEST_FILE
        if not manifest_path.is_file():
            self._save()

    def index_articles(
        self,
        articles: Sequence[Article],
        *,
        source_by_feed_id: Mapping[int, str] | None = None,
        predicted_categories: (
            Mapping[str, tuple[str, float] | tuple[str, float, str | None, float | None]] | None
        ) = None,
    ) -> int:
        """Incrementally index unseen articles without dropping or clearing existing records."""
        newly_indexed = 0

        for article in articles:
            if article.doc_id in self._documents:
                continue

            normalized_kicker = normalize_category(article.kicker)
            token_source = (
                article.model_copy(update={"kicker": normalized_kicker})
                if normalized_kicker != article.kicker
                else article
            )
            field_tokens = tokenize_article_fields(token_source)
            field_lengths = {field: len(tokens) for field, tokens in field_tokens.items()}

            for field, tokens in field_tokens.items():
                self._field_total_lengths[field] += len(tokens)
                field_postings = self._postings[field]
                for term, tf in Counter(tokens).items():
                    term_docs = field_postings.setdefault(term, {})
                    term_docs[article.doc_id] = tf

            source = (
                source_by_feed_id.get(article.feed_id)
                if source_by_feed_id is not None
                else None
            )
            pred_cat: str | None = None
            pred_conf: float | None = None
            sec_cat: str | None = None
            sec_conf: float | None = None
            if predicted_categories and article.doc_id in predicted_categories:
                pred_tuple = predicted_categories[article.doc_id]
                if len(pred_tuple) == 4:
                    pred_cat, pred_conf, sec_cat, sec_conf = pred_tuple
                else:
                    pred_cat, pred_conf = pred_tuple

            self._documents[article.doc_id] = IndexedDocument(
                doc_id=article.doc_id,
                feed_id=article.feed_id,
                url=article.url,
                canonical_url=article.canonical_url,
                title=article.title,
                kicker=normalized_kicker,
                author=article.author,
                source=source,
                published_date=article.published_date,
                body_text=article.body_text,
                predicted_category=pred_cat,
                category_confidence=pred_conf,
                secondary_category=sec_cat,
                secondary_confidence=sec_conf,
                field_lengths=field_lengths,
            )
            newly_indexed += 1

        if newly_indexed > 0:
            self._raw_vocab_df = None
            self._raw_vocab_buckets = None
        if newly_indexed > 0 or not (self._index_dir / _MANIFEST_FILE).is_file():
            self._save()

        return newly_indexed

    def sync_from_db(
        self,
        conn: sqlite3.Connection,
        *,
        apply_ml_classifier: bool = True,
    ) -> IndexSyncResult:
        """Read unindexed articles from SQLite and add them incrementally to the search index."""
        rows = conn.execute(
            """
            SELECT a.*, f.source AS feed_source
            FROM articles AS a
            LEFT JOIN rss_feeds AS f ON f.id = a.feed_id
            ORDER BY a.fetched_at ASC, a.doc_id ASC;
            """
        ).fetchall()

        total_in_db = len(rows)
        pending_articles: list[Article] = []
        source_by_feed_id: dict[int, str] = {}

        for row in rows:
            if row["feed_source"]:
                source_by_feed_id[int(row["feed_id"])] = str(row["feed_source"])
            if row["doc_id"] in self._documents:
                continue
            pending_articles.append(
                Article(
                    doc_id=row["doc_id"],
                    feed_id=row["feed_id"],
                    url=row["url"],
                    canonical_url=row["canonical_url"],
                    title=row["title"],
                    published_date=row["published_date"],
                    author=row["author"],
                    kicker=row["kicker"],
                    body_text=row["body_text"],
                    content_hash=row["content_hash"],
                    fetched_at=row["fetched_at"],
                )
            )

        predicted_map: (
            dict[str, tuple[str, float, str | None, float | None]] | None
        ) = None
        if apply_ml_classifier and pending_articles:
            from src.search.classifier import filter_borderline_secondary, get_cached_classifier

            ml_clf = get_cached_classifier()
            if ml_clf is not None:
                preds = ml_clf.predict_top2_batch(
                    [(a.title, a.body_text) for a in pending_articles]
                )
                predicted_map = {}
                for article, (cat1, conf1, cat2, conf2) in zip(pending_articles, preds):
                    sec_cat, sec_conf = filter_borderline_secondary(conf1, cat2, conf2)
                    predicted_map[article.doc_id] = (cat1, conf1, sec_cat, sec_conf)

        newly_indexed = self.index_articles(
            pending_articles,
            source_by_feed_id=source_by_feed_id,
            predicted_categories=predicted_map,
        )
        return IndexSyncResult(
            total_in_db=total_in_db,
            newly_indexed=newly_indexed,
            already_indexed=total_in_db - len(pending_articles),
            total_indexed=self.doc_count,
            index_dir=str(self._index_dir),
        )

    def rebuild_from_db(
        self,
        conn: sqlite3.Connection,
        *,
        apply_ml_classifier: bool = True,
    ) -> IndexSyncResult:
        """Reset in-memory index structures and rebuild cleanly from all SQLite articles."""
        self._documents.clear()
        self._postings = {field: {} for field in IndexField}
        self._field_total_lengths = {field: 0 for field in IndexField}
        self._raw_vocab_df = None
        self._raw_vocab_buckets = None
        result = self.sync_from_db(conn, apply_ml_classifier=apply_ml_classifier)
        if result.total_in_db == 0:
            self._save()
        return result

    @staticmethod
    def _field_text(doc: IndexedDocument, field: IndexField) -> str:
        if field == IndexField.TITLE:
            return doc.title
        if field == IndexField.KICKER:
            return doc.kicker or ""
        if field == IndexField.AUTHOR:
            return doc.author or ""
        if field == IndexField.BODY:
            return doc.body_text
        return ""

    @staticmethod
    def _phrase_matches_field_text(
        field_text: str,
        raw_phrase: str,
        phrase_stems: Sequence[str],
    ) -> bool:
        if not field_text or not phrase_stems:
            return False
        target = " " + " ".join(phrase_stems) + " "
        if target in _cached_field_stem_str(field_text):
            return True
        if raw_phrase.lower() not in field_text.lower():
            return False
        return bool(re.search(rf"\b{re.escape(raw_phrase)}\b", field_text, re.IGNORECASE))

    @staticmethod
    def _doc_effective_category(doc: IndexedDocument) -> str | None:
        if doc.predicted_category:
            return doc.predicted_category.strip().lower()
        return normalize_category(doc.kicker)

    def _ensure_raw_vocab(self) -> dict[tuple[str, int], list[tuple[str, int]]]:
        if self._raw_vocab_buckets is not None:
            return self._raw_vocab_buckets
        vocab: Counter[str] = Counter()
        for doc in self._documents.values():
            words = set(_RAW_WORD_RE.findall(f"{doc.title} {doc.body_text[:2000]}".lower()))
            for w in words:
                if w not in ENGLISH_STOPWORDS:
                    vocab[w] += 1
        buckets: dict[tuple[str, int], list[tuple[str, int]]] = {}
        for word, df in vocab.items():
            buckets.setdefault((word[0], len(word)), []).append((word, df))
        for bucket in buckets.values():
            bucket.sort(key=lambda item: (-item[1], item[0]))
        self._raw_vocab_df = vocab
        self._raw_vocab_buckets = buckets
        return buckets

    def suggest_spelling(self, query: str) -> str | None:
        """Suggest a corrected query when a non-stopword query word has zero index matches."""
        if not query or not query.strip() or self.doc_count == 0:
            return None
        title_postings = self._postings[IndexField.TITLE]
        body_postings = self._postings[IndexField.BODY]
        buckets = self._ensure_raw_vocab()
        if not buckets:
            return None

        tokens = re.findall(r"[A-Za-z0-9'-]+", query)
        corrected_tokens: list[str] = []
        changed = False

        for raw_tok in tokens:
            lower = raw_tok.lower()
            if not lower.isalpha() or len(lower) < 4 or lower in ENGLISH_STOPWORDS:
                corrected_tokens.append(raw_tok)
                continue
            stemmed = stem_token(lower)
            if stemmed in title_postings or stemmed in body_postings:
                corrected_tokens.append(raw_tok)
                continue

            best_word: str | None = None
            best_key: tuple[int, int, str] | None = None
            max_dist = 1 if len(lower) <= 4 else 2
            first_char = lower[0]
            qlen = len(lower)

            for length in range(max(3, qlen - max_dist), qlen + max_dist + 1):
                for cand, df in buckets.get((first_char, length), ())[:250]:
                    dist = _edit_distance_le2(lower, cand)
                    if dist == 0 or dist > max_dist:
                        continue
                    key = (dist, -df, cand)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_word = cand
                    if dist == 1:
                        break

            if best_word is not None:
                corrected_tokens.append(best_word)
                changed = True
            else:
                corrected_tokens.append(raw_tok)

        if not changed:
            return None
        suggestion = " ".join(corrected_tokens)
        if suggestion.lower() == query.strip().lower():
            return None
        return suggestion

    def _matches_filters(
        self,
        doc: IndexedDocument,
        *,
        category: str | None = None,
        source: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> bool:
        if category:
            effective = self._doc_effective_category(doc)
            if doc.predicted_category:
                canon_filter = map_to_canonical_category(category) or normalize_category(category)
                if not effective or effective != canon_filter:
                    return False
            else:
                norm_filter = normalize_category(category)
                if not effective or effective != norm_filter:
                    return False
        if source:
            if not doc.source or doc.source.lower() != source.strip().lower():
                return False
        if date_from or date_to:
            if doc.published_date is None:
                return False
            if date_from and doc.published_date < date_from:
                return False
            if date_to and doc.published_date > date_to:
                return False
        return True

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        category: str | None = None,
        source: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        extra_terms: Mapping[str, float] | None = None,
        expansion_phrases: Sequence[tuple[str, Sequence[str], float]] | None = None,
        concept_groups: Sequence[Sequence[str]] | None = None,
        min_should_match: float = 0.6,
        allow_empty_with_filters: bool = False,
        sort: str = "relevance",
        apply_recency_boost: bool = False,
    ) -> list[SearchHit]:
        """Score and rank matching documents using read-only multi-field Okapi BM25 with quorum."""
        if self.doc_count == 0 or top_k <= 0:
            return []

        if not query.strip():
            has_filter = bool(
                (category and category.strip())
                or (source and source.strip())
                or date_from is not None
                or date_to is not None
            )
            if not (allow_empty_with_filters and has_filter):
                return []
            matching = [
                doc
                for doc in self._documents.values()
                if self._matches_filters(
                    doc,
                    category=category,
                    source=source,
                    date_from=date_from,
                    date_to=date_to,
                )
            ]
            matching.sort(
                key=lambda doc: (
                    -(doc.published_date.timestamp() if doc.published_date is not None else 0.0),
                    doc.doc_id,
                )
            )
            return [
                SearchHit(
                    doc_id=doc.doc_id,
                    feed_id=doc.feed_id,
                    url=doc.url,
                    canonical_url=doc.canonical_url,
                    title=doc.title,
                    kicker=doc.kicker,
                    author=doc.author,
                    source=doc.source,
                    published_date=doc.published_date,
                    body_text=doc.body_text,
                    predicted_category=doc.predicted_category,
                    category_confidence=doc.category_confidence,
                    score=0.0,
                    matched_fields=[],
                    field_scores={},
                )
                for doc in matching[:top_k]
            ]

        n_docs = self.doc_count
        scores: dict[str, float] = {}
        matched_fields_map: dict[str, set[IndexField]] = {}
        field_scores_map: dict[str, dict[IndexField, float]] = {}
        doc_matched_terms: dict[str, set[str]] = {}
        quoted_phrases = [
            (raw_p.strip(), preprocess_text(raw_p))
            for raw_p in _QUOTED_PHRASE_RE.findall(query)
            if raw_p.strip() and preprocess_text(raw_p)
        ]

        for field in IndexField:
            weight = self._field_weights.get(field, 0.0)
            if weight <= 0.0:
                continue

            base_terms = preprocess_query_for_field(query, field)
            weighted_terms: list[tuple[str, float]] = [(term, 1.0) for term in base_terms]
            if extra_terms and field != IndexField.PUBLISHED_DATE:
                seen_base = set(base_terms)
                for exp_term, exp_weight in extra_terms.items():
                    if exp_term not in seen_base and exp_weight > 0.0:
                        weighted_terms.append((exp_term, float(exp_weight)))

            has_phrases = bool(expansion_phrases) and field != IndexField.PUBLISHED_DATE
            if not weighted_terms and not has_phrases:
                continue

            field_postings = self._postings[field]
            total_field_len = self._field_total_lengths[field]
            avgdl = (total_field_len / n_docs) if total_field_len > 0 else 1.0

            for term, term_boost in weighted_terms:
                doc_tfs = field_postings.get(term)
                if not doc_tfs:
                    continue

                df = len(doc_tfs)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

                for doc_id, tf in doc_tfs.items():
                    doc = self._documents[doc_id]
                    if not self._matches_filters(
                        doc,
                        category=category,
                        source=source,
                        date_from=date_from,
                        date_to=date_to,
                    ):
                        continue

                    dl = doc.field_lengths.get(field, 0)
                    denom = tf + self._k1 * (1.0 - self._b + self._b * (dl / avgdl))
                    bm25_term = idf * ((tf * (self._k1 + 1.0)) / denom)
                    contribution = term_boost * weight * bm25_term
                    scores[doc_id] = scores.get(doc_id, 0.0) + contribution
                    matched_fields_map.setdefault(doc_id, set()).add(field)
                    doc_fscores = field_scores_map.setdefault(doc_id, {})
                    doc_fscores[field] = doc_fscores.get(field, 0.0) + contribution
                    doc_matched_terms.setdefault(doc_id, set()).add(term)

            if has_phrases and expansion_phrases:
                for raw_phrase, phrase_stems, phrase_weight in expansion_phrases:
                    if not phrase_stems or phrase_weight <= 0.0:
                        continue
                    stem_doc_maps = [field_postings.get(stem) for stem in phrase_stems]
                    if any(doc_map is None or not doc_map for doc_map in stem_doc_maps):
                        continue
                    non_empty_maps = [m for m in stem_doc_maps if m is not None]
                    candidate_ids = set(non_empty_maps[0].keys())
                    for doc_map in non_empty_maps[1:]:
                        candidate_ids &= doc_map.keys()
                        if not candidate_ids:
                            break
                    if not candidate_ids:
                        continue

                    phrase_token = f"__phrase__:{'_'.join(phrase_stems)}"
                    stem_idfs = [
                        math.log(1.0 + (n_docs - len(doc_map) + 0.5) / (len(doc_map) + 0.5))
                        for doc_map in non_empty_maps
                    ]
                    for doc_id in candidate_ids:
                        doc = self._documents[doc_id]
                        if not self._matches_filters(
                            doc,
                            category=category,
                            source=source,
                            date_from=date_from,
                            date_to=date_to,
                        ):
                            continue
                        if not self._phrase_matches_field_text(
                            self._field_text(doc, field),
                            raw_phrase,
                            phrase_stems,
                        ):
                            continue

                        dl = doc.field_lengths.get(field, 0)
                        stem_scores = []
                        for doc_map, idf in zip(non_empty_maps, stem_idfs):
                            tf = doc_map[doc_id]
                            denom = tf + self._k1 * (1.0 - self._b + self._b * (dl / avgdl))
                            stem_scores.append(idf * ((tf * (self._k1 + 1.0)) / denom))
                        mean_bm25 = sum(stem_scores) / len(stem_scores)
                        contribution = float(phrase_weight) * weight * mean_bm25
                        scores[doc_id] = scores.get(doc_id, 0.0) + contribution
                        matched_fields_map.setdefault(doc_id, set()).add(field)
                        doc_fscores = field_scores_map.setdefault(doc_id, {})
                        doc_fscores[field] = doc_fscores.get(field, 0.0) + contribution
                        doc_matched_terms.setdefault(doc_id, set()).add(phrase_token)

        # Enforce explicit quoted phrases when present
        if quoted_phrases and scores:
            for doc_id in list(scores.keys()):
                doc = self._documents[doc_id]
                matched_all = True
                for raw_p, p_stems in quoted_phrases:
                    if not (
                        self._phrase_matches_field_text(doc.title, raw_p, p_stems)
                        or self._phrase_matches_field_text(doc.body_text, raw_p, p_stems)
                    ):
                        matched_all = False
                        break
                if not matched_all:
                    del scores[doc_id]
                    matched_fields_map.pop(doc_id, None)
                    field_scores_map.pop(doc_id, None)

        # Enforce quorum (minimum non-stopword concept overlap) & coordinate Soft-AND penalty
        if concept_groups is not None:
            resolved_concepts: list[set[str]] = [set(group) for group in concept_groups if group]
        else:
            date_tokens = preprocess_query_for_field(query, IndexField.PUBLISHED_DATE)
            date_re = re.compile(r"\b\d{4}(?:-\d{2}(?:-\d{2})?)?\b")
            non_date_query = date_re.sub(" ", query)
            seen_default: set[str] = set()
            resolved_concepts = []
            for dt in date_tokens:
                if dt not in seen_default:
                    seen_default.add(dt)
                    resolved_concepts.append({dt})
            for term in preprocess_text(non_date_query):
                if term not in seen_default:
                    seen_default.add(term)
                    resolved_concepts.append({term})

        explicit_years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))

        num_concepts = len(resolved_concepts)
        if num_concepts > 1 and scores:
            clamped_msm = max(0.0, min(1.0, min_should_match))
            required_concepts = (
                max(1, math.ceil(num_concepts * clamped_msm)) if clamped_msm > 0.0 else 1
            )
            for doc_id in list(scores.keys()):
                matched_set = doc_matched_terms.get(doc_id, set())
                if explicit_years and not (explicit_years & matched_set):
                    del scores[doc_id]
                    matched_fields_map.pop(doc_id, None)
                    field_scores_map.pop(doc_id, None)
                    continue
                matched_concepts = sum(1 for group in resolved_concepts if group & matched_set)
                if matched_concepts < required_concepts:
                    del scores[doc_id]
                    matched_fields_map.pop(doc_id, None)
                    field_scores_map.pop(doc_id, None)
                elif matched_concepts < num_concepts:
                    coord_factor = (matched_concepts / num_concepts) ** 2
                    scores[doc_id] *= coord_factor
                    doc_fscores = field_scores_map.get(doc_id)
                    if doc_fscores:
                        for f in doc_fscores:
                            doc_fscores[f] *= coord_factor

        # Ordered bigram / exact phrase proximity boost for multi-word queries
        query_stems = preprocess_text(query)
        if len(query_stems) >= 2 and scores:
            bigrams = [
                f" {query_stems[i]} {query_stems[i + 1]} "
                for i in range(len(query_stems) - 1)
            ]
            full_phrase = " " + " ".join(query_stems) + " "
            for doc_id in list(scores.keys()):
                doc = self._documents[doc_id]
                doc_fscores = field_scores_map.get(doc_id, {})
                title_stems = _cached_field_stem_str(doc.title)
                body_stems = _cached_field_stem_str(doc.body_text)

                if IndexField.TITLE in doc_fscores:
                    if full_phrase in title_stems:
                        boost = doc_fscores[IndexField.TITLE] * 0.40
                    elif any(bg in title_stems for bg in bigrams):
                        boost = doc_fscores[IndexField.TITLE] * 0.25
                    else:
                        boost = 0.0
                    if boost > 0.0:
                        doc_fscores[IndexField.TITLE] += boost
                        scores[doc_id] += boost

                if IndexField.BODY in doc_fscores:
                    if full_phrase in body_stems:
                        boost = doc_fscores[IndexField.BODY] * 0.25
                    elif any(bg in body_stems for bg in bigrams):
                        boost = doc_fscores[IndexField.BODY] * 0.12
                    else:
                        boost = 0.0
                    if boost > 0.0:
                        doc_fscores[IndexField.BODY] += boost
                        scores[doc_id] += boost

        # Mild freshness tie-breaker when recency boost is enabled in relevance mode
        if apply_recency_boost and sort == "relevance" and scores:
            dated_ts = [
                self._documents[d].published_date.timestamp()
                for d in scores
                if self._documents[d].published_date is not None
            ]
            if dated_ts:
                ref_ts = max(dated_ts)
                for doc_id in list(scores.keys()):
                    pub = self._documents[doc_id].published_date
                    if pub is None:
                        continue
                    age_days = max(0.0, (ref_ts - pub.timestamp()) / 86400.0)
                    mult = 1.0 + 0.10 * math.exp(-age_days / 14.0)
                    scores[doc_id] *= mult
                    doc_fscores = field_scores_map.get(doc_id)
                    if doc_fscores:
                        for f in doc_fscores:
                            doc_fscores[f] *= mult

        if sort == "latest":
            ranked_ids = sorted(
                scores.keys(),
                key=lambda doc_id: (
                    -(
                        self._documents[doc_id].published_date.timestamp()
                        if self._documents[doc_id].published_date is not None
                        else 0.0
                    ),
                    -scores[doc_id],
                    doc_id,
                ),
            )[:top_k]
        else:
            ranked_ids = sorted(
                scores.keys(),
                key=lambda doc_id: (
                    -scores[doc_id],
                    -(
                        self._documents[doc_id].published_date.timestamp()
                        if self._documents[doc_id].published_date is not None
                        else 0.0
                    ),
                    doc_id,
                ),
            )[:top_k]

        results: list[SearchHit] = []
        for doc_id in ranked_ids:
            doc = self._documents[doc_id]
            ordered_fields = [
                f for f in IndexField if f in matched_fields_map.get(doc_id, set())
            ]
            doc_fscores = field_scores_map.get(doc_id, {})
            ordered_fscores = {
                f: round(doc_fscores[f], 4) for f in ordered_fields if f in doc_fscores
            }
            results.append(
                SearchHit(
                    doc_id=doc.doc_id,
                    feed_id=doc.feed_id,
                    url=doc.url,
                    canonical_url=doc.canonical_url,
                    title=doc.title,
                    kicker=doc.kicker,
                    author=doc.author,
                    source=doc.source,
                    published_date=doc.published_date,
                    body_text=doc.body_text,
                    predicted_category=doc.predicted_category,
                    category_confidence=doc.category_confidence,
                    secondary_category=doc.secondary_category,
                    secondary_confidence=doc.secondary_confidence,
                    score=round(scores[doc_id], 6),
                    matched_fields=ordered_fields,
                    field_scores=ordered_fscores,
                )
            )
        return results

    def corpus_overview(self) -> dict[str, object]:
        """Return read-only corpus coverage, live facet counts, and sample queries."""
        category_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        dates: list[datetime] = []

        for doc in self._documents.values():
            eff_cat = self._doc_effective_category(doc)
            if eff_cat:
                category_counts[eff_cat] += 1
            if doc.source:
                source_counts[doc.source] += 1
            if doc.published_date is not None:
                dates.append(doc.published_date)

        # Pick guaranteed-matching sample queries from indexed titles
        sample_queries: list[str] = []
        seen_samples: set[str] = set()
        for doc in sorted(
            self._documents.values(),
            key=lambda d: (
                -(d.published_date.timestamp() if d.published_date else 0.0),
                d.doc_id,
            ),
        ):
            words = [
                w
                for w in doc.title.split()
                if len(w) >= 4 and w.isalpha() and w.lower() not in ENGLISH_STOPWORDS
            ]
            if len(words) >= 2:
                candidate = " ".join(words[:2]).lower()
            elif words:
                candidate = words[0].lower()
            else:
                continue
            if candidate not in seen_samples:
                seen_samples.add(candidate)
                sample_queries.append(candidate)
            if len(sample_queries) >= 5:
                break

        return {
            "doc_count": self.doc_count,
            "categories": [
                {"name": name, "count": count}
                for name, count in sorted(category_counts.items(), key=lambda x: (-x[1], x[0].lower()))
            ],
            "sources": [
                {"name": name, "count": count}
                for name, count in sorted(source_counts.items(), key=lambda x: (-x[1], x[0].lower()))
            ],
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
            "sample_queries": sample_queries,
        }

    def get_document(self, doc_id: str) -> IndexedDocument | None:
        """Retrieve an indexed document record by doc_id without mutating the index."""
        return self._documents.get(doc_id)

    def stats(self) -> dict[str, object]:
        """Return read-only index statistics."""
        return {
            "index_dir": str(self._index_dir),
            "doc_count": self.doc_count,
            "field_weights": {f.value: w for f, w in self._field_weights.items()},
            "vocabulary_sizes": {
                field.value: len(term_map) for field, term_map in self._postings.items()
            },
            "last_checked_utc": to_iso8601_utc(utc_now()),
        }
