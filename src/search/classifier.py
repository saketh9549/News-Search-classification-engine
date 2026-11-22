"""Supervised ML news category classifier (TF-IDF + Calibrated LinearSVC) with lexical fallback."""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.models.schemas import CANONICAL_CATEGORIES, map_to_canonical_category

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("data/models/category_classifier.joblib")
DEFAULT_METRICS_PATH = Path("data/models/classifier_metrics.json")
BORDERLINE_CONFIDENCE_THRESHOLD = 0.50
MIN_SECONDARY_CONFIDENCE = 0.15


def filter_borderline_secondary(
    primary_confidence: float,
    secondary_category: str | None,
    secondary_confidence: float | None,
    *,
    threshold: float = BORDERLINE_CONFIDENCE_THRESHOLD,
    min_secondary: float = MIN_SECONDARY_CONFIDENCE,
) -> tuple[str | None, float | None]:
    """Keep secondary category prediction only when primary confidence is borderline."""
    if (
        primary_confidence < threshold
        and secondary_category is not None
        and secondary_confidence is not None
        and secondary_confidence >= min_secondary
    ):
        return secondary_category, secondary_confidence
    return None, None

_TOKEN_RE = re.compile(r"[a-z0-9]+")

CATEGORY_KEYWORDS: dict[str, dict[str, float]] = {
    "sport": {
        "cricket": 3.0,
        "football": 3.0,
        "soccer": 3.0,
        "tennis": 3.0,
        "olympics": 3.0,
        "tournament": 2.5,
        "championship": 2.5,
        "league": 2.2,
        "match": 2.0,
        "matches": 2.0,
        "wicket": 3.0,
        "wickets": 3.0,
        "innings": 3.0,
        "batsman": 3.0,
        "batter": 2.8,
        "bowler": 3.0,
        "bowling": 2.8,
        "century": 2.2,
        "goal": 2.0,
        "goals": 2.0,
        "stadium": 2.5,
        "coach": 2.2,
        "captain": 2.2,
        "medal": 2.5,
        "trophy": 2.5,
        "semifinal": 2.5,
        "playoff": 2.5,
        "icc": 3.0,
        "bcci": 3.0,
        "fifa": 3.0,
        "ipl": 3.0,
        "u19": 3.0,
        "odi": 3.0,
        "t20": 3.0,
        "athletics": 2.8,
        "badminton": 3.0,
        "hockey": 2.8,
        "chess": 2.6,
        "kohli": 3.0,
        "dhoni": 3.0,
        "sport": 3.5,
        "sports": 3.5,
    },
    "technology": {
        "technology": 3.5,
        "tech": 3.0,
        "software": 2.8,
        "hardware": 2.8,
        "ai": 3.0,
        "artificial": 2.0,
        "intelligence": 1.8,
        "semiconductor": 3.5,
        "semiconductors": 3.5,
        "chip": 2.6,
        "chips": 2.6,
        "silicon": 2.8,
        "foundry": 2.8,
        "processor": 2.8,
        "gpu": 3.0,
        "cpu": 3.0,
        "cloud": 2.2,
        "cybersecurity": 3.2,
        "cyber": 2.6,
        "hacker": 2.8,
        "algorithm": 2.8,
        "algorithms": 2.8,
        "llm": 3.0,
        "robotics": 3.0,
        "robot": 2.6,
        "smartphone": 2.8,
        "app": 2.0,
        "apps": 2.0,
        "digital": 2.0,
        "internet": 2.2,
        "telecom": 2.4,
        "quantum": 2.8,
        "apple": 2.0,
        "google": 2.4,
        "microsoft": 2.4,
        "nvidia": 3.0,
        "openai": 3.0,
        "meta": 2.0,
        "startup": 2.0,
        "automation": 2.4,
    },
    "business": {
        "business": 3.2,
        "economy": 3.0,
        "economic": 2.8,
        "market": 2.4,
        "markets": 2.4,
        "stock": 2.6,
        "stocks": 2.6,
        "shares": 2.4,
        "profit": 2.6,
        "revenue": 2.8,
        "earnings": 2.8,
        "quarterly": 2.2,
        "bank": 2.4,
        "banking": 2.8,
        "rbi": 3.2,
        "fed": 2.6,
        "inflation": 3.0,
        "gdp": 3.2,
        "trade": 2.2,
        "tariff": 2.8,
        "tariffs": 2.8,
        "export": 2.4,
        "exports": 2.4,
        "import": 2.2,
        "imports": 2.2,
        "investment": 2.5,
        "investor": 2.5,
        "investors": 2.5,
        "ipo": 3.0,
        "merger": 2.8,
        "acquisition": 2.8,
        "rupee": 2.6,
        "dollar": 2.4,
        "fiscal": 2.6,
        "tax": 2.2,
        "oil": 1.8,
        "manufacturing": 2.2,
        "finance": 2.8,
        "financial": 2.5,
        "currency": 2.6,
        "retail": 2.2,
    },
    "world": {
        "world": 3.0,
        "global": 2.4,
        "international": 2.4,
        "summit": 2.6,
        "treaty": 2.8,
        "diplomat": 2.8,
        "diplomatic": 2.8,
        "embassy": 2.8,
        "un": 2.6,
        "nato": 3.0,
        "eu": 2.6,
        "europe": 2.4,
        "european": 2.2,
        "ukraine": 2.8,
        "russia": 2.6,
        "china": 2.4,
        "beijing": 2.6,
        "washington": 2.4,
        "london": 2.2,
        "paris": 2.2,
        "tokyo": 2.2,
        "gaza": 2.8,
        "israel": 2.8,
        "iran": 2.6,
        "ceasefire": 3.0,
        "sanctions": 2.8,
        "refugee": 2.6,
        "foreign": 2.0,
        "conflict": 2.4,
        "war": 2.2,
    },
    "india": {
        "india": 3.0,
        "indian": 2.5,
        "delhi": 2.8,
        "mumbai": 2.6,
        "bengaluru": 2.6,
        "chennai": 2.6,
        "kolkata": 2.6,
        "hyderabad": 2.6,
        "kerala": 2.6,
        "maharashtra": 2.6,
        "karnataka": 2.6,
        "gujarat": 2.6,
        "tamil": 2.4,
        "punjab": 2.4,
        "isro": 2.6,
        "modi": 2.8,
        "lok": 2.6,
        "sabha": 2.6,
        "rajya": 2.6,
        "bjp": 2.8,
        "congress": 2.2,
        "monsoon": 2.4,
        "niti": 2.6,
    },
    "science_health": {
        "science": 3.2,
        "scientific": 2.8,
        "research": 2.2,
        "researchers": 2.4,
        "scientist": 2.8,
        "scientists": 2.8,
        "health": 3.2,
        "medical": 2.8,
        "hospital": 2.6,
        "disease": 2.8,
        "vaccine": 3.0,
        "cancer": 2.8,
        "patient": 2.4,
        "patients": 2.4,
        "clinical": 2.8,
        "virus": 2.6,
        "climate": 2.8,
        "emissions": 2.6,
        "nasa": 3.0,
        "space": 2.4,
        "planet": 2.6,
        "species": 2.6,
        "biology": 2.8,
        "physics": 2.8,
        "astronomy": 2.8,
    },
    "politics": {
        "politics": 3.2,
        "political": 2.8,
        "parliament": 2.8,
        "cabinet": 2.6,
        "ministry": 2.5,
        "minister": 2.2,
        "election": 2.8,
        "elections": 2.8,
        "voters": 2.6,
        "constitution": 2.8,
        "legislation": 2.8,
        "lawmakers": 2.8,
        "senate": 2.8,
        "senator": 2.6,
        "congressman": 2.6,
        "democrat": 2.6,
        "republican": 2.6,
        "labour": 2.4,
        "conservative": 2.4,
        "mp": 2.2,
        "campaign": 2.2,
        "ballot": 2.6,
    },
    "entertainment": {
        "entertainment": 3.2,
        "culture": 2.8,
        "film": 2.8,
        "movie": 2.8,
        "movies": 2.8,
        "cinema": 2.8,
        "music": 2.8,
        "album": 2.8,
        "concert": 2.6,
        "actor": 2.8,
        "actress": 2.8,
        "director": 2.2,
        "theatre": 2.8,
        "theater": 2.8,
        "novel": 2.6,
        "book": 2.2,
        "books": 2.4,
        "festival": 2.4,
        "award": 2.2,
        "awards": 2.2,
        "oscars": 3.0,
        "grammy": 3.0,
        "eurovision": 3.0,
        "television": 2.6,
        "museum": 2.6,
        "exhibition": 2.6,
        "painting": 2.6,
        "auction": 2.2,
    },
}

_KICKER_ALIASES: dict[str, str] = {
    "sport": "sport",
    "sports": "sport",
    "cricket": "sport",
    "football": "sport",
    "tennis": "sport",
    "tech": "technology",
    "technology": "technology",
    "ai": "technology",
    "business": "business",
    "economy": "business",
    "markets": "business",
    "finance": "business",
    "world": "world",
    "international": "world",
    "global": "world",
    "india": "india",
    "states": "india",
    "national": "india",
    "science": "science_health",
    "health": "science_health",
    "science_health": "science_health",
    "climate": "science_health",
    "environment": "science_health",
    "politics": "politics",
    "elections": "politics",
    "policy": "politics",
    "entertainment": "entertainment",
    "culture": "entertainment",
    "arts": "entertainment",
    "lifestyle": "entertainment",
}


_CLASSIFIER_NOISE_RE = re.compile(
    r"\b(?:"
    r"economic\s+times|hindustan\s+times|times\s+of\s+india|the\s+hindu|"
    r"indian\s+express|india\s+today|the\s+guardian|al\s+jazeera|new\s+york\s+times|"
    r"bbc\s+news|bbc\s+sport|bbc\s+sounds|bbc|reuters|getty\s+images|"
    r"associated\s+press|toi\s+app|download\s+toi|recommended\s+stories|"
    r"tech\s+life|read\s+more|sign\s+up|subscribe|follow\s+us|news\s+live|"
    r"ht|toi|download"
    r")\b",
    re.IGNORECASE,
)

_SPECIFIC_TOPICAL_CLASSES: frozenset[str] = frozenset(
    {"sport", "technology", "business", "science_health", "politics", "entertainment"}
)


def _compose_document_text(title: str, body_text: str) -> str:
    """Combine outlet-debiased title (3x weight via repetition) and body text for TF-IDF."""
    clean_title = _CLASSIFIER_NOISE_RE.sub(" ", (title or "").strip())
    clean_body = _CLASSIFIER_NOISE_RE.sub(" ", (body_text or "").strip())
    if clean_title and clean_body:
        return f"{clean_title}\n{clean_title}\n{clean_title}\n{clean_body}"
    return clean_title or clean_body


def _build_sklearn_pipeline(cv_folds: int = 3) -> Pipeline:
    """Create a pure-sklearn TF-IDF + Calibrated LinearSVC pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    max_df=0.90,
                    max_features=30000,
                    stop_words="english",
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{1,}\b",
                ),
            ),
            (
                "clf",
                CalibratedClassifierCV(
                    LinearSVC(C=1.0, class_weight="balanced", random_state=42),
                    cv=max(2, cv_folds),
                    method="sigmoid",
                ),
            ),
        ]
    )


def _extract_top_features(
    texts: list[str],
    labels: list[str],
    class_names: list[str],
    top_n: int = 8,
) -> dict[str, list[str]]:
    """Extract top discriminative TF-IDF n-grams per class using a linear SVM fit."""
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.90,
        max_features=20000,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{1,}\b",
    )
    X = vec.fit_transform(texts)
    svm = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    svm.fit(X, labels)
    feature_names = vec.get_feature_names_out()
    result: dict[str, list[str]] = {}
    if len(class_names) == 2:
        coefs = svm.coef_[0]
        top_pos = coefs.argsort()[-top_n:][::-1]
        top_neg = coefs.argsort()[:top_n]
        result[class_names[0]] = [str(feature_names[i]) for i in top_neg]
        result[class_names[1]] = [str(feature_names[i]) for i in top_pos]
        return result

    for idx, cls in enumerate(svm.classes_):
        top_idx = svm.coef_[idx].argsort()[-top_n:][::-1]
        result[str(cls)] = [str(feature_names[i]) for i in top_idx]
    return result


class NewsCategoryClassifier:
    """Supervised TF-IDF + Calibrated LinearSVC classifier with persistence and CV metrics."""

    def __init__(
        self,
        pipeline: Pipeline,
        labels: list[str],
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.labels = labels
        self.metrics = metrics or {}

    def predict_top2(self, title: str, body_text: str) -> tuple[str, float, str | None, float | None]:
        """Predict top-2 canonical categories and calibrated probabilities strictly from text."""
        text = _compose_document_text(title, body_text)
        if not text.strip():
            return "world", round(1.0 / max(1, len(self.labels)), 4), None, None
        probs = self.pipeline.predict_proba([text])[0]
        order = probs.argsort()[::-1]
        classes = [str(c) for c in self.pipeline.classes_]
        top1_idx = int(order[0])
        top1_cat = classes[top1_idx]
        top1_conf = round(float(probs[top1_idx]), 4)
        if len(order) >= 2:
            top2_idx = int(order[1])
            return top1_cat, top1_conf, classes[top2_idx], round(float(probs[top2_idx]), 4)
        return top1_cat, top1_conf, None, None

    def predict(self, title: str, body_text: str) -> tuple[str, float]:
        """Predict canonical category and calibrated probability strictly from article text."""
        label, confidence, _, _ = self.predict_top2(title, body_text)
        return label, confidence

    def predict_top2_batch(
        self, items: list[tuple[str, str]]
    ) -> list[tuple[str, float, str | None, float | None]]:
        """Batch-predict top-2 (cat1, conf1, cat2, conf2) tuples efficiently."""
        if not items:
            return []
        texts = [_compose_document_text(t, b) for t, b in items]
        prob_matrix = self.pipeline.predict_proba(texts)
        classes = [str(c) for c in self.pipeline.classes_]
        results: list[tuple[str, float, str | None, float | None]] = []
        for probs in prob_matrix:
            order = probs.argsort()[::-1]
            top1_idx = int(order[0])
            top1_cat = classes[top1_idx]
            top1_conf = round(float(probs[top1_idx]), 4)
            if len(order) >= 2:
                top2_idx = int(order[1])
                results.append(
                    (top1_cat, top1_conf, classes[top2_idx], round(float(probs[top2_idx]), 4))
                )
            else:
                results.append((top1_cat, top1_conf, None, None))
        return results

    def predict_batch(self, items: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """Batch-predict (category, confidence) pairs efficiently."""
        return [(cat1, conf1) for cat1, conf1, _, _ in self.predict_top2_batch(items)]

    def save(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        metrics_path: Path = DEFAULT_METRICS_PATH,
    ) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "labels": self.labels},
            model_path,
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(self.metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        model_path: Path = DEFAULT_MODEL_PATH,
        metrics_path: Path = DEFAULT_METRICS_PATH,
    ) -> NewsCategoryClassifier | None:
        if not model_path.is_file():
            return None
        try:
            payload = joblib.load(model_path)
            metrics: dict[str, Any] = {}
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            return cls(
                pipeline=payload["pipeline"],
                labels=list(payload["labels"]),
                metrics=metrics,
            )
        except Exception as exc:
            logger.warning("Unable to load ML classifier from %s: %s", model_path, exc)
            return None

    @classmethod
    def train_from_samples(
        cls,
        samples: list[tuple[str, str, str]],
        *,
        min_samples_per_class: int = 4,
    ) -> NewsCategoryClassifier:
        """Train and evaluate via Stratified K-Fold CV strictly on (title, body_text) without kicker leakage.

        ``samples`` is a list of ``(title, body_text, canonical_label)`` tuples.
        """
        counts: dict[str, int] = {}
        for _, _, lbl in samples:
            counts[lbl] = counts.get(lbl, 0) + 1

        valid_labels = sorted(lbl for lbl, cnt in counts.items() if cnt >= min_samples_per_class)
        if len(valid_labels) < 2:
            raise ValueError(
                f"Need at least 2 classes with >= {min_samples_per_class} samples to train ML classifier "
                f"(found {counts})"
            )

        filtered = [(t, b, lbl) for t, b, lbl in samples if lbl in valid_labels]
        texts = [_compose_document_text(t, b) for t, b, _ in filtered]
        y = [lbl for _, _, lbl in filtered]

        min_class_count = min(counts[lbl] for lbl in valid_labels)
        cv_folds = min(5, max(2, min_class_count))
        inner_cal_folds = min(3, max(2, min_class_count - 1))

        cv_pipeline = _build_sklearn_pipeline(cv_folds=inner_cal_folds)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        y_pred = cross_val_predict(cv_pipeline, texts, y, cv=skf)

        acc = float(accuracy_score(y, y_pred))
        macro_f1 = float(f1_score(y, y_pred, labels=valid_labels, average="macro", zero_division=0))
        weighted_f1 = float(
            f1_score(y, y_pred, labels=valid_labels, average="weighted", zero_division=0)
        )
        prec, rec, f1s, supp = precision_recall_fscore_support(
            y, y_pred, labels=valid_labels, zero_division=0
        )
        cm = confusion_matrix(y, y_pred, labels=valid_labels).tolist()

        per_class: dict[str, dict[str, float | int]] = {}
        for idx, lbl in enumerate(valid_labels):
            per_class[lbl] = {
                "precision": round(float(prec[idx]), 4),
                "recall": round(float(rec[idx]), 4),
                "f1": round(float(f1s[idx]), 4),
                "support": int(supp[idx]),
            }

        final_pipeline = _build_sklearn_pipeline(cv_folds=min(3, min_class_count))
        final_pipeline.fit(texts, y)
        top_features = _extract_top_features(texts, y, valid_labels)

        metrics: dict[str, Any] = {
            "model_type": "TF-IDF (1,2)-gram + Calibrated LinearSVC",
            "trained_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "num_samples": len(filtered),
            "num_classes": len(valid_labels),
            "cv_folds": cv_folds,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "labels": valid_labels,
            "per_class": per_class,
            "confusion_matrix": cm,
            "top_features": top_features,
        }
        return cls(pipeline=final_pipeline, labels=valid_labels, metrics=metrics)


def extract_training_samples_from_db(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Build high-purity (title, body_text, canonical_label) training samples from SQLite.

    Excludes audio/podcast stubs and resolves specific topical kickers (sport, business, tech,
    science_health, politics, entertainment) ahead of broad geographic feeds (world, india)
    so cross-posted articles receive accurate ground-truth section labels.
    """
    rows = conn.execute(
        """
        SELECT a.title, a.body_text, a.kicker, f.category AS feed_category
        FROM articles a
        JOIN rss_feeds f ON f.id = a.feed_id
        ORDER BY a.doc_id ASC;
        """
    ).fetchall()

    samples: list[tuple[str, str, str]] = []
    for row in rows:
        title = str(row["title"] or "")
        if "bbc sounds" in title.lower():
            continue
        feed_label = map_to_canonical_category(row["feed_category"])
        kicker_label = map_to_canonical_category(row["kicker"])

        if kicker_label in _SPECIFIC_TOPICAL_CLASSES:
            label: str | None = kicker_label
        elif feed_label in _SPECIFIC_TOPICAL_CLASSES:
            label = feed_label
        elif feed_label is not None:
            label = feed_label
        else:
            label = kicker_label

        if label is not None and label in CANONICAL_CATEGORIES:
            samples.append((title, str(row["body_text"] or ""), label))
    return samples


def train_and_save_classifier(
    conn: sqlite3.Connection,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    min_samples_per_class: int = 4,
) -> dict[str, Any]:
    """Train ML category classifier on SQLite articles, persist artifacts, and return CV metrics."""
    samples = extract_training_samples_from_db(conn)
    clf = NewsCategoryClassifier.train_from_samples(
        samples,
        min_samples_per_class=min_samples_per_class,
    )
    clf.save(model_path=model_path, metrics_path=metrics_path)
    invalidate_cached_classifier()
    return clf.metrics


def load_classifier_metrics(metrics_path: Path = DEFAULT_METRICS_PATH) -> dict[str, Any] | None:
    """Load persisted 5-fold CV classifier metrics if available."""
    if not metrics_path.is_file():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


_CACHED_ML_KEY: tuple[str, int, int] | None = None
_CACHED_ML_MODEL: NewsCategoryClassifier | None = None


def invalidate_cached_classifier() -> None:
    """Clear the process-level cached ML classifier so newly trained models reload immediately."""
    global _CACHED_ML_KEY, _CACHED_ML_MODEL
    _CACHED_ML_KEY = None
    _CACHED_ML_MODEL = None


def get_cached_classifier(
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> NewsCategoryClassifier | None:
    """Return cached ML classifier, automatically reloading if the .joblib file changed on disk."""
    global _CACHED_ML_KEY, _CACHED_ML_MODEL
    if not model_path.is_file():
        _CACHED_ML_KEY = None
        _CACHED_ML_MODEL = None
        return None
    stat = model_path.stat()
    key = (str(model_path.resolve()), stat.st_mtime_ns, stat.st_size)
    if _CACHED_ML_KEY == key and _CACHED_ML_MODEL is not None:
        return _CACHED_ML_MODEL
    loaded = NewsCategoryClassifier.load(model_path=model_path, metrics_path=metrics_path)
    _CACHED_ML_KEY = key
    _CACHED_ML_MODEL = loaded
    return loaded


def _score_tokens(tokens: list[str], weight: float, scores: dict[str, float]) -> None:
    if not tokens or weight <= 0.0:
        return
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    for category, lexicon in CATEGORY_KEYWORDS.items():
        cat_hit = 0.0
        for tok, freq in counts.items():
            kw_weight = lexicon.get(tok)
            if kw_weight:
                cat_hit += kw_weight * (1.0 + math.log1p(freq - 1))
        scores[category] += weight * cat_hit


def predict_category_lexical_top2(
    title: str,
    body_text: str,
    *,
    kicker: str | None = None,
) -> tuple[str, float, str | None, float | None]:
    """Deterministic lexical fallback returning top-2 (cat1, conf1, cat2, conf2)."""
    scores: dict[str, float] = {cat: 0.0 for cat in CATEGORY_KEYWORDS}

    title_tokens = _TOKEN_RE.findall((title or "").lower())
    kicker_tokens = _TOKEN_RE.findall((kicker or "").lower())
    body_tokens = _TOKEN_RE.findall((body_text or "").lower())

    _score_tokens(title_tokens, 2.5, scores)
    _score_tokens(kicker_tokens, 3.0, scores)
    _score_tokens(body_tokens, 1.0, scores)

    if kicker:
        normalized_kicker = kicker.strip().lower()
        aliased = _KICKER_ALIASES.get(normalized_kicker)
        if aliased in scores:
            scores[aliased] += 6.0

    ranked_cats = sorted(scores, key=lambda cat: (-scores[cat], cat))
    best_cat = ranked_cats[0]
    best_raw = scores[best_cat]
    if best_raw <= 0.0:
        if kicker:
            normalized = kicker.strip().lower()
            mapped = _KICKER_ALIASES.get(normalized, normalized)
            if mapped in scores:
                return mapped, 0.5, None, None
        return "world", round(1.0 / len(scores), 4), None, None

    temperature = 1.6
    scaled = {cat: math.log1p(val) * temperature for cat, val in scores.items()}
    max_s = max(scaled.values())
    exps = {cat: math.exp(val - max_s) for cat, val in scaled.items()}
    denom = sum(exps.values()) or 1.0
    probs = [(cat, round(exps[cat] / denom, 4)) for cat in ranked_cats]
    top1_cat, top1_conf = probs[0]
    if len(probs) >= 2 and scores[probs[1][0]] > 0.0:
        return top1_cat, top1_conf, probs[1][0], probs[1][1]
    return top1_cat, top1_conf, None, None


def predict_category_lexical(
    title: str,
    body_text: str,
    *,
    kicker: str | None = None,
) -> tuple[str, float]:
    """Deterministic lexical fallback classifier used when no trained ML model is present."""
    cat, conf, _, _ = predict_category_lexical_top2(title, body_text, kicker=kicker)
    return cat, conf


def predict_category_top2(
    title: str,
    body_text: str,
    *,
    kicker: str | None = None,
    use_ml: bool = True,
    model_path: Path = DEFAULT_MODEL_PATH,
    borderline_threshold: float = BORDERLINE_CONFIDENCE_THRESHOLD,
) -> tuple[str, float, str | None, float | None]:
    """Predict primary category and optional borderline secondary category when conf < threshold."""
    if use_ml:
        ml_model = get_cached_classifier(model_path=model_path)
        if ml_model is not None:
            ml_cat, ml_conf, ml_sec_cat, ml_sec_conf = ml_model.predict_top2(title, body_text)
            if ml_conf >= 0.45:
                sec_cat, sec_conf = filter_borderline_secondary(
                    ml_conf, ml_sec_cat, ml_sec_conf, threshold=borderline_threshold
                )
                return ml_cat, ml_conf, sec_cat, sec_conf
            lex_cat, lex_conf, lex_sec_cat, lex_sec_conf = predict_category_lexical_top2(
                title, body_text, kicker=kicker
            )
            if lex_conf >= 0.65:
                sec_cat, sec_conf = filter_borderline_secondary(
                    lex_conf, lex_sec_cat, lex_sec_conf, threshold=borderline_threshold
                )
                return lex_cat, lex_conf, sec_cat, sec_conf
            sec_cat, sec_conf = filter_borderline_secondary(
                ml_conf, ml_sec_cat, ml_sec_conf, threshold=borderline_threshold
            )
            return ml_cat, ml_conf, sec_cat, sec_conf

    lex_cat, lex_conf, lex_sec_cat, lex_sec_conf = predict_category_lexical_top2(
        title, body_text, kicker=kicker
    )
    sec_cat, sec_conf = filter_borderline_secondary(
        lex_conf, lex_sec_cat, lex_sec_conf, threshold=borderline_threshold
    )
    return lex_cat, lex_conf, sec_cat, sec_conf


def predict_category(
    title: str,
    body_text: str,
    *,
    kicker: str | None = None,
    use_ml: bool = True,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> tuple[str, float]:
    """Predict topical news category and calibrated confidence [0, 1].

    Uses the trained supervised ML model strictly on ``(title, body_text)`` when persisted on disk,
    falling back seamlessly to the deterministic lexical classifier when no model exists yet.
    """
    cat, conf, _, _ = predict_category_top2(
        title,
        body_text,
        kicker=kicker,
        use_ml=use_ml,
        model_path=model_path,
    )
    return cat, conf
