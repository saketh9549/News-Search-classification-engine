"""Lightweight deterministic news category classifier with calibrated confidence."""

from __future__ import annotations

import math
import re

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
        "final": 1.6,
        "semifinal": 2.5,
        "playoff": 2.5,
        "icc": 3.0,
        "bcci": 3.0,
        "fifa": 3.0,
        "ipl": 3.0,
        "u19": 3.0,
        "under": 0.6,
        "odi": 3.0,
        "t20": 3.0,
        "athletics": 2.8,
        "badminton": 3.0,
        "hockey": 2.8,
        "chess": 2.6,
        "kohli": 3.0,
        "sharma": 2.2,
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
        "model": 1.4,
        "llm": 3.0,
        "robotics": 3.0,
        "robot": 2.6,
        "satellite": 2.4,
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
        "data": 1.5,
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
        "quarter": 1.8,
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
        "middle": 1.4,
        "gaza": 2.8,
        "israel": 2.8,
        "iran": 2.6,
        "ceasefire": 3.0,
        "sanctions": 2.8,
        "climate": 2.6,
        "emissions": 2.6,
        "cop": 2.4,
        "refugee": 2.6,
        "foreign": 2.0,
        "border": 1.8,
        "conflict": 2.4,
        "war": 2.2,
        "united": 1.4,
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
        "rupee": 2.0,
        "lok": 2.6,
        "sabha": 2.6,
        "rajya": 2.6,
        "bjp": 2.8,
        "congress": 2.2,
        "monsoon": 2.4,
        "niti": 2.6,
    },
    "national": {
        "national": 3.0,
        "parliament": 2.8,
        "bill": 2.0,
        "cabinet": 2.6,
        "ministry": 2.5,
        "minister": 2.2,
        "election": 2.6,
        "elections": 2.6,
        "voters": 2.4,
        "constitution": 2.8,
        "supreme": 2.2,
        "court": 2.0,
        "judicial": 2.4,
        "legislation": 2.8,
        "lawmakers": 2.6,
        "senate": 2.6,
        "federal": 2.2,
        "policy": 2.0,
        "police": 2.0,
        "infrastructure": 2.2,
        "railway": 2.2,
        "highway": 2.0,
        "healthcare": 2.0,
        "education": 2.0,
        "census": 2.6,
        "budget": 2.2,
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
    "science": "technology",
    "ai": "technology",
    "business": "business",
    "economy": "business",
    "markets": "business",
    "finance": "business",
    "world": "world",
    "international": "world",
    "global": "world",
    "climate": "world",
    "india": "india",
    "states": "india",
    "national": "national",
    "politics": "national",
    "policy": "national",
    "top_stories": "national",
}


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


def predict_category(
    title: str,
    body_text: str,
    *,
    kicker: str | None = None,
) -> tuple[str, float]:
    """Predict topical news category and calibrated softmax confidence [0, 1]."""
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

    best_cat = max(scores, key=lambda cat: (scores[cat], cat))
    best_raw = scores[best_cat]
    if best_raw <= 0.0:
        if kicker:
            normalized = kicker.strip().lower()
            mapped = _KICKER_ALIASES.get(normalized, normalized)
            if mapped in scores:
                return mapped, 0.5
        return "world", round(1.0 / len(scores), 4)

    # Temperature-scaled softmax over log-compressed lexical evidence for calibrated confidence.
    temperature = 1.6
    scaled = {cat: math.log1p(val) * temperature for cat, val in scores.items()}
    max_s = max(scaled.values())
    exps = {cat: math.exp(val - max_s) for cat, val in scaled.items()}
    denom = sum(exps.values()) or 1.0
    confidence = exps[best_cat] / denom
    return best_cat, round(confidence, 4)
