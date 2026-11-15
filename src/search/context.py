"""TREC-style Named Entity Recognition (PERSON, ORG, GPE, EVENT) and Background Linking."""

from __future__ import annotations

import functools
import re
from collections import Counter

from src.indexer.bm25_index import BM25Index
from src.indexer.preprocessing import ENGLISH_STOPWORDS, preprocess_text
from src.indexer.schemas import IndexedDocument
from src.search.schemas import BackgroundLink, EntityLabel, NamedEntity

_GPE_GAZETTEER: frozenset[str] = frozenset(
    {
        "india", "new delhi", "delhi", "mumbai", "bengaluru", "bangalore", "chennai",
        "kolkata", "hyderabad", "pune", "ahmedabad", "kashmir", "maharashtra", "kerala",
        "tamil nadu", "gujarat", "uttar pradesh", "bihar", "rajasthan", "karnataka",
        "punjab", "srinagar", "jaipur", "lucknow", "patna", "bhopal", "chandigarh",
        "guwahati", "kochi", "nagpur", "surat", "pakistan", "islamabad", "lahore",
        "karachi", "sharjah", "dubai", "abu dhabi", "saudi arabia", "riyadh", "qatar",
        "doha", "south asia", "asia", "europe", "european union", "united states",
        "us", "usa", "washington", "new york", "new orleans", "california",
        "silicon valley", "los angeles", "chicago", "boston", "san francisco",
        "united kingdom", "uk", "britain", "london", "china", "beijing", "shanghai",
        "hong kong", "taiwan", "taipei", "russia", "moscow", "ukraine", "kyiv",
        "israel", "gaza", "iran", "tehran", "turkey", "istanbul", "ankara", "syria",
        "lebanon", "afghanistan", "kabul", "bangladesh", "dhaka", "sri lanka",
        "colombo", "nepal", "kathmandu", "japan", "tokyo", "south korea", "seoul",
        "singapore", "indonesia", "jakarta", "thailand", "bangkok", "vietnam",
        "germany", "berlin", "france", "paris", "italy", "rome", "spain", "madrid",
        "brussels", "switzerland", "geneva", "ethiopia", "addis ababa", "nigeria",
        "kenya", "nairobi", "egypt", "cairo", "south africa", "johannesburg",
        "australia", "sydney", "melbourne", "new zealand", "canada", "toronto",
        "ottawa", "arctic", "pacific", "atlantic",
    }
)

# Short uppercase acronyms must match strictly in ALL-CAPS so pronouns like "who" or "us" never match.
_ACRONYM_ORGS: frozenset[str] = frozenset(
    {
        "UN", "WHO", "IMF", "NATO", "EU", "RBI", "SEBI", "ISRO", "NASA",
        "BBC", "NDTV", "BCCI", "ICC",
    }
)

# Capitalized single-word brand/company ORGs matched case-sensitively to avoid common-noun collisions.
_PROPER_ORGS: frozenset[str] = frozenset(
    {
        "Google", "Microsoft", "Apple", "Meta", "Amazon", "OpenAI", "Anthropic", "Reuters",
        "SpaceX", "Tesla", "Netflix", "Boeing", "Airbus", "Nvidia", "Oracle", "Samsung",
    }
)

# Multi-word organizations safely matched case-insensitively across body text.
_MULTIWORD_ORGS: dict[str, str] = {
    "united nations": "United Nations",
    "world health organization": "WHO",
    "world health organisation": "WHO",
    "world bank": "World Bank",
    "european union": "European Union",
    "reserve bank of india": "Reserve Bank of India",
    "bbc news": "BBC News",
    "the hindu": "The Hindu",
    "indian express": "Indian Express",
    "hindustan times": "Hindustan Times",
    "times of india": "Times of India",
    "economic times": "Economic Times",
    "india today": "India Today",
    "the guardian": "The Guardian",
    "al jazeera": "Al Jazeera",
    "new york times": "New York Times",
    "supreme court": "Supreme Court",
    "election commission": "Election Commission",
    "blue origin": "Blue Origin",
    "rockstar games": "Rockstar Games",
    "royal mail": "Royal Mail",
    "white house": "White House",
    "rajya sabha": "Rajya Sabha",
    "lok sabha": "Lok Sabha",
    "getty images": "Getty Images",
    "homeland security": "Homeland Security",
    "microsoft outlook": "Microsoft Outlook",
    "microsoft teams": "Microsoft Teams",
    "exchange online": "Exchange Online",
    "delta air lines": "Delta Air Lines",
    "royal aeronautical society": "Royal Aeronautical Society",
    "small business charter": "Small Business Charter",
    "phoenix publishing house": "Phoenix Publishing House",
    "lloyd law college": "Lloyd Law College",
    "rajaratnam school": "Rajaratnam School",
}

_ORG_SUFFIXES: frozenset[str] = frozenset(
    {
        "ministry", "commission", "authority", "council", "committee", "board",
        "institute", "university", "bank", "corporation", "corp", "ltd", "inc",
        "group", "foundation", "agency", "federation", "court", "association",
        "party", "union", "department", "organization", "organisation",
        "administration", "office", "command", "force", "forces", "army",
        "movement", "college", "school", "center", "centre", "house",
        "cathedral", "society", "charter", "lines", "airlines", "airways",
        "games", "mail", "sabha", "sangh", "kazhagam", "pathippagam",
        "ventures", "partners", "holdings", "industries", "telecom",
        "publications", "publishing", "networks", "network", "hospital",
        "museum", "embassy", "academy",
    }
)

_EVENT_KEYWORDS: frozenset[str] = frozenset(
    {
        "summit", "conference", "forum", "championship", "tournament", "cup",
        "olympics", "election", "elections", "budget", "treaty", "accord", "pact",
        "crisis", "festival", "awards", "expo", "symposium", "war",
    }
)

_EVENT_ACRONYM_RE = re.compile(r"\b(?:G7|G20|COP\d{2}|IPL|FIFA|UEFA|ASEAN|BRICS)\b")

_ORG_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9&'.-]{1,}(?:\s+(?:of|for|and|on|the))?){1,5}\s+"
    r"(?:Ministry|Commission|Authority|Council|Committee|Board|Institute|University|"
    r"Bank|Corporation|Corp|Ltd|Inc|Group|Foundation|Agency|Federation|Court|"
    r"Association|Party|Union|Department|Organization|Organisation)\b"
)

_EVENT_PHRASE_RE = re.compile(
    r"\b(?:[A-Z0-9][a-zA-Z0-9&'.-]{1,}\s+){1,4}"
    r"(?:Summit|Conference|Forum|Championship|Tournament|Cup|Olympics|"
    r"Election|Elections|Budget|Treaty|Accord|Pact|Crisis|Festival|Awards|Expo)\b"
)

_PERSON_TITLE_RE = re.compile(
    r"\b(?:Prime Minister|President|Chief Minister|Minister|Governor|Mayor|Justice|"
    r"Judge|Dr\.?|Prof\.?|CEO|Chairman|Chairperson|Director|Senator|Secretary)\s+"
    r"([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,2})\b"
)

_CAP_PERSON_RE = re.compile(r"\b([A-Z][a-z]{2,18}(?:\s+[A-Z][a-z]{2,18}){1,3})\b")

_NON_PERSON_WORDS: frozenset[str] = (
    ENGLISH_STOPWORDS
    | _ORG_SUFFIXES
    | _EVENT_KEYWORDS
    | frozenset(w for gpe in _GPE_GAZETTEER for w in gpe.split())
    | frozenset(
        {
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december",
            "amid", "despite", "following", "including", "regarding", "without", "within",
            "across", "among", "around", "near", "beyond", "toward", "towards", "upon",
            "per", "else", "also", "even", "still", "yet", "already", "since",
            "historic", "historical", "contemporary", "modern", "ancient", "classic",
            "heritage", "north", "south", "east", "west", "northern", "southern",
            "eastern", "western", "central", "global", "national", "international",
            "regional", "local", "federal", "state", "supreme", "high", "senior",
            "junior", "former", "current", "present", "future", "new", "old", "big",
            "major", "minor", "main", "primary", "clean", "renewable", "solar", "green",
            "open", "digital", "artificial", "quantum", "atomic", "nuclear", "economic",
            "financial", "political", "social", "cultural", "medical", "environmental",
            "daily", "weekly", "monthly", "annual", "special", "official", "public",
            "private", "united", "european", "indian", "british", "american", "pakistani",
            "chinese", "russian", "ukrainian", "asian", "african", "top", "breaking",
            "exclusive", "latest", "chief", "prime", "minister", "ministers", "president",
            "vice", "deputy", "governor", "mayor", "councillor", "justice", "judge",
            "prosecutor", "attorney", "ambassador", "envoy", "ceo", "chairman",
            "chairperson", "director", "senator", "secretary", "speaker", "leader",
            "commissioner", "chancellor", "general", "admiral", "colonel", "captain",
            "coach", "doctor", "prof", "professor", "spokesperson", "officials",
            "analysts", "engineers", "researchers", "police", "bench", "government",
            "cabinet", "assembly", "desk", "reporter", "correspondent", "editor",
            "pti", "ani", "afp", "reuters", "getty", "images", "image", "shutterstock",
            "alamy", "world", "mosque", "temple", "church", "palace", "fort",
            "stadium", "airport", "station", "railway", "railways", "road", "street",
            "avenue", "square", "park", "garden", "city", "town", "village",
            "district", "province", "region", "sector", "phase", "series", "match",
            "test", "league", "report", "news", "times", "post", "express", "hindu",
            "guardian", "herald", "journal", "tribune", "chronicle", "monitor",
            "today", "week", "month", "year", "edition", "software", "hardware",
            "energy", "climate", "policy", "act", "bill", "scheme", "project",
            "mission", "program", "programme", "corridor", "grid", "first", "second",
            "third", "last", "next", "live", "watch", "read", "more", "photo", "photos",
            "video", "source",
        }
    )
)

_BOILERPLATE_LINE_RE = re.compile(
    r"^(?:read\s+more|also\s+read|follow\s+us|subscribe|sign\s+up|advertisement|"
    r"published\s*:|updated\s*:|edited\s+by|reported\s+by|photo\s+credit|getty\s+images|"
    r"image\s+caption|all\s+rights\s+reserved|copyright|https?://|pti\b|ani\b|afp\b)",
    re.IGNORECASE,
)

_NON_PERSON_AUTHOR_RE = re.compile(
    r"\b(?:bureau|desk|team|staff|reporter|correspondent|news|agency|reuters|pti|ani|afp)\b",
    re.IGNORECASE,
)

# Expand GPEs, non-person words, and single-line/curated extractors after base definitions.
_GPE_GAZETTEER = _GPE_GAZETTEER | frozenset(
    {
        "latin america", "boca chica", "congo basin", "mississippi river",
        "jantar mantar", "ramlila maidan", "congo", "peru", "uganda",
        "venezuela", "antarctica", "arctic", "greenland", "siberia",
    }
)

_NON_PERSON_WORDS = _NON_PERSON_WORDS | frozenset(
    {
        "air", "missions", "mission", "race", "course", "asset", "recovery",
        "conservation", "education", "comparative", "zoology", "super", "heavy",
        "booster", "grand", "theft", "auto", "bluest", "sky", "bhagavad", "gita",
        "peruvian", "pope", "catholic", "many", "ugandans", "others", "democrats",
        "republicans", "unknown", "soldier", "soldiers", "cool", "calc",
        "individual", "neutral", "athlete", "athletes", "standard", "book",
        "books", "numbers", "number", "tech", "life", "christian", "fairy",
        "maiden", "one", "two", "three", "four", "five", "part", "woman",
        "women", "man", "men", "moon", "building", "produced", "producer",
        "related", "questions", "question", "get", "small", "business",
        "charter", "aviation", "administration", "homeland", "security",
        "forces", "force", "command", "protection", "office", "resistance",
        "army", "movement", "operation", "purgatory", "law", "college",
        "school", "center", "centre", "cathedral", "publishing", "house",
        "society", "aeronautical", "royal", "mail", "games", "rockstar",
        "rajya", "sabha", "lok", "cruise", "lines", "line", "delta", "outlook",
        "teams", "exchange", "online", "origin", "getty", "images", "veterans",
        "day", "weightlifting", "championship", "championships", "winter",
        "summer", "spring", "autumn", "paralympics", "olympics", "basin",
        "river", "valley", "congo", "mississippi", "boca", "chica", "latin",
        "america", "jantar", "mantar", "ramlila", "maidan",
        "american", "british", "indian", "chinese", "russian", "french",
        "german", "japanese", "korean", "ukrainian", "israeli", "palestinian",
        "australian", "canadian", "european", "african", "asian", "arab",
        "muslim", "hindu", "buddhist", "jewish", "protestant", "orthodox",
        "north", "south", "east", "west", "northern", "southern", "eastern",
        "western", "central", "national", "international", "federal", "global",
        "local", "public", "private", "general", "special", "senior", "junior",
        "chief", "prime", "minister", "president", "historic", "contemporary",
        "modern", "ancient", "annual", "daily", "weekly", "monthly", "world",
        "news", "report", "reports", "update", "updates", "breaking",
        "exclusive", "analysis", "opinion", "editorial", "live", "video",
        "videos", "audio", "podcast", "photo", "photos", "image", "watch",
        "listen", "read", "more", "share", "follow", "subscribe", "copyright",
        "reuters", "bbc", "cnn", "climate", "energy", "solar", "carbon",
        "health", "mental", "science", "space", "market", "markets", "trade",
        "economy", "economic", "finance", "financial", "digital", "cyber",
        "artificial", "intelligence", "quantum", "data", "cloud", "software",
        "mobile", "electric", "vehicle", "vehicles", "border", "police",
        "military", "defence", "defense", "foreign", "interior", "justice",
        "supreme", "high", "court", "parliament", "senate", "congress",
        "cabinet", "embassy", "council", "commission", "authority",
        "committee", "board", "agency", "institute", "university",
        "department", "ministry", "party", "union", "federation",
        "association", "foundation", "corporation", "group", "bank",
        "hospital", "station", "airport", "railway", "port", "city", "town",
        "village", "state", "province", "county", "district", "region",
        "street", "road", "avenue", "square", "park", "bridge", "tower",
        "palace", "temple", "mosque", "church", "shrine", "monument",
        "inside", "author", "whitening", "cream",
    }
)

_ORG_PHRASE_SL_RE = re.compile(_ORG_PHRASE_RE.pattern.replace(r"\s+", r"[ \t]+"), _ORG_PHRASE_RE.flags)
_EVENT_PHRASE_SL_RE = re.compile(_EVENT_PHRASE_RE.pattern.replace(r"\s+", r"[ \t]+"), _EVENT_PHRASE_RE.flags)
_PERSON_TITLE_SL_RE = re.compile(_PERSON_TITLE_RE.pattern.replace(r"\s+", r"[ \t]+"), _PERSON_TITLE_RE.flags)
_CAP_PERSON_SL_RE = re.compile(_CAP_PERSON_RE.pattern.replace(r"\s+", r"[ \t]+"), _CAP_PERSON_RE.flags)

_EXTRA_ORG_SUFFIX_ALTERNATION = "|".join(
    sorted(
        {
            "Administration", "Office", "Command", "Force", "Forces", "Army",
            "Movement", "College", "School", "Center", "Centre", "House",
            "Cathedral", "Society", "Charter", "Lines", "Airlines", "Airways",
            "Games", "Mail", "Sabha", "Sangh", "Kazhagam", "Pathippagam",
            "Ventures", "Partners", "Holdings", "Industries", "Telecom",
            "Publications", "Publishing", "Networks", "Network", "Hospital",
            "Museum", "Embassy", "Academy",
        },
        key=len,
        reverse=True,
    )
)
_EXTRA_ORG_PHRASE_RE = re.compile(
    rf"\b(?:[A-Z][a-zA-Z&.-]+[ \t]+){{1,4}}(?:{_EXTRA_ORG_SUFFIX_ALTERNATION})\b"
)
_EXTRA_EVENT_PHRASE_RE = re.compile(
    r"\b(?:(?:[A-Z][a-zA-Z0-9.-]+[ \t]+){1,3}(?:Championships|Paralympics|Olympiad|Classic|Day)|Operation[ \t]+[A-Z][a-zA-Z0-9.-]+(?:[ \t]+[A-Z][a-zA-Z0-9.-]+)?)\b"
)
_GEO_FEATURE_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z.-]+[ \t]+){1,3}(?:Basin|River|Valley|Mountains|Mountain|Range|Desert|Peninsula|Plateau|Strait|Bay|Gulf|Lake|Island|Islands|Canyon|Glacier|Reef|Harbor|Harbour)\b"
)

_GPE_SHORT_WORDS = sorted({g.upper() for g in _GPE_GAZETTEER if len(g) <= 3}, key=len, reverse=True)
_GPE_LONG_WORDS = sorted({g.lower() for g in _GPE_GAZETTEER if len(g) > 3}, key=len, reverse=True)
_GPE_SHORT_RE = re.compile(rf"\b(?:{'|'.join(re.escape(w) for w in _GPE_SHORT_WORDS)})\b")
_GPE_LONG_RE = re.compile(rf"\b(?:{'|'.join(re.escape(w) for w in _GPE_LONG_WORDS)})\b")

_MULTIWORD_ORG_KEYS = sorted(_MULTIWORD_ORGS.keys(), key=len, reverse=True)
_MULTIWORD_ORGS_RE = re.compile(
    rf"\b(?:{'|'.join(re.escape(k) for k in _MULTIWORD_ORG_KEYS)})\b"
)
_PROPER_ORGS_RE = re.compile(
    rf"\b(?:{'|'.join(re.escape(o) for o in sorted(_PROPER_ORGS, key=len, reverse=True))})\b"
)
_PROPER_ORGS_LOWER: frozenset[str] = frozenset(o.lower() for o in _PROPER_ORGS)
_ACRONYM_ORGS_LOWER: frozenset[str] = frozenset(a.lower() for a in _ACRONYM_ORGS)
_ACRONYM_ORGS_RE = re.compile(
    rf"(?<![A-Z])\b(?:{'|'.join(re.escape(a) for a in sorted(_ACRONYM_ORGS, key=len, reverse=True))})\b(?![A-Z])"
)

_BAD_PERSON_SUFFIX_RE = re.compile(
    r"(?:ation|ition|ution|ment|ness|logy|graphy|nomy|ships?|ishing|lifting|lding|puting|eering|keting|ics)$",
    re.IGNORECASE,
)
_PRE_PERSON_CUE_RE = re.compile(
    r"(?:^|[ \t(])(?:by|with|for|to|from|against|unlike|according[ \t]+to|led[ \t]+by|named|called|told|asked|greeted|thanked|met|joined|appointed|elected|nominated|succeeded|replaced|quoted|interviewed|signed|directed|written|reported|produced|hosted|presented|featuring|starring|alongside|including|between|and|mr\.?|mrs\.?|ms\.?|dr\.?|prof\.?|sir|dame)[ \t]+$",
    re.IGNORECASE,
)
_POST_PERSON_CUE_RE = re.compile(
    r"^(?:[ \t]+(?:said|says|told|tells|added|adds|wrote|writes|warned|warns|stated|states|announced|announces|claimed|claims|confirmed|confirms|denied|denies|argued|argues|noted|notes|explained|explains|asked|asks|urged|urges|called|calls|praised|criticized|criticised|recalled|believes|thinks|hopes|insisted|testified|posted|replied|responded|spoke|speaks|met|meets|visited|led|leads|won|wins|scored|died|resigned|survived|directed|starred|founded|worked|works|joined|attended|launched|headed|became|faces|was|is|were|has|had)\b|[ \t]*,[ \t]*(?:\d{2}\b|(?:a|an|the|who|whom|whose|former|chief|executive|senior|deputy|lead|head|president|prime|minister|mp|senator|governor|mayor|ceo|director|founder|co-founder|chair|chairman|chairperson|judge|justice|professor|dr|scientist|author|writer|journalist|reporter|correspondent|coach|captain|manager|officer|spokesperson|lawyer|attorney|analyst|economist|historian|researcher|activist|artist|actor|actress|singer|player)\b)|[ \t]*['’]s[ \t]+(?:wife|husband|son|daughter|father|mother|parents|family|lawyer|attorney|spokesperson|agent|team|office|remarks|comments|statement|speech|testimony|book|film|work|campaign|death|trial|case|career|tenure|visit)\b)",
    re.IGNORECASE,
)
_AUTHOR_SUFFIX_RE = re.compile(r"(?:,\s*(?:Jr\.?|Sr\.?|II|III|IV)\b|\s+(?:Produced|Producer|Edited|Reported)\s+by\b.*$)", re.IGNORECASE)
_AUTHOR_SPLIT_RE = re.compile(r"\s*,\s*|\s+\band\b\s+|\s*&\s*|\s*;\s*", re.IGNORECASE)


_BYLINE_LINE_RE = re.compile(
    r"^(?:by|reporting[ \t]+by|reported[ \t]+by|written[ \t]+by|analysis[ \t]+by)[ \t]+[A-Z]",
    re.IGNORECASE,
)


def _clean_body_for_ner(text: str) -> str:
    """Strip boilerplate lines, ALL-CAPS banners, and Title-Cased headline lines before NER."""
    if not text or not text.strip():
        return ""

    kept_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            continue
        if _BOILERPLATE_LINE_RE.match(line):
            continue

        alpha_words = re.findall(r"[A-Za-z]+", line)
        if len(alpha_words) == 1 and line[-1] not in ".!?":
            continue
        if len(alpha_words) >= 2 and not _BYLINE_LINE_RE.match(line):
            if line.isupper():
                continue
            cap_count = sum(1 for w in alpha_words if w[0].isupper())
            cap_ratio = cap_count / len(alpha_words)
            # Filter out Title-Cased headline/subheading lines while preserving prose sentences
            ends_with_sentence_punct = line[-1] in ".!?"
            if cap_ratio >= 0.72 and (not ends_with_sentence_punct or len(alpha_words) <= 10):
                continue
        kept_lines.append(line)

    return "\n".join(kept_lines)


def _is_plausible_person_token(word: str) -> bool:
    lower = word.strip(".").lower()
    if not lower:
        return False
    if (
        lower in _NON_PERSON_WORDS
        or lower in _GPE_GAZETTEER
        or lower in _PROPER_ORGS_LOWER
        or lower in _ACRONYM_ORGS_LOWER
    ):
        return False
    if _BAD_PERSON_SUFFIX_RE.search(lower):
        return False
    return True


def _is_valid_author_person(author: str) -> bool:
    cleaned = re.sub(r"\s+", " ", author).strip(" .,;:()[]{}\"'")
    words = cleaned.split()
    if not cleaned or not (2 <= len(words) <= 4):
        return False
    if _NON_PERSON_AUTHOR_RE.search(cleaned):
        return False
    lowered = cleaned.lower()
    if lowered in _MULTIWORD_ORGS or lowered in _GPE_GAZETTEER:
        return False
    if not all(_is_plausible_person_token(w) for w in words):
        return False
    return True


def _extract_author_people(author: str | None) -> list[str]:
    if not author or not author.strip():
        return []
    stripped = _AUTHOR_SUFFIX_RE.sub("", author)
    people: list[str] = []
    for part in _AUTHOR_SPLIT_RE.split(stripped):
        cleaned = re.sub(r"\s+", " ", part).strip(" .,;:()[]{}\"'")
        if _is_valid_author_person(cleaned):
            people.append(cleaned)
    return people


def _clean_person_words(raw_phrase: str) -> str | None:
    words = raw_phrase.split()
    while len(words) > 0 and words[0].lower() in _NON_PERSON_WORDS:
        words.pop(0)
    while len(words) > 0 and words[-1].lower() in _NON_PERSON_WORDS:
        words.pop()
    if not (2 <= len(words) <= 3):
        return None
    if not all(_is_plausible_person_token(w) for w in words):
        return None
    return " ".join(words)


def _overlaps_occupied(occupied: list[tuple[int, int]], start: int, end: int) -> bool:
    for occ_start, occ_end in occupied:
        if start < occ_end and end > occ_start:
            return True
    return False


def _has_person_context(line: str, raw_start: int, cand_end: int) -> bool:
    before = line[:raw_start]
    after = line[cand_end:]
    if _POST_PERSON_CUE_RE.match(after):
        return True
    if _PRE_PERSON_CUE_RE.search(before):
        return True
    return False


def _record_entity(
    counts: Counter[tuple[str, EntityLabel]],
    seen_spans: set[str],
    text: str,
    label: EntityLabel,
) -> None:
    cleaned = re.sub(r"\s+", " ", text).strip(" .,;:()[]{}\"'")
    if not cleaned:
        return
    counts[(cleaned, label)] += 1
    seen_spans.add(cleaned.lower())


@functools.lru_cache(maxsize=4096)
def _extract_entities_cached(
    text: str,
    author: str | None,
    max_entities: int,
) -> tuple[NamedEntity, ...]:
    counts: Counter[tuple[str, EntityLabel]] = Counter()
    seen_spans: set[str] = set()
    trusted_persons: set[str] = set()

    for person_name in _extract_author_people(author):
        _record_entity(counts, seen_spans, person_name, EntityLabel.PERSON)
        trusted_persons.add(person_name.lower())

    clean_text = _clean_body_for_ner(text)
    if not clean_text.strip():
        return tuple(
            NamedEntity(text=name, label=label, count=cnt)
            for (name, label), cnt in counts.most_common(max_entities)
        )

    lines = clean_text.splitlines()
    occupied_by_line: list[list[tuple[int, int]]] = [[] for _ in lines]

    for line_idx, line in enumerate(lines):
        occupied = occupied_by_line[line_idx]
        lower_line = line.lower()

        # 1. Explicit named events (multi-word & acronyms)
        for regex in (_EVENT_PHRASE_SL_RE, _EXTRA_EVENT_PHRASE_RE, _EVENT_ACRONYM_RE):
            for match in regex.finditer(line):
                _record_entity(counts, seen_spans, match.group(0), EntityLabel.EVENT)
                occupied.append((match.start(), match.end()))

        # 2. Organizations by suffix, multi-word curated names, and strict ALL-CAPS acronyms
        for regex in (_ORG_PHRASE_SL_RE, _EXTRA_ORG_PHRASE_RE):
            for match in regex.finditer(line):
                if not _overlaps_occupied(occupied, match.start(), match.end()):
                    _record_entity(counts, seen_spans, match.group(0), EntityLabel.ORG)
                    occupied.append((match.start(), match.end()))

        for match in _MULTIWORD_ORGS_RE.finditer(lower_line):
            canonical = _MULTIWORD_ORGS.get(match.group(0).lower())
            if canonical:
                _record_entity(counts, seen_spans, canonical, EntityLabel.ORG)
                occupied.append((match.start(), match.end()))

        for match in _PROPER_ORGS_RE.finditer(line):
            _record_entity(counts, seen_spans, match.group(0), EntityLabel.ORG)
            occupied.append((match.start(), match.end()))

        for match in _ACRONYM_ORGS_RE.finditer(line):
            _record_entity(counts, seen_spans, match.group(0), EntityLabel.ORG)
            occupied.append((match.start(), match.end()))

        # 3. Geo-political entities (GPE) and geographic features
        for match in _GEO_FEATURE_RE.finditer(line):
            if not _overlaps_occupied(occupied, match.start(), match.end()):
                _record_entity(counts, seen_spans, match.group(0), EntityLabel.GPE)
                occupied.append((match.start(), match.end()))

        for match in _GPE_SHORT_RE.finditer(line):
            _record_entity(counts, seen_spans, match.group(0).upper(), EntityLabel.GPE)
            occupied.append((match.start(), match.end()))

        for match in _GPE_LONG_RE.finditer(lower_line):
            if not _overlaps_occupied(occupied, match.start(), match.end()):
                _record_entity(counts, seen_spans, match.group(0).title(), EntityLabel.GPE)
                occupied.append((match.start(), match.end()))

        # 4a. Title-anchored person mentions
        for match in _PERSON_TITLE_SL_RE.finditer(line):
            if _overlaps_occupied(occupied, match.start(1), match.end(1)):
                continue
            cleaned_person = _clean_person_words(match.group(1))
            if not cleaned_person:
                continue
            lower_cand = cleaned_person.lower()
            trusted_persons.add(lower_cand)
            if lower_cand not in seen_spans:
                _record_entity(counts, seen_spans, cleaned_person, EntityLabel.PERSON)
                occupied.append((match.start(1), match.end(1)))

    # 4b. Context-verified capitalized person names across clean body lines
    for line_idx, line in enumerate(lines):
        occupied = occupied_by_line[line_idx]
        for match in _CAP_PERSON_SL_RE.finditer(line):
            if _overlaps_occupied(occupied, match.start(1), match.end(1)):
                continue
            cleaned_person = _clean_person_words(match.group(1))
            if not cleaned_person:
                continue
            lower_cand = cleaned_person.lower()
            if (
                lower_cand in seen_spans
                or lower_cand in _GPE_GAZETTEER
                or lower_cand in _MULTIWORD_ORGS
            ):
                continue
            if lower_cand not in trusted_persons and not _has_person_context(
                line, match.start(1), match.end(1)
            ):
                continue
            _record_entity(counts, seen_spans, cleaned_person, EntityLabel.PERSON)
            occupied.append((match.start(1), match.end(1)))

    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0][0].lower(), item[0][1].value),
    )[:max_entities]
    return tuple(
        NamedEntity(text=name, label=label, count=cnt)
        for (name, label), cnt in ranked
    )


def extract_entities(
    text: str,
    *,
    author: str | None = None,
    max_entities: int = 15,
) -> list[NamedEntity]:
    """Extract PERSON, ORG, GPE, and EVENT entities from clean article body text."""
    return list(_extract_entities_cached(text or "", author, max_entities))


def aggregate_entities(
    entity_lists: list[list[NamedEntity]],
    *,
    max_entities: int = 15,
) -> list[NamedEntity]:
    """Aggregate entity frequencies across multiple retrieved documents."""
    totals: Counter[tuple[str, EntityLabel]] = Counter()
    for entities in entity_lists:
        for ent in entities:
            totals[(ent.text, ent.label)] += ent.count
    ranked = sorted(
        totals.items(),
        key=lambda item: (-item[1], item[0][0].lower(), item[0][1].value),
    )[:max_entities]
    return [
        NamedEntity(text=name, label=label, count=cnt)
        for (name, label), cnt in ranked
    ]


def suggest_background_links(
    index: BM25Index,
    anchor_doc: IndexedDocument,
    *,
    limit: int = 3,
    entities_by_doc: dict[str, list[NamedEntity]] | None = None,
    token_sets_by_doc: dict[str, set[str]] | None = None,
) -> list[BackgroundLink]:
    """Suggest related background stories grounded in shared clean-body entities and topical overlap."""
    if limit <= 0 or index.doc_count <= 1:
        return []

    anchor_entities = (
        entities_by_doc[anchor_doc.doc_id][:12]
        if entities_by_doc is not None and anchor_doc.doc_id in entities_by_doc
        else extract_entities(anchor_doc.body_text, author=None, max_entities=12)
    )
    anchor_entity_set = {e.text.lower() for e in anchor_entities}
    anchor_specific_entities = {
        e.text.lower() for e in anchor_entities if e.label != EntityLabel.GPE
    }

    body_tokens = preprocess_text(anchor_doc.body_text)[:25]
    entity_terms = [e.text for e in anchor_entities[:8]]
    query_parts = [anchor_doc.title, anchor_doc.title]
    if anchor_doc.kicker:
        query_parts.append(anchor_doc.kicker)
    query_parts.extend(entity_terms)
    query_parts.extend(body_tokens)
    query = " ".join(query_parts)

    candidates = index.search(
        query,
        top_k=min(index.doc_count, max(limit * 4, 12)),
        min_should_match=0.15,
    )
    if not candidates:
        return []

    anchor_token_set = (
        token_sets_by_doc[anchor_doc.doc_id]
        if token_sets_by_doc is not None and anchor_doc.doc_id in token_sets_by_doc
        else set(preprocess_text(f"{anchor_doc.title} {anchor_doc.body_text[:1200]}"))
    )
    max_bm25 = max((hit.score for hit in candidates if hit.doc_id != anchor_doc.doc_id), default=1.0)
    if max_bm25 <= 0.0:
        max_bm25 = 1.0

    scored: list[tuple[float, list[str], object]] = []
    for hit in candidates:
        if hit.doc_id == anchor_doc.doc_id or hit.canonical_url == anchor_doc.canonical_url:
            continue

        cand_token_set = (
            token_sets_by_doc[hit.doc_id]
            if token_sets_by_doc is not None and hit.doc_id in token_sets_by_doc
            else set(preprocess_text(f"{hit.title} {hit.body_text[:1200]}"))
        )
        token_union = len(anchor_token_set | cand_token_set)
        token_jaccard = (
            len(anchor_token_set & cand_token_set) / token_union if token_union > 0 else 0.0
        )

        cand_entities = (
            entities_by_doc[hit.doc_id][:12]
            if entities_by_doc is not None and hit.doc_id in entities_by_doc
            else extract_entities(hit.body_text, author=None, max_entities=12)
        )
        cand_entity_map = {e.text.lower(): e.text for e in cand_entities}
        cand_specific = {
            e.text.lower() for e in cand_entities if e.label != EntityLabel.GPE
        }
        shared_keys = anchor_entity_set & set(cand_entity_map.keys())
        shared_specific = anchor_specific_entities & cand_specific
        shared_names = sorted(cand_entity_map[k] for k in shared_keys)

        # Reject coincidental single-country or low-overlap pairs that share no topic
        if not shared_specific and token_jaccard < 0.08:
            continue
        if token_jaccard < 0.04 and len(shared_keys) < 2:
            continue

        bm25_norm = hit.score / max_bm25
        overlap_gate = min(1.0, token_jaccard / 0.22) if token_jaccard > 0 else 0.0
        entity_union = len(anchor_entity_set | set(cand_entity_map.keys()))
        entity_jaccard = (len(shared_keys) / entity_union) if entity_union > 0 else 0.0
        kicker_bonus = (
            0.08
            if (anchor_doc.kicker and hit.kicker and anchor_doc.kicker.lower() == hit.kicker.lower())
            else 0.0
        )
        earlier_bonus = (
            0.04
            if (
                anchor_doc.published_date
                and hit.published_date
                and hit.published_date <= anchor_doc.published_date
            )
            else 0.0
        )
        combined_score = round(
            min(
                1.0,
                0.50 * (bm25_norm * overlap_gate)
                + 0.25 * min(1.0, token_jaccard / 0.30)
                + 0.20 * entity_jaccard
                + kicker_bonus
                + earlier_bonus,
            ),
            6,
        )
        scored.append((combined_score, shared_names, hit))

    scored.sort(key=lambda entry: (-entry[0], -len(entry[1]), entry[2].doc_id))  # type: ignore[attr-defined]
    links: list[BackgroundLink] = []
    for combined_score, shared_names, hit in scored[:limit]:
        links.append(
            BackgroundLink(
                doc_id=hit.doc_id,  # type: ignore[attr-defined]
                title=hit.title,  # type: ignore[attr-defined]
                source=hit.source,  # type: ignore[attr-defined]
                kicker=hit.kicker,  # type: ignore[attr-defined]
                published_date=hit.published_date,  # type: ignore[attr-defined]
                canonical_url=hit.canonical_url,  # type: ignore[attr-defined]
                similarity_score=combined_score,
                shared_entities=shared_names,
            )
        )
    return links
