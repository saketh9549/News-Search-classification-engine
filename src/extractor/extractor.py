"""HTML boilerplate removal, metadata extraction, validation, and permanent persistence."""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import logging
import re
import sqlite3
from datetime import datetime
from urllib.parse import urljoin

from lxml import etree, html as lxml_html

from src.crawler.schemas import CrawlResult
from src.extractor.schemas import (
    DEFAULT_MAX_LINK_TO_TEXT_RATIO,
    DEFAULT_MIN_WORD_COUNT,
    ExtractionResult,
)
from src.models.schemas import Article, normalize_category, utc_now
from src.models.urls import normalize_url
from src.poller.parser import is_article_url, parse_feed_timestamp
from src.storage.repository import get_feed_by_id, save_article

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;!?)\]]|\.(?!\d))")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[])\s+")
_DUP_COMMA_RE = re.compile(r",\s*,+")
_COMMA_PERIOD_RE = re.compile(r",\s*\.(?!\d)")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", re.UNICODE)
_BOILERPLATE_PARAGRAPH_RE = re.compile(
    r"^(?:"
    r"sign\s+up\b|subscribe\b|follow\s+us\b|read\s+more\b|also\s+read\b|"
    r"watch\s*:|listen\s*:|image\s+source\b|image\s+caption\b|"
    r"getty\s+images\b|related\s+topics?\b|more\s+on\s+this\s+story\b|"
    r"recommended\s+stories\b|download\s+(?:the\s+)?(?:toi|times|hindu|ht|app)\b|"
    r"published\s+by\b|published\s*[-–—:]|updated\s*[-–—:]|"
    r"copyright\s+©|all\s+rights\s+reserved\b|"
    r"click\s+here\b|download\s+the\b|join\s+our\b"
    r")",
    re.IGNORECASE,
)
_INLINE_CREDIT_RE = re.compile(
    r"\s*(?:\((?:getty\s+images|reuters|ap\s+photo|afp|pti|file\s+photo)\)|"
    r"image\s+source,\s*getty\s+images|"
    r"\|\s*(?:photo\s+credit|file\s+photo|image\s+credit)\s*:[^\n]*)\s*",
    re.IGNORECASE,
)
_NON_ARTICLE_PAGE_TITLE_RE = re.compile(r"(?:\s-\s*bbc\s+sounds$)", re.IGNORECASE)
_BOILERPLATE_ATTR_RE = re.compile(
    r"(?:^|[-_\s])("
    r"ad|ads|advert|advertisement|promo|sponsored|sidebar|social|share|"
    r"newsletter|subscribe|subscription|cookie|consent|popup|modal|"
    r"related|recommended|comment|comments|breadcrumb|menu|navbar|footer|"
    r"most-read|trending"
    r")(?:$|[-_\s])",
    re.IGNORECASE,
)
_NOISE_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "canvas",
        "form",
        "button",
        "input",
        "select",
        "textarea",
        "nav",
        "footer",
        "aside",
    }
)
_NOISE_ROLES = frozenset({"navigation", "banner", "contentinfo", "complementary", "dialog"})
_ARTICLE_LD_TYPES = frozenset(
    {"NewsArticle", "Article", "ReportageNewsArticle", "BlogPosting", "AnalysisNewsArticle"}
)
_ISO_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
)
_BY_PREFIX_RE = re.compile(r"^by\s+", re.IGNORECASE)


def _clean_text(raw: str | None) -> str:
    if not raw:
        return ""
    unescaped = html_lib.unescape(raw)
    cleaned = _WHITESPACE_RE.sub(" ", unescaped).strip()
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _SPACE_AFTER_OPEN_RE.sub(r"\1", cleaned)
    cleaned = _DUP_COMMA_RE.sub(",", cleaned)
    cleaned = _COMMA_PERIOD_RE.sub(".", cleaned)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _extract_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _iter_json_ld_objects(doc: lxml_html.HtmlElement) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for script in doc.xpath("//script[@type='application/ld+json']"):
        raw = (script.text or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        stack: list[object] = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, dict):
                objects.append(item)
                graph = item.get("@graph")
                if isinstance(graph, list):
                    stack.extend(graph)
    return objects


def _find_article_ld(objects: list[dict[str, object]]) -> dict[str, object]:
    for obj in objects:
        obj_type = obj.get("@type")
        if isinstance(obj_type, str) and obj_type in _ARTICLE_LD_TYPES:
            return obj
        if isinstance(obj_type, list) and any(
            isinstance(t, str) and t in _ARTICLE_LD_TYPES for t in obj_type
        ):
            return obj
    for obj in objects:
        if isinstance(obj.get("articleBody"), str) and obj["articleBody"].strip():
            return obj
    return objects[0] if objects else {}


def _meta_content(doc: lxml_html.HtmlElement, *selectors: tuple[str, str]) -> str | None:
    for attr, value in selectors:
        nodes = doc.xpath(f"//meta[translate(@{attr}, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='{value.lower()}']")
        for node in nodes:
            content = _clean_text(node.get("content"))
            if content:
                return content
    return None


def _extract_author(doc: lxml_html.HtmlElement, ld: dict[str, object]) -> str | None:
    ld_author = ld.get("author")
    if isinstance(ld_author, str) and _clean_text(ld_author):
        return _clean_text(ld_author)
    if isinstance(ld_author, dict):
        name = _clean_text(str(ld_author.get("name", "")))
        if name:
            return name
    if isinstance(ld_author, list):
        names: list[str] = []
        for entry in ld_author:
            if isinstance(entry, dict) and entry.get("name"):
                names.append(_clean_text(str(entry["name"])))
            elif isinstance(entry, str) and _clean_text(entry):
                names.append(_clean_text(entry))
        if names:
            return ", ".join(names)

    meta_author = _meta_content(
        doc,
        ("name", "author"),
        ("name", "byl"),
        ("property", "article:author"),
    )
    if meta_author and not meta_author.lower().startswith(("http://", "https://")):
        return _BY_PREFIX_RE.sub("", meta_author).strip() or None

    for xpath in (
        "//*[@rel='author']",
        "//*[contains(translate(@class, 'BYLINEAUTHOR', 'bylineauthor'), 'byline')]",
        "//*[contains(translate(@class, 'BYLINEAUTHOR', 'bylineauthor'), 'author')]",
    ):
        for node in doc.xpath(xpath):
            text = _clean_text(node.text_content())
            if text and len(text) <= 120:
                cleaned = _BY_PREFIX_RE.sub("", text).strip()
                if cleaned:
                    return cleaned
    return None


def _extract_kicker(
    doc: lxml_html.HtmlElement,
    ld: dict[str, object],
    default_kicker: str | None,
) -> str | None:
    section = _meta_content(
        doc,
        ("property", "article:section"),
        ("name", "section"),
        ("name", "category"),
    )
    if section:
        normalized = normalize_category(section)
        if normalized:
            return normalized

    ld_section = ld.get("articleSection")
    if isinstance(ld_section, str) and _clean_text(ld_section):
        normalized = normalize_category(_clean_text(ld_section))
        if normalized:
            return normalized
    if isinstance(ld_section, list) and ld_section:
        first = _clean_text(str(ld_section[0]))
        if first:
            normalized = normalize_category(first)
            if normalized:
                return normalized

    for node in doc.xpath(
        "//*[contains(translate(@class, 'KICKER', 'kicker'), 'kicker')]"
    ):
        text = _clean_text(node.text_content())
        if text and len(text) <= 60:
            normalized = normalize_category(text)
            if normalized:
                return normalized

    return normalize_category(_clean_text(default_kicker))


def _extract_published_date(
    doc: lxml_html.HtmlElement,
    ld: dict[str, object],
) -> datetime | None:
    candidates: list[str] = []

    for key in ("datePublished", "dateCreated"):
        val = ld.get(key)
        if isinstance(val, str) and val.strip():
            candidates.append(val)

    meta_date = _meta_content(
        doc,
        ("property", "article:published_time"),
        ("property", "og:published_time"),
        ("name", "pubdate"),
        ("name", "publishdate"),
        ("name", "publish-date"),
        ("name", "dc.date.issued"),
        ("name", "parsely-pub-date"),
        ("name", "sailthru.date"),
    )
    if meta_date:
        candidates.append(meta_date)

    for time_node in doc.xpath("//time[@datetime]"):
        dt_attr = time_node.get("datetime")
        if dt_attr:
            candidates.append(dt_attr)

    for candidate in candidates:
        parsed = parse_feed_timestamp(candidate)
        if parsed is not None:
            return parsed

    for time_node in doc.xpath("//time"):
        text = _clean_text(time_node.text_content())
        match = _ISO_DATE_RE.search(text)
        if match:
            parsed = parse_feed_timestamp(match.group(0))
            if parsed is not None:
                return parsed

    return None


def _extract_canonical_url(
    doc: lxml_html.HtmlElement,
    ld: dict[str, object],
    fallback_url: str,
) -> str:
    candidates: list[str] = []
    for link in doc.xpath("//link[translate(@rel, 'CANONICAL', 'canonical')='canonical']"):
        href = link.get("href")
        if href:
            candidates.append(href)

    og_url = _meta_content(doc, ("property", "og:url"))
    if og_url:
        candidates.append(og_url)

    main_entity = ld.get("mainEntityOfPage")
    if isinstance(main_entity, dict) and isinstance(main_entity.get("@id"), str):
        candidates.append(main_entity["@id"])
    elif isinstance(main_entity, str):
        candidates.append(main_entity)
    elif isinstance(ld.get("url"), str):
        candidates.append(str(ld["url"]))

    for candidate in candidates:
        resolved = urljoin(fallback_url, candidate.strip())
        try:
            return normalize_url(resolved)
        except ValueError:
            continue

    try:
        return normalize_url(fallback_url)
    except ValueError:
        return fallback_url


def _extract_title(doc: lxml_html.HtmlElement, ld: dict[str, object]) -> str:
    for candidate in (
        _meta_content(doc, ("property", "og:title"), ("name", "twitter:title")),
        _clean_text(str(ld["headline"])) if isinstance(ld.get("headline"), str) else None,
    ):
        if candidate:
            return candidate

    for h1 in doc.xpath("//h1"):
        text = _clean_text(h1.text_content())
        if text:
            return text

    for title_node in doc.xpath("//title"):
        raw_title = _clean_text(title_node.text_content())
        if raw_title:
            return re.split(r"\s+[|\-–—]\s+", raw_title, maxsplit=1)[0].strip()

    return ""


def _strip_boilerplate(root: lxml_html.HtmlElement) -> None:
    for comment in root.xpath("//comment()"):
        parent = comment.getparent()
        if parent is not None:
            parent.remove(comment)

    to_remove: list[lxml_html.HtmlElement] = []
    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        tag = elem.tag.lower()
        if tag in _NOISE_TAGS:
            to_remove.append(elem)
            continue
        if tag == "header" and not elem.xpath("ancestor::article"):
            to_remove.append(elem)
            continue
        role = (elem.get("role") or "").strip().lower()
        if role in _NOISE_ROLES:
            to_remove.append(elem)
            continue
        attr_blob = f"{elem.get('id', '')} {elem.get('class', '')}".strip()
        if attr_blob and _BOILERPLATE_ATTR_RE.search(attr_blob):
            to_remove.append(elem)

    for elem in to_remove:
        parent = elem.getparent()
        if parent is not None:
            parent.remove(elem)


def _container_score(container: lxml_html.HtmlElement) -> tuple[int, float]:
    paragraphs = [
        _clean_text(p.text_content())
        for p in container.xpath(".//p")
        if _clean_text(p.text_content())
    ]
    p_words = sum(len(_extract_words(p)) for p in paragraphs)
    total_words = len(_extract_words(_clean_text(container.text_content())))
    link_words = sum(
        len(_extract_words(_clean_text(a.text_content())))
        for a in container.xpath(".//a")
    )
    ratio = (link_words / total_words) if total_words > 0 else 1.0
    effective_words = p_words if p_words > 0 else total_words
    return effective_words, ratio


def _weighted_container_score(container: lxml_html.HtmlElement) -> float:
    words, ratio = _container_score(container)
    return words * (1.0 - min(0.9, ratio))


def _select_primary_container(body: lxml_html.HtmlElement) -> lxml_html.HtmlElement:
    for selector in (".//article", ".//main", ".//*[@itemprop='articleBody']"):
        candidates = body.xpath(selector)
        if candidates:
            return max(candidates, key=lambda c: _container_score(c)[0])

    block_candidates = body.xpath(".//div | .//section")
    if block_candidates:
        best = max(block_candidates, key=_weighted_container_score)
        if _container_score(best)[0] > 0:
            return best

    return body


def _clean_paragraph_blocks(blocks: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_block in blocks:
        text = _INLINE_CREDIT_RE.sub(" ", raw_block)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        if not text:
            continue
        if _BOILERPLATE_PARAGRAPH_RE.match(text) and len(_extract_words(text)) <= 35:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def _extract_body_and_density(
    doc: lxml_html.HtmlElement,
    ld: dict[str, object],
    *,
    min_word_count: int = DEFAULT_MIN_WORD_COUNT,
    max_link_to_text_ratio: float = DEFAULT_MAX_LINK_TO_TEXT_RATIO,
) -> tuple[str, float]:
    bodies = doc.xpath("//body")
    body = bodies[0] if bodies else doc
    _strip_boilerplate(body)

    container = _select_primary_container(body)
    total_text = _clean_text(container.text_content())
    total_words = len(_extract_words(total_text))
    link_words = sum(
        len(_extract_words(_clean_text(a.text_content())))
        for a in container.xpath(".//a")
    )
    link_to_text_ratio = min(1.0, link_words / total_words) if total_words > 0 else 1.0

    raw_blocks: list[str] = []
    for node in container.xpath(".//p | .//h2 | .//h3 | .//blockquote[not(.//p)]"):
        text = _clean_text(node.text_content())
        if text:
            raw_blocks.append(text)

    blocks = _clean_paragraph_blocks(raw_blocks)
    if blocks:
        body_text = "\n\n".join(blocks)
    elif total_text:
        body_text = total_text
    else:
        body_text = ""

    ld_body = ld.get("articleBody")
    if isinstance(ld_body, str):
        cleaned_ld_paragraphs = _clean_paragraph_blocks(
            [_clean_text(p) for p in re.split(r"\n+", ld_body) if _clean_text(p)]
        )
        cleaned_ld = "\n\n".join(cleaned_ld_paragraphs) if cleaned_ld_paragraphs else _clean_text(ld_body)
        if cleaned_ld:
            dom_words = len(_extract_words(body_text))
            ld_words = len(_extract_words(cleaned_ld))
            if not body_text:
                body_text = cleaned_ld
                link_to_text_ratio = 0.0
            elif (
                dom_words < min_word_count
                and link_to_text_ratio <= max_link_to_text_ratio
                and ld_words > dom_words
            ):
                body_text = cleaned_ld

    return body_text, link_to_text_ratio


def extract_article(
    crawl_result: CrawlResult,
    *,
    default_kicker: str | None = None,
    min_word_count: int = DEFAULT_MIN_WORD_COUNT,
    max_link_to_text_ratio: float = DEFAULT_MAX_LINK_TO_TEXT_RATIO,
    fetched_at: datetime | None = None,
) -> ExtractionResult:
    """Extract, clean, and validate an Article from a CrawlResult."""
    if not crawl_result.html or not crawl_result.html.strip():
        return ExtractionResult(
            accepted=False,
            rejection_reason="Missing raw HTML payload",
        )

    try:
        doc = lxml_html.fromstring(crawl_result.html)
    except (etree.ParserError, ValueError) as exc:
        return ExtractionResult(
            accepted=False,
            rejection_reason=f"Unable to parse HTML: {exc}",
        )

    ld_objects = _iter_json_ld_objects(doc)
    ld = _find_article_ld(ld_objects)

    title = _extract_title(doc, ld)
    if not title:
        return ExtractionResult(
            accepted=False,
            rejection_reason="Missing article headline",
        )
    raw_doc_title = " ".join(
        _clean_text(node.text_content()) for node in doc.xpath("//title")
    )
    canonical_url = _extract_canonical_url(doc, ld, crawl_result.final_url)
    if (
        _NON_ARTICLE_PAGE_TITLE_RE.search(title)
        or _NON_ARTICLE_PAGE_TITLE_RE.search(raw_doc_title)
        or not is_article_url(canonical_url)
    ):
        return ExtractionResult(
            accepted=False,
            rejection_reason=f"Non-article audio/podcast or multimedia page ({canonical_url})",
        )

    published_date = _extract_published_date(doc, ld)
    author = _extract_author(doc, ld)
    kicker = _extract_kicker(doc, ld, default_kicker)

    body_text, link_to_text_ratio = _extract_body_and_density(
        doc,
        ld,
        min_word_count=min_word_count,
        max_link_to_text_ratio=max_link_to_text_ratio,
    )
    if not body_text:
        return ExtractionResult(
            accepted=False,
            rejection_reason="Missing article body text",
            link_to_text_ratio=link_to_text_ratio,
        )

    word_count = len(_extract_words(body_text))
    if link_to_text_ratio > max_link_to_text_ratio:
        return ExtractionResult(
            accepted=False,
            rejection_reason=(
                f"High link-to-text ratio ({link_to_text_ratio:.2f} > {max_link_to_text_ratio:.2f})"
            ),
            word_count=word_count,
            link_to_text_ratio=link_to_text_ratio,
        )

    if word_count < min_word_count:
        return ExtractionResult(
            accepted=False,
            rejection_reason=f"Article body too short ({word_count} < {min_word_count} words)",
            word_count=word_count,
            link_to_text_ratio=link_to_text_ratio,
        )

    doc_id = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
    content_hash = hashlib.sha256(body_text.encode("utf-8")).hexdigest()

    article = Article(
        doc_id=doc_id,
        feed_id=crawl_result.feed_id,
        url=crawl_result.url,
        canonical_url=canonical_url,
        title=title,
        published_date=published_date,
        author=author,
        kicker=kicker,
        body_text=body_text,
        content_hash=content_hash,
        fetched_at=fetched_at or crawl_result.fetched_at or utc_now(),
    )
    return ExtractionResult(
        accepted=True,
        article=article,
        word_count=word_count,
        link_to_text_ratio=link_to_text_ratio,
    )


def extract_and_store(
    conn: sqlite3.Connection,
    crawl_result: CrawlResult,
    *,
    min_word_count: int = DEFAULT_MIN_WORD_COUNT,
    max_link_to_text_ratio: float = DEFAULT_MAX_LINK_TO_TEXT_RATIO,
) -> ExtractionResult:
    """Extract an article from CrawlResult and permanently store valid records in SQLite."""
    feed = get_feed_by_id(conn, crawl_result.feed_id)
    default_kicker = feed.category if feed is not None else None

    result = extract_article(
        crawl_result,
        default_kicker=default_kicker,
        min_word_count=min_word_count,
        max_link_to_text_ratio=max_link_to_text_ratio,
    )
    if not result.accepted or result.article is None:
        logger.info("Rejected extracted page %s: %s", crawl_result.url, result.rejection_reason)
        return result

    persisted = save_article(conn, result.article)
    return result.model_copy(update={"persisted": persisted})
