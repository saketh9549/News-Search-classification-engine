"""Shared URL normalization and domain extraction utilities."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAM_PREFIXES: tuple[str, ...] = ("utm_", "at_")
TRACKING_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "ref",
        "ref_src",
        "fbclid",
        "gclid",
        "yclid",
        "mc_cid",
        "mc_eid",
        "cmpid",
        "ocid",
        "ncid",
        "ito",
        "taid",
        "guccounter",
        "guce_referrer",
        "guce_referrer_sig",
        "traffic_source",
    }
)


def _is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    return lowered in TRACKING_PARAM_NAMES or any(
        lowered.startswith(prefix) for prefix in TRACKING_PARAM_PREFIXES
    )


def normalize_url(url: str) -> str:
    """Normalize an HTTP(S) URL by lowercasing host, stripping fragments and tracking params."""
    cleaned = url.strip()
    parts = urlsplit(cleaned)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"Unsupported or invalid HTTP(S) URL: {url!r}")

    hostname = parts.hostname.lower()
    port = parts.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
    netloc = f"{hostname}:{port}" if port else hostname

    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(key)
    ]
    query = urlencode(filtered_query, doseq=True)
    path = parts.path or "/"
    return urlunsplit((scheme, netloc, path, query, ""))


def extract_domain(url: str) -> str:
    """Extract lowercase network hostname from a URL."""
    parts = urlsplit(url.strip())
    if not parts.hostname:
        raise ValueError(f"URL has no hostname: {url!r}")
    return parts.hostname.lower()
