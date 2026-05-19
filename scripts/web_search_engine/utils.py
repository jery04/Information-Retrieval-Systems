"""Utility functions for web search: URL normalization and language detection."""

import re
from typing import Optional
from urllib.parse import urlparse, urlunparse


def canonicalize_url(raw_url: str) -> Optional[str]:
    """Normalize URL by scheme/netloc/path and strip query/fragment."""
    if not raw_url:
        return None

    parsed = urlparse(raw_url.strip())
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None

    clean_path = re.sub(r"/+", "/", parsed.path or "/")
    if clean_path != "/" and clean_path.endswith("/"):
        clean_path = clean_path[:-1]

    canonical = parsed._replace(path=clean_path, query="", fragment="", params="")
    return urlunparse(canonical)


def normalize_language_code(raw_language: Optional[str]) -> Optional[str]:
    """Normalize language code to simple lowercase tag."""
    if not raw_language:
        return None

    value = raw_language.strip().lower().replace("_", "-")
    if not value:
        return None

    match = re.match(r"([a-z]{2,3})", value)
    return match.group(1) if match else None


def infer_language_from_url(url: str) -> Optional[str]:
    """Infer language from host/path patterns like es.site.com or /en/ paths."""
    try:
        parsed = urlparse(url)
    except Exception:
        return None

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()

    host_match = re.match(r"^([a-z]{2,3})(?:-[a-z]{2,3})?\.", host)
    if host_match:
        return host_match.group(1)

    path_match = re.match(r"^/([a-z]{2,3})(?:-[a-z]{2,3})?(/|$)", path)
    if path_match:
        return path_match.group(1)

    return None


LANGUAGE_STOPWORDS = {
    "en": {
        "the", "and", "or", "of", "to", "in", "for", "on", "with", "as", "by", "from",
        "that", "this", "it", "is", "are", "be", "was", "were", "at", "an", "a", "not",
        "you", "your", "we", "our", "can", "will",
    },
    "es": {
        "el", "la", "los", "las", "de", "del", "y", "o", "en", "para", "con", "por",
        "que", "es", "son", "una", "un", "como", "se", "al", "más", "sus", "sobre", "esta",
        "esto", "puede", "pueden", "nuestro", "nuestra",
    },
}


def infer_language_from_text(text: str) -> Optional[str]:
    """Infer ES/EN language from stopword overlap. Returns None if uncertain."""
    snippet = (text or "")[:5000].lower()
    if not snippet:
        return None

    tokens = re.findall(r"[a-záéíóúñü]+", snippet)
    if len(tokens) < 40:
        return None

    token_set = set(tokens)
    scores = {
        lang: sum(1 for word in stopwords if word in token_set)
        for lang, stopwords in LANGUAGE_STOPWORDS.items()
    }

    best_lang, best_score = max(scores.items(), key=lambda item: item[1])
    other_score = max((score for lang, score in scores.items() if lang != best_lang), default=0)

    if best_score < 4:
        return None
    if best_score <= other_score:
        return None
    return best_lang


def detect_language_metadata(
    title: str,
    text: str,
    url: str,
    html_hint: Optional[str] = None,
    header_hint: Optional[str] = None,
) -> str:
    """Multi-step language detection: meta tags → headers → URL → text heuristic."""
    for hint in (html_hint, header_hint, infer_language_from_url(url), infer_language_from_text(f"{title} {text}")):
        normalized = normalize_language_code(hint)
        if normalized:
            return normalized
    return "unknown"
