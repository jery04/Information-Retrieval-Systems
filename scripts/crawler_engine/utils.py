from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, urlunparse


LANGUAGE_STOPWORDS: Dict[str, Set[str]] = {
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

BINARY_EXTENSIONS = (
    ".pdf", ".zip", ".rar", ".7z", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".webp", ".mp4", ".webm", ".mp3", ".wav", ".avi", ".mov", ".ppt", ".pptx", ".doc", ".docx",
    ".xls", ".xlsx", ".exe", ".dmg", ".iso",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonicalize_url(raw_url: str) -> Optional[str]:
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
    if not raw_language:
        return None

    value = raw_language.strip().lower().replace("_", "-")
    if not value:
        return None

    match = re.match(r"([a-z]{2,3})", value)
    return match.group(1) if match else None


def infer_language_from_url(url: str) -> Optional[str]:
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


def infer_language_from_text(text: str) -> Optional[str]:
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

    if best_score < 4 or best_score <= other_score:
        return None
    return best_lang


def load_seeds(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Seeds file not found: {path}")

    seeds: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            seeds.append(line)

    if not seeds:
        raise ValueError(f"No seeds found in: {path}")

    return seeds
