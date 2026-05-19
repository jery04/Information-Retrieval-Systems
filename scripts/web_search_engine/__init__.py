"""Web search module: DuckDuckGo-based fallback for insufficient local results."""

from .config import WebSearchConfig
from .ddg_scraper import DDGScraper
from .extractor import SimpleHTMLExtractor, extract_page_content
from .utils import (
    canonicalize_url,
    detect_language_metadata,
    infer_language_from_text,
    infer_language_from_url,
    normalize_language_code,
)

__all__ = [
    "WebSearchConfig",
    "DDGScraper",
    "SimpleHTMLExtractor",
    "extract_page_content",
    "canonicalize_url",
    "detect_language_metadata",
    "infer_language_from_text",
    "infer_language_from_url",
    "normalize_language_code",
]
