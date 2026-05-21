"""Modular components for the focused technology crawler."""

from .extractor import ExtractionResult, SimpleHTMLExtractor, extract_page_content
from .models import CrawlConfig
from .relevance import TECH_KEYWORDS, score_relevance
from .utils import (
	BINARY_EXTENSIONS,
	canonicalize_url,
	infer_language_from_text,
	infer_language_from_url,
	load_seeds,
	normalize_language_code,
	utc_now_iso,
)

__all__ = [
	"CrawlConfig",
	"ExtractionResult",
	"SimpleHTMLExtractor",
	"extract_page_content",
	"TECH_KEYWORDS",
	"score_relevance",
	"BINARY_EXTENSIONS",
	"canonicalize_url",
	"infer_language_from_text",
	"infer_language_from_url",
	"load_seeds",
	"normalize_language_code",
	"utc_now_iso",
]
