"""HTML extractor adapter for web search.

This module reuses the shared crawler extractor and keeps a tuple-based
interface for the web search pipeline.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from crawler_engine.extractor import ExtractionResult, SimpleHTMLExtractor, extract_page_content as _shared_extract_page_content


def extract_page_content(html: str) -> Tuple[str, str, List[str], Optional[str]]:
    """Parse HTML and return the legacy tuple expected by web search code."""
    result = _shared_extract_page_content(html)
    if isinstance(result, ExtractionResult):
        return result.title, result.text, result.links, result.html_language

    # Defensive fallback for unexpected shapes.
    title = getattr(result, "title", "")
    text = getattr(result, "text", "")
    links = getattr(result, "links", []) or []
    html_language = getattr(result, "html_language", None)
    return title, text, links, html_language


__all__ = ["SimpleHTMLExtractor", "extract_page_content"]
