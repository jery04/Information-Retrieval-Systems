"""HTML extractor for web search results: title, text, links, language hint."""

import re
from html import unescape
from html.parser import HTMLParser
from typing import List, Optional, Tuple

from .utils import normalize_language_code


class SimpleHTMLExtractor(HTMLParser):
    """Extract title, visible text, links and language hint from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: List[str] = []
        self._text_parts: List[str] = []
        self.links: List[str] = []
        self.language_hint: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        attr_map = {
            name.lower(): value.strip()
            for name, value in attrs
            if isinstance(name, str) and isinstance(value, str) and value.strip()
        }

        if tag == "html" and self.language_hint is None:
            self.language_hint = normalize_language_code(attr_map.get("lang"))

        if tag == "meta" and self.language_hint is None:
            name = attr_map.get("name", "").lower()
            prop = attr_map.get("property", "").lower()
            http_equiv = attr_map.get("http-equiv", "").lower()
            content = attr_map.get("content")

            if name == "language" and content:
                self.language_hint = normalize_language_code(content)
            elif http_equiv == "content-language" and content:
                self.language_hint = normalize_language_code(content)
            elif prop == "og:locale" and content:
                self.language_hint = normalize_language_code(content)

        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return

        if tag == "title":
            self._in_title = True

        if tag == "a":
            href = attr_map.get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return

        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return

        text = data.strip()
        if not text:
            return

        if self._in_title:
            self._title_parts.append(text)
        self._text_parts.append(text)

    def extract(self) -> Tuple[str, str, List[str], Optional[str]]:
        """Return (title, text, links, language_hint)."""
        title = " ".join(self._title_parts).strip()
        text = unescape(re.sub(r"\s+", " ", " ".join(self._text_parts))).strip()
        return title, text, self.links, self.language_hint


def extract_page_content(html: str) -> Tuple[str, str, List[str], Optional[str]]:
    """Parse HTML and extract structured content."""
    extractor = SimpleHTMLExtractor()
    try:
        extractor.feed(html)
    except Exception:
        pass
    return extractor.extract()
