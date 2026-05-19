"""DuckDuckGo scraper: fetch results and download URLs in parallel."""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from .extractor import extract_page_content
from .utils import canonicalize_url, detect_language_metadata


logger = logging.getLogger("web_search.ddg")


def _parse_ddg_results(html: str) -> List[str]:
    """Extract URLs from DuckDuckGo HTML response (simple regex-based parsing)."""
    # This is a basic scraper; DuckDuckGo HTML structure may change
    urls = []
    
    # Look for href patterns in DuckDuckGo results
    # DuckDuckGo typically returns results with data-url attributes or href links
    pattern = r'(?:data-url=|href=)["\']([^"\']+)["\']'
    matches = re.findall(pattern, html)
    
    for match in matches:
        url = match.strip()
        if url and url.startswith("http"):
            normalized = canonicalize_url(url)
            if normalized and normalized not in urls:
                urls.append(normalized)
    
    # Fallback: look for common URL patterns
    if not urls:
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]*'
        matches = re.findall(url_pattern, html)
        for match in matches:
            url = match.strip().rstrip('.,;:)')
            if url and len(url) > 10:
                normalized = canonicalize_url(url)
                if normalized and normalized not in urls and "duckduckgo.com" not in normalized:
                    urls.append(normalized)
    
    return urls


def _fetch_ddg_search(query: str, user_agent: str = "Mozilla/5.0") -> Optional[str]:
    """Fetch DuckDuckGo search results page."""
    search_url = f"https://html.duckduckgo.com/?q={query.replace(' ', '+')}&kl=us-en"
    
    try:
        request = Request(search_url, headers={"User-Agent": user_agent})
        with urlopen(request, timeout=10) as response:
            if response.status == 200:
                html = response.read().decode("utf-8", errors="replace")
                return html
    except Exception as exc:
        logger.debug("DuckDuckGo fetch error: %s", exc)
    
    return None


def _download_url(
    url: str,
    user_agent: str,
    timeout: int = 10,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Download a single URL and extract content. Returns (url, title, text, language, html_hint)."""
    try:
        request = Request(url, headers={"User-Agent": user_agent})
        with urlopen(request, timeout=timeout) as response:
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "text/html" not in content_type:
                logger.debug("Skipping non-HTML: %s", url)
                return url, None, None, None, None

            header_lang = response.headers.get("Content-Language")
            charset = response.headers.get_content_charset() or "utf-8"
            html_bytes = response.read()

        try:
            html = html_bytes.decode(charset, errors="replace")
        except LookupError:
            html = html_bytes.decode("utf-8", errors="replace")

        title, text, _, html_hint = extract_page_content(html)
        return url, title, text, header_lang, html_hint
    except Exception as exc:
        logger.debug("Download error for %s: %s", url, exc)
        return url, None, None, None, None


class DDGScraper:
    """DuckDuckGo search and parallel download engine."""

    def __init__(
        self,
        max_results: int = 10,
        parallel_download: int = 5,
        timeout: int = 12,
        user_agent: str = "Mozilla/5.0",
        delay: float = 0.5,
    ) -> None:
        self.max_results = max_results
        self.parallel_download = parallel_download
        self.timeout = timeout
        self.user_agent = user_agent
        self.delay = delay

    def search(self, query: str) -> List[dict]:
        """Search query on DuckDuckGo, download top URLs in parallel, return records."""
        logger.info("Searching DDG for: %s", query)

        # Fetch DDG search page
        html = _fetch_ddg_search(query, user_agent=self.user_agent)
        if not html:
            logger.warning("Failed to fetch DDG results for: %s", query)
            return []

        # Parse URLs from results
        urls = _parse_ddg_results(html)
        urls = urls[: self.max_results]
        if not urls:
            logger.warning("No URLs found in DDG results for: %s", query)
            return []

        logger.info("Found %d URLs from DDG, downloading up to %d in parallel", len(urls), self.parallel_download)

        # Download URLs in parallel (limit concurrent workers, not total URLs)
        records: List[dict] = []
        with ThreadPoolExecutor(max_workers=self.parallel_download) as executor:
            futures = {
                executor.submit(_download_url, url, self.user_agent, self.timeout): (idx, url)
                for idx, url in enumerate(urls)
            }

            ordered_results = {}
            for future in as_completed(futures):
                idx, url = futures[future]
                url, title, text, header_lang, html_hint = future.result()
                
                # Skip if extraction failed
                if not text or not title:
                    logger.debug("Skipped %s: no content extracted", url)
                    continue

                # Detect language
                language = detect_language_metadata(
                    title=title,
                    text=text,
                    url=url,
                    html_hint=html_hint,
                    header_hint=header_lang,
                )

                # Create record
                record = {
                    "url": url,
                    "title": title,
                    "text": text,
                    "language": language,
                    "source_type": "web_search",
                    "domain": urlparse(url).netloc.lower(),
                    "web_rank": idx,
                }
                ordered_results[idx] = record

                time.sleep(self.delay)

        for idx in sorted(ordered_results):
            records.append(ordered_results[idx])

        logger.info("Successfully extracted %d records from DDG search", len(records))
        return records
