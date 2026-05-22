"""DuckDuckGo scraper: fetch results and download URLs in parallel."""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from .extractor import extract_page_content
from .utils import canonicalize_url, detect_language_metadata


logger = logging.getLogger("web_search.ddg")


def _parse_ddg_results(html: str) -> List[str]:
    """Extract URLs from DuckDuckGo HTML response (simple regex-based parsing)."""
    urls = []

    def _add_url(raw_url: str) -> None:
        url = (raw_url or "").strip()
        if not url:
            return

        # DuckDuckGo often wraps the final target in /l/?uddg=<encoded_url>
        if url.startswith("/l/") or url.startswith("//duckduckgo.com/l/"):
            parsed = urlparse(url)
            query = parse_qs(parsed.query)
            uddg_values = query.get("uddg") or []
            if uddg_values:
                url = unquote(uddg_values[0])

        if not url.startswith("http"):
            return

        normalized = canonicalize_url(url)
        if not normalized:
            return

        # Filter obvious non-result/documentation artifacts from the DDG page.
        if "duckduckgo.com" in normalized:
            return
        if normalized.endswith("xhtml1-transitional.dtd"):
            return
        if normalized.endswith("/1999/xhtml"):
            return

        if normalized not in urls:
            urls.append(normalized)

    # DuckDuckGo HTML usually exposes result links via result__a anchors and
    # visible URLs via result__url. Extract both to maximize robustness.
    patterns = [
        r'<a[^>]+class=["\'][^"\']*result__a[^"\']*["\'][^>]+href=["\']([^"\']+)["\']',
        r'<a[^>]+href=["\']([^"\']+)["\'][^>]+class=["\'][^"\']*result__a[^"\']*["\']',
        r'<span[^>]+class=["\'][^"\']*result__url[^"\']*["\'][^>]*>([^<]+)</span>',
    ]

    for pattern in patterns:
        for match in re.findall(pattern, html, flags=re.IGNORECASE | re.DOTALL):
            _add_url(match)

    # Fallback: look for common URL patterns
    if not urls:
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]*'
        matches = re.findall(url_pattern, html)
        for match in matches:
            url = match.strip().rstrip('.,;:)')
            if url and len(url) > 10:
                _add_url(url)
    
    return urls


def _fetch_ddg_search(query: str, user_agent: str = "Mozilla/5.0") -> Optional[str]:
    """Fetch DuckDuckGo search results page."""
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}&kl=us-en"
    
    try:
        request = Request(
            search_url,
            headers={
                "User-Agent": (
                    user_agent
                    or "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Upgrade-Insecure-Requests": "1",
            },
        )
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
