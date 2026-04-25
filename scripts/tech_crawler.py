"""Focused technology crawler for JSONL corpus generation.

This script crawls a set of seed URLs, respects robots.txt, applies
basic relevance filtering for the technology domain, and writes cleaned
documents to `data/extracted/webpages/webpages.jsonl`.

The output schema is compatible with the current indexing pipeline,
including required fields `doc_id` and `text`.
"""

from __future__ import annotations   # Enables postponed evaluation of type hints
import argparse                      # Parses command-line arguments
import hashlib                       # Provides hashing functions (e.g., SHA-256)
import heapq                         # Implements priority queues / heaps
import json                          # Reads and writes JSON data
import os                            # Interacts with the operating system (paths, env vars)
import re                            # Regular expressions for text matching
import time                          # Time utilities (sleep, timestamps)
from collections import defaultdict  # Dict that auto-creates default values
from dataclasses import dataclass    # Simplifies creation of data-holding classes
from datetime import datetime, timezone  # Date/time handling with timezone support
from html import unescape            # Converts HTML entities to normal characters
from html.parser import HTMLParser   # Basic HTML parsing utilities
from pathlib import Path             # Object-oriented filesystem paths
from typing import Dict, Iterable, List, Optional, Set, Tuple  # Type hints
from urllib.parse import urljoin, urlparse, urlunparse  # URL manipulation helpers
from urllib.request import Request, urlopen             # HTTP requests
from urllib.robotparser import RobotFileParser          # Parses robots.txt rules

# Default configuration values
DEFAULT_USER_AGENT = "IRS-TechCrawler/0.1 (+academic-project)"
DEFAULT_SEEDS_FILE = Path("scripts") / "tech_seeds.txt"
DEFAULT_OUTPUT = Path("data") / "extracted" / "webpages" / "webpages.jsonl"
DEFAULT_RAW_DIR = Path("data") / "raw" / "webpages"
DEFAULT_REPORT = Path("logs") / "crawl_report.txt"

DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = {"en", "es"}

TECH_KEYWORDS_BY_LANGUAGE: Dict[str, Set[str]] = {
    "en": {
        "software",
        "programming",
        "developer",
        "development",
        "coding",
        "python",
        "java",
        "javascript",
        "typescript",
        "rust",
        "golang",
        "go",
        "react",
        "angular",
        "vue",
        "django",
        "flask",
        "node",
        "api",
        "backend",
        "frontend",
        "microservices",
        "kubernetes",
        "devops",
        "cloud",
        "database",
        "algorithms",
        "architecture",
        "security",
        "machine learning",
        "artificial intelligence",
        "data science",
        "web development",
    },
    "es": {
        "software",
        "programacion",
        "programación",
        "desarrollador",
        "desarrollo",
        "codificacion",
        "codificación",
        "python",
        "java",
        "javascript",
        "typescript",
        "rust",
        "golang",
        "go",
        "react",
        "angular",
        "vue",
        "django",
        "flask",
        "nodo",
        "api",
        "backend",
        "frontend",
        "microservicios",
        "kubernetes",
        "devops",
        "nube",
        "base de datos",
        "algoritmos",
        "arquitectura",
        "seguridad",
        "aprendizaje automatico",
        "aprendizaje automático",
        "inteligencia artificial",
        "ciencia de datos",
        "desarrollo web",
    },
}


@dataclass(frozen=True)
class LanguageProfile:
    """Language-specific relevance profile used to score pages."""

    code: str
    label: str
    keywords: Set[str]


LANGUAGE_PROFILES: Dict[str, LanguageProfile] = {
    code: LanguageProfile(code=code, label=("English" if code == "en" else "Español"), keywords=keywords)
    for code, keywords in TECH_KEYWORDS_BY_LANGUAGE.items()
}

def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format (no microseconds)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def canonicalize_url(raw_url: str) -> Optional[str]:
    """Normalize and validate a URL; return canonical http(s) URL or None."""
    parsed = urlparse(raw_url.strip())
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    clean_path = re.sub(r"/+", "/", parsed.path or "/")
    if clean_path != "/" and clean_path.endswith("/"):
        clean_path = clean_path[:-1]
    canonical = parsed._replace(fragment="", params="", query="", path=clean_path)
    return urlunparse(canonical)


def normalize_language_code(raw_language: Optional[str]) -> Optional[str]:
    """Normalize language hints to supported profile codes when possible."""
    if not raw_language:
        return None

    lang = raw_language.strip().lower().replace("_", "-")
    if not lang:
        return None

    if lang.startswith("en"):
        return "en"
    if lang.startswith("es"):
        return "es"
    return None

class SimpleHTMLExtractor(HTMLParser):
    """Extract title, visible text and links from HTML."""

    def __init__(self) -> None:
        """Initialize parser state for extracting title, visible text and links."""
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: List[str] = []
        self._text_parts: List[str] = []
        self.links: List[str] = []
        self.detected_language: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        """Process start tag; collect links and track title/skip regions."""
        tag = tag.lower()
        attr_map = {
            name.lower(): value.strip()
            for name, value in attrs
            if name and value and isinstance(value, str)
        }

        if tag == "html" and self.detected_language is None:
            hinted = normalize_language_code(attr_map.get("lang"))
            if hinted:
                self.detected_language = hinted

        if tag == "meta" and self.detected_language is None:
            name = attr_map.get("name", "").lower()
            prop = attr_map.get("property", "").lower()
            http_equiv = attr_map.get("http-equiv", "").lower()
            content = attr_map.get("content")
            if name == "language" and content:
                hinted = normalize_language_code(content)
                if hinted:
                    self.detected_language = hinted
            elif http_equiv == "content-language" and content:
                hinted = normalize_language_code(content)
                if hinted:
                    self.detected_language = hinted
            elif prop == "og:locale" and content:
                hinted = normalize_language_code(content)
                if hinted:
                    self.detected_language = hinted

        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True

        if tag == "a":
            for name, value in attrs:
                if name.lower() == "href" and value:
                    self.links.append(value.strip())
                    break

    def handle_endtag(self, tag: str) -> None:
        """Process end tag; update title state and skip depth."""
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        """Collect visible text unless inside skipped sections."""
        if self._skip_depth > 0:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self._title_parts.append(text)
        self._text_parts.append(text)

    def extracted(self) -> Tuple[str, str, List[str], Optional[str]]:
        """Return title, cleaned text, links, and detected language hint."""
        title = " ".join(self._title_parts).strip()
        text = " ".join(self._text_parts)
        text = unescape(re.sub(r"\s+", " ", text)).strip()
        return title, text, self.links, self.detected_language

@dataclass(order=True)
class QueueItem:
    """Priority queue item for frontier scheduling."""

    priority: float
    depth: int
    url: str
    parent_url: Optional[str]

class FocusedCrawler:
    """Focused crawler with robots.txt, limits, and relevance filtering."""

    def __init__(
        self,
        seeds: Iterable[str],
        output_path: Path,
        raw_dir: Path,
        report_path: Path,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        max_pages: int = 500,
        max_depth: int = 2,
        min_chars: int = 300,
        per_domain_limit: int = 120,
        request_timeout: int = 12,
        domain_delay_sec: float = 1.0,
        save_raw: bool = False,
        doc_id_mode: str = "int",
        language: str = DEFAULT_LANGUAGE,
        only_new: bool = False,
    ):
        language = (language or DEFAULT_LANGUAGE).lower().strip()
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language must be one of {sorted(SUPPORTED_LANGUAGES)}")

        self.user_agent = user_agent
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.min_chars = min_chars
        self.per_domain_limit = per_domain_limit
        self.request_timeout = request_timeout
        self.domain_delay_sec = domain_delay_sec
        self.save_raw = save_raw
        self.doc_id_mode = doc_id_mode
        self.language = language
        self.only_new = only_new
        """Initialize crawler with seeds, output dirs, and runtime limits."""

        self.output_path = output_path
        self.raw_dir = raw_dir
        self.report_path = report_path

        self.frontier: List[QueueItem] = []
        self.visited: Set[str] = set()
        self.seen_urls: Set[str] = set()
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.last_access_by_domain: Dict[str, float] = defaultdict(lambda: 0.0)
        self.text_hashes: Set[str] = set()
        self.docs_written = 0
        self.stats: Dict[str, int] = defaultdict(int)
        self.domain_counts: Dict[str, int] = defaultdict(int)
        self.existing_urls: Set[str] = set()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        if self.only_new:
            self._load_existing_documents()

        for seed in seeds:
            canonical = canonicalize_url(seed)
            if not canonical:
                continue
            item = QueueItem(priority=-3.0, depth=0, url=canonical, parent_url=None)
            heapq.heappush(self.frontier, item)
            self.seen_urls.add(canonical)

    def _load_existing_documents(self) -> None:
        """Load already indexed URLs/text hashes from output JSONL for incremental crawling."""
        if not self.output_path.exists():
            return

        loaded = 0
        with self.output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                raw_url = obj.get("url")
                if isinstance(raw_url, str):
                    canonical = canonicalize_url(raw_url)
                    if canonical:
                        self.existing_urls.add(canonical)

                text = obj.get("text")
                if isinstance(text, str) and text:
                    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    self.text_hashes.add(text_hash)

                loaded += 1

        self.stats["existing_loaded"] = loaded

    def _next_doc_id(self) -> int:
        """Scan the output file and return the next integer document id."""
        if not self.output_path.exists():
            return 1

        last_doc_id = 0
        with self.output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_doc_id = obj.get("doc_id")
                if isinstance(raw_doc_id, int) and raw_doc_id > last_doc_id:
                    last_doc_id = raw_doc_id
        return last_doc_id + 1

    def _robot_parser_for(self, url: str) -> RobotFileParser:
        """Return a cached RobotFileParser for the URL's base domain."""
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        parser = self.robots_cache.get(base)
        if parser is not None:
            return parser

        robots_url = f"{base}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.read()
        except Exception:
            pass
        self.robots_cache[base] = parser
        return parser

    def _allowed_by_robots(self, url: str) -> bool:
        """Check whether the crawler is allowed to fetch `url` per robots.txt."""
        parser = self._robot_parser_for(url)
        try:
            return parser.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def _wait_if_needed(self, domain: str) -> None:
        """Enforce domain-specific delay between requests to avoid hammering."""
        now = time.time()
        elapsed = now - self.last_access_by_domain[domain]
        if elapsed < self.domain_delay_sec:
            time.sleep(self.domain_delay_sec - elapsed)
        self.last_access_by_domain[domain] = time.time()

    @staticmethod
    def _score_keywords(blob: str, keywords: Set[str]) -> float:
        """Score a text blob by counting keyword hits."""
        score = 0.0
        for keyword in keywords:
            if keyword in blob:
                score += 1.0
        return score

    def _resolve_language_profile(
        self,
        title: str,
        text: str,
        url: str,
        hinted_language: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Return selected language and relevance score, validating hinted page language first."""
        blob = f"{title} {text[:5000]} {url}".lower()
        hinted = normalize_language_code(hinted_language)

        if hinted and hinted != self.language:
            self.stats["language_mismatch"] += 1
            return hinted, 0.0

        profile = LANGUAGE_PROFILES[self.language]
        return profile.code, self._score_keywords(blob, profile.keywords)

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML content for `url`; return decoded text or None on error."""
        parsed = urlparse(url)
        self._wait_if_needed(parsed.netloc)

        request = Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                content_type = (response.headers.get("Content-Type") or "").lower()
                if "text/html" not in content_type:
                    self.stats["non_html"] += 1
                    return None
                charset = response.headers.get_content_charset() or "utf-8"
                html_bytes = response.read()
        except Exception:
            self.stats["fetch_errors"] += 1
            return None

        try:
            return html_bytes.decode(charset, errors="replace")
        except LookupError:
            return html_bytes.decode("utf-8", errors="replace")

    def _relevance_score(
        self,
        title: str,
        text: str,
        url: str,
        hinted_language: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Compute the page language and a relevance score for the selected profile."""
        return self._resolve_language_profile(title, text, url, hinted_language=hinted_language)

    def _extract(self, html: str) -> Tuple[str, str, List[str], Optional[str]]:
        """Parse HTML and return (title, visible text, links, detected_language)."""
        parser = SimpleHTMLExtractor()
        parser.feed(html)
        return parser.extracted()

    def _iter_clean_links(self, current_url: str, raw_links: Iterable[str]) -> Iterable[str]:
        """Yield canonical, allowed, non-binary links resolved from raw hrefs."""
        for href in raw_links:
            absolute = urljoin(current_url, href)
            canonical = canonicalize_url(absolute)
            if not canonical:
                continue
            parsed = urlparse(canonical)
            if parsed.scheme not in {"http", "https"}:
                continue
            path = parsed.path.lower()
            if path.endswith((".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4", ".mp3")):
                continue
            yield canonical

    def _save_raw_html(self, url: str, html: str) -> None:
        """Save raw HTML content to the raw directory using a hashed filename."""
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        target = self.raw_dir / f"{digest}.html"
        target.write_text(html, encoding="utf-8")

    def _append_record(self, record: Dict[str, object]) -> None:
        """Append a JSON record object as a new line to the output JSONL file."""
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _webpage_doc_id(self, next_doc_id: int, url: str) -> object:
        """Return the document id for a webpage (int or hashed string)."""
        if self.doc_id_mode == "hash":
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            return f"web_{digest}"
        return next_doc_id

    def crawl(self) -> None:
        """Main crawl loop: fetch pages, apply filters, save records, and enqueue links."""
        next_doc_id = self._next_doc_id()

        while self.frontier and self.docs_written < self.max_pages:
            current = heapq.heappop(self.frontier)
            url = current.url
            depth = current.depth

            if url in self.visited:
                self.stats["already_visited"] += 1
                continue

            if self.only_new and url in self.existing_urls:
                self.stats["already_in_output"] += 1
                continue

            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if self.domain_counts[domain] >= self.per_domain_limit:
                self.stats["domain_limit_skips"] += 1
                continue

            if not self._allowed_by_robots(url):
                self.stats["blocked_by_robots"] += 1
                continue

            self.visited.add(url)
            self.domain_counts[domain] += 1
            self.stats["visited"] += 1

            html = self._fetch_html(url)
            if not html:
                continue

            if self.save_raw:
                self._save_raw_html(url, html)

            title, text, out_links_raw, hinted_language = self._extract(html)
            if len(text) < self.min_chars:
                self.stats["too_short"] += 1
                continue

            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in self.text_hashes:
                self.stats["duplicate_text"] += 1
                continue
            self.text_hashes.add(text_hash)

            content_language, relevance = self._relevance_score(
                title,
                text,
                url,
                hinted_language=hinted_language,
            )
            if relevance <= 0.0:
                self.stats["low_relevance"] += 1
                continue

            out_links: List[str] = []
            for link in self._iter_clean_links(url, out_links_raw):
                out_links.append(link)
                if depth + 1 > self.max_depth:
                    continue
                if link in self.seen_urls or link in self.visited:
                    continue
                self.seen_urls.add(link)
                child_priority = -max(0.1, relevance / 2)
                heapq.heappush(
                    self.frontier,
                    QueueItem(priority=child_priority, depth=depth + 1, url=link, parent_url=url),
                )

            record = {
                "doc_id": self._webpage_doc_id(next_doc_id, url),
                "url": url,
                "title": title,
                "domain": domain,
                "crawl_date": utc_now_iso(),
                "text": text,
                "source_type": "webpage",
                "language": content_language,
                "out_links": out_links,
                "relevance_score": relevance,
                "depth": depth,
                "parent_url": current.parent_url,
            }
            self._append_record(record)
            if self.only_new:
                self.existing_urls.add(url)

            if self.doc_id_mode == "int":
                next_doc_id += 1
            self.docs_written += 1
            self.stats["written"] += 1

        self._write_report()

    def _write_report(self) -> None:
        """Write a simple crawl report summarizing statistics and domains."""
        lines = [
            f"generated_at: {utc_now_iso()}",
            f"language: {self.language}",
            f"only_new: {self.only_new}",
            f"docs_written: {self.docs_written}",
            f"frontier_remaining: {len(self.frontier)}",
            "stats:",
        ]
        for key in sorted(self.stats):
            lines.append(f"  - {key}: {self.stats[key]}")
        lines.append("domains:")
        for domain, count in sorted(self.domain_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {domain}: {count}")

        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def load_seeds(path: Path) -> List[str]:
    """Load seed URLs from `path`, skipping blank lines and comments."""
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

def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser for the crawler."""
    parser = argparse.ArgumentParser(description="Focused technology crawler")
    parser.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--per-domain-limit", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Evita reinsertar documentos ya presentes en el output (por URL y hash de texto)",
    )
    parser.add_argument(
        "--language",
        choices=sorted(SUPPORTED_LANGUAGES),
        default=DEFAULT_LANGUAGE,
        help="Idioma de crawl: valida idioma detectado de la página y luego puntúa por keywords",
    )
    parser.add_argument(
        "--doc-id-mode",
        choices=["int", "hash"],
        default="int",
        help="int: compatible con indexador/GVSM actual; hash: formato tipo web_<hash>",
    )
    return parser

def main() -> None:
    """Parse arguments, load seeds, instantiate the crawler, and run it."""
    args = build_parser().parse_args()
    seeds = load_seeds(args.seeds_file)

    crawler = FocusedCrawler(
        seeds=seeds,
        output_path=args.output,
        raw_dir=args.raw_dir,
        report_path=args.report,
        user_agent=args.user_agent,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        min_chars=args.min_chars,
        per_domain_limit=args.per_domain_limit,
        request_timeout=args.timeout,
        domain_delay_sec=args.delay,
        save_raw=args.save_raw,
        doc_id_mode=args.doc_id_mode,
        language=args.language,
        only_new=args.only_new,
    )
    crawler.crawl()

if __name__ == "__main__":
    main()
