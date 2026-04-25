"""Professional focused tech crawler for JSONL corpus generation.

Design goals:
- Keep output fully compatible with current indexing/search pipeline.
- Crawl bilingual tech sources (ES+EN) without language filtering.
- Store detected language as metadata only.
- Provide robust deduplication, polite crawling, and operational reports.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

DEFAULT_USER_AGENT = "IRS-TechCrawler/1.0 (+academic-project)"
DEFAULT_SEEDS_FILE = Path("scripts") / "tech_seeds.txt"
DEFAULT_OUTPUT = Path("data") / "extracted" / "webpages" / "webpages.jsonl"
DEFAULT_RAW_DIR = Path("data") / "raw" / "webpages"
DEFAULT_REPORT = Path("logs") / "crawl_report.txt"

TECH_KEYWORDS: Set[str] = {
    "software", "programming", "programacion", "programación", "developer", "desarrollador",
    "development", "desarrollo", "coding", "codificacion", "codificación", "python", "java",
    "javascript", "typescript", "rust", "golang", "go", "react", "angular", "vue", "django",
    "flask", "node", "api", "backend", "frontend", "microservices", "microservicios",
    "kubernetes", "devops", "cloud", "nube", "database", "base de datos", "algorithms",
    "algoritmos", "architecture", "arquitectura", "security", "seguridad", "machine learning",
    "aprendizaje automatico", "aprendizaje automático", "artificial intelligence",
    "inteligencia artificial", "data science", "ciencia de datos", "web development",
    "desarrollo web", "open source", "opensource",
}

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


@dataclass(order=True)
class QueueItem:
    """Priority queue item for crawl frontier."""

    priority: float
    depth: int
    url: str = field(compare=False)
    parent_url: Optional[str] = field(default=None, compare=False)


@dataclass
class CrawlConfig:
    """Runtime configuration for the crawler."""

    seeds_file: Path = DEFAULT_SEEDS_FILE
    output: Path = DEFAULT_OUTPUT
    raw_dir: Path = DEFAULT_RAW_DIR
    report: Path = DEFAULT_REPORT
    max_pages: int = 100
    max_depth: int = 2
    min_chars: int = 300
    per_domain_limit: int = 120
    timeout: int = 12
    delay: float = 1.0
    save_raw: bool = False
    user_agent: str = DEFAULT_USER_AGENT
    only_new: bool = False
    doc_id_mode: str = "int"
    log_level: str = "INFO"


@dataclass
class CrawlState:
    """Mutable crawl state and counters."""

    docs_written: int = 0
    frontier: List[QueueItem] = field(default_factory=list)
    visited_urls: Set[str] = field(default_factory=set)
    seen_urls: Set[str] = field(default_factory=set)
    existing_urls: Set[str] = field(default_factory=set)
    text_hashes: Set[str] = field(default_factory=set)
    domain_counts: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_access_by_domain: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    stats: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))


def utc_now_iso() -> str:
    """Return UTC timestamp in ISO 8601 format without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    """Normalize language code to simple lowercase tag when possible."""
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


def infer_language_from_text(text: str) -> Optional[str]:
    """Infer ES/EN language from stopword overlap. Returns None if uncertain."""
    snippet = (text or "")[:5000].lower()
    if not snippet:
        return None

    tokens = re.findall(r"[a-záéíóúñü]+", snippet)
    if len(tokens) < 40:
        return None

    token_set = set(tokens)
    scores: Dict[str, int] = {
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


def score_relevance(title: str, text: str, url: str) -> float:
    """Compute topic relevance score using bilingual tech keyword hits."""
    blob = f"{title} {text[:5000]} {url}".lower()
    return float(sum(1 for keyword in TECH_KEYWORDS if keyword in blob))


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
        title = " ".join(self._title_parts).strip()
        text = unescape(re.sub(r"\s+", " ", " ".join(self._text_parts))).strip()
        return title, text, self.links, self.language_hint


class TechCrawler:
    """Focused technology crawler with robust and maintainable architecture."""

    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self.state = CrawlState()
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.logger = logging.getLogger("tech_crawler")

        self._prepare_paths()
        if self.config.only_new:
            self._load_existing_output()
        self._seed_frontier(load_seeds(self.config.seeds_file))

    def _prepare_paths(self) -> None:
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        self.config.report.parent.mkdir(parents=True, exist_ok=True)

    def _seed_frontier(self, seeds: Iterable[str]) -> None:
        for raw_seed in seeds:
            seed = canonicalize_url(raw_seed)
            if not seed:
                self.state.stats["invalid_seed"] += 1
                continue

            if seed in self.state.seen_urls:
                continue

            self.state.seen_urls.add(seed)
            heapq.heappush(self.state.frontier, QueueItem(priority=-5.0, depth=0, url=seed, parent_url=None))

    def _load_existing_output(self) -> None:
        if not self.config.output.exists():
            return

        loaded = 0
        with self.config.output.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self.state.stats["malformed_existing_line"] += 1
                    continue

                raw_url = record.get("url")
                if isinstance(raw_url, str):
                    normalized = canonicalize_url(raw_url)
                    if normalized:
                        self.state.existing_urls.add(normalized)

                text = record.get("text")
                if isinstance(text, str) and text:
                    self.state.text_hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())

                loaded += 1

        self.state.stats["existing_loaded"] = loaded

    def _next_doc_id(self) -> int:
        if not self.config.output.exists():
            return 1

        max_doc_id = 0
        with self.config.output.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                doc_id = record.get("doc_id")
                if isinstance(doc_id, int) and doc_id > max_doc_id:
                    max_doc_id = doc_id

        return max_doc_id + 1

    def _robot_parser_for(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        domain_root = f"{parsed.scheme}://{parsed.netloc}"

        parser = self.robots_cache.get(domain_root)
        if parser is not None:
            return parser

        parser = RobotFileParser()
        parser.set_url(f"{domain_root}/robots.txt")
        try:
            parser.read()
        except Exception:
            self.state.stats["robots_fetch_error"] += 1

        self.robots_cache[domain_root] = parser
        return parser

    def _allowed_by_robots(self, url: str) -> bool:
        parser = self._robot_parser_for(url)
        try:
            return parser.can_fetch(self.config.user_agent, url)
        except Exception:
            return True

    def _throttle_domain(self, domain: str) -> None:
        now = time.time()
        elapsed = now - self.state.last_access_by_domain[domain]
        if elapsed < self.config.delay:
            time.sleep(self.config.delay - elapsed)
        self.state.last_access_by_domain[domain] = time.time()

    def _fetch_html(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        parsed = urlparse(url)
        self._throttle_domain(parsed.netloc)

        request = Request(url, headers={"User-Agent": self.config.user_agent})
        try:
            with urlopen(request, timeout=self.config.timeout) as response:
                content_type = (response.headers.get("Content-Type") or "").lower()
                if "text/html" not in content_type:
                    self.state.stats["non_html"] += 1
                    return None, None

                header_lang = normalize_language_code(response.headers.get("Content-Language"))
                charset = response.headers.get_content_charset() or "utf-8"
                html_bytes = response.read()
        except Exception as exc:
            self.logger.debug("fetch error for %s: %s", url, exc)
            self.state.stats["fetch_errors"] += 1
            return None, None

        try:
            html = html_bytes.decode(charset, errors="replace")
        except LookupError:
            html = html_bytes.decode("utf-8", errors="replace")

        return html, header_lang

    def _extract_page(self, html: str) -> Tuple[str, str, List[str], Optional[str]]:
        extractor = SimpleHTMLExtractor()
        extractor.feed(html)
        return extractor.extract()

    def _save_raw_html(self, url: str, html: str) -> None:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        target = self.config.raw_dir / f"{digest}.html"
        target.write_text(html, encoding="utf-8")

    def _iter_clean_links(self, base_url: str, raw_links: Iterable[str]) -> Iterable[str]:
        for href in raw_links:
            absolute = urljoin(base_url, href)
            normalized = canonicalize_url(absolute)
            if not normalized:
                continue

            parsed = urlparse(normalized)
            if parsed.scheme not in {"http", "https"}:
                continue

            if parsed.path.lower().endswith(BINARY_EXTENSIONS):
                continue

            yield normalized

    def _detect_language_metadata(
        self,
        title: str,
        text: str,
        url: str,
        html_hint: Optional[str],
        header_hint: Optional[str],
    ) -> str:
        for hint in (html_hint, header_hint, infer_language_from_url(url), infer_language_from_text(f"{title} {text}")):
            normalized = normalize_language_code(hint)
            if normalized:
                return normalized
        return "unknown"

    def _record_doc_id(self, next_doc_id: int, url: str) -> object:
        if self.config.doc_id_mode == "hash":
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            return f"web_{digest}"
        return next_doc_id

    def _append_record(self, record: Dict[str, object]) -> None:
        with self.config.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_report(self) -> None:
        lines = [
            f"generated_at: {utc_now_iso()}",
            f"seeds_file: {self.config.seeds_file}",
            f"only_new: {self.config.only_new}",
            f"docs_written: {self.state.docs_written}",
            f"frontier_remaining: {len(self.state.frontier)}",
            "stats:",
        ]

        for key in sorted(self.state.stats):
            lines.append(f"  - {key}: {self.state.stats[key]}")

        lines.append("domains:")
        for domain, count in sorted(self.state.domain_counts.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"  - {domain}: {count}")

        self.config.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def crawl(self) -> None:
        next_doc_id = self._next_doc_id()
        self.logger.info("crawl started with %d seed/frontier URLs", len(self.state.frontier))

        while self.state.frontier and self.state.docs_written < self.config.max_pages:
            current = heapq.heappop(self.state.frontier)
            url = current.url
            depth = current.depth

            if url in self.state.visited_urls:
                self.state.stats["already_visited"] += 1
                continue

            if self.config.only_new and url in self.state.existing_urls:
                self.state.stats["already_in_output"] += 1
                continue

            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if self.state.domain_counts[domain] >= self.config.per_domain_limit:
                self.state.stats["domain_limit_skips"] += 1
                continue

            if not self._allowed_by_robots(url):
                self.state.stats["blocked_by_robots"] += 1
                continue

            self.state.visited_urls.add(url)
            self.state.domain_counts[domain] += 1
            self.state.stats["visited"] += 1

            html, header_language = self._fetch_html(url)
            if not html:
                continue

            if self.config.save_raw:
                self._save_raw_html(url, html)

            title, text, raw_links, html_language = self._extract_page(html)
            if len(text) < self.config.min_chars:
                self.state.stats["too_short"] += 1
                continue

            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in self.state.text_hashes:
                self.state.stats["duplicate_text"] += 1
                continue
            self.state.text_hashes.add(text_hash)

            relevance = score_relevance(title=title, text=text, url=url)
            if relevance <= 0.0:
                self.state.stats["low_relevance"] += 1
                continue

            out_links: List[str] = []
            for link in self._iter_clean_links(url, raw_links):
                out_links.append(link)
                if depth + 1 > self.config.max_depth:
                    continue
                if link in self.state.seen_urls or link in self.state.visited_urls:
                    continue

                self.state.seen_urls.add(link)
                child_priority = -max(0.1, relevance / 2.0)
                heapq.heappush(
                    self.state.frontier,
                    QueueItem(priority=child_priority, depth=depth + 1, url=link, parent_url=url),
                )

            language_metadata = self._detect_language_metadata(
                title=title,
                text=text,
                url=url,
                html_hint=html_language,
                header_hint=header_language,
            )

            record = {
                "doc_id": self._record_doc_id(next_doc_id=next_doc_id, url=url),
                "url": url,
                "title": title,
                "domain": domain,
                "crawl_date": utc_now_iso(),
                "text": text,
                "source_type": "webpage",
                "language": language_metadata,
                "out_links": out_links,
                "relevance_score": relevance,
                "depth": depth,
                "parent_url": current.parent_url,
            }
            self._append_record(record)

            if self.config.only_new:
                self.state.existing_urls.add(url)

            if self.config.doc_id_mode == "int":
                next_doc_id += 1

            self.state.docs_written += 1
            self.state.stats["written"] += 1

            if self.state.docs_written % 10 == 0:
                self.logger.info(
                    "progress: docs_written=%d frontier=%d",
                    self.state.docs_written,
                    len(self.state.frontier),
                )

        self._write_report()
        self.logger.info("crawl finished: docs_written=%d", self.state.docs_written)


def load_seeds(path: Path) -> List[str]:
    """Load seed URLs, skipping comments and blank lines."""
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
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description="Focused bilingual technology crawler")

    parser.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)

    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--per-domain-limit", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--delay", type=float, default=1.0)

    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--only-new", action="store_true")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument(
        "--doc-id-mode",
        choices=["int", "hash"],
        default="int",
        help="int: compatible with current indexer/GVSM; hash: web_<hash>",
    )

    return parser


def build_config_from_args(args: argparse.Namespace) -> CrawlConfig:
    """Convert parsed arguments into CrawlConfig."""
    return CrawlConfig(
        seeds_file=args.seeds_file,
        output=args.output,
        raw_dir=args.raw_dir,
        report=args.report,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        min_chars=args.min_chars,
        per_domain_limit=args.per_domain_limit,
        timeout=args.timeout,
        delay=args.delay,
        save_raw=args.save_raw,
        user_agent=args.user_agent,
        only_new=args.only_new,
        doc_id_mode=args.doc_id_mode,
        log_level=args.log_level,
    )


def configure_logging(level: str) -> None:
    """Configure application logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    config = build_config_from_args(args)

    configure_logging(config.log_level)
    crawler = TechCrawler(config)
    crawler.crawl()


if __name__ == "__main__":
    main()
