"""Focused technology crawler for JSONL corpus generation.

This script crawls a set of seed URLs, respects robots.txt, applies
basic relevance filtering for the technology domain, and writes cleaned
documents to `data/extracted/webpages/webpages.jsonl`.

The output schema is compatible with the current indexing pipeline,
including required fields `doc_id` and `text`.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser


DEFAULT_USER_AGENT = "IRS-TechCrawler/0.1 (+academic-project)"
DEFAULT_SEEDS_FILE = Path("scripts") / "tech_seeds.txt"
DEFAULT_OUTPUT = Path("data") / "extracted" / "webpages" / "webpages.jsonl"
DEFAULT_RAW_DIR = Path("data") / "raw" / "webpages"
DEFAULT_REPORT = Path("logs") / "crawl_report.txt"

TECH_KEYWORDS: Set[str] = {
    "software",
    "programming",
    "developer",
    "development",
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
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonicalize_url(raw_url: str) -> Optional[str]:
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


class SimpleHTMLExtractor(HTMLParser):
    """Extract title, visible text and links from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: List[str] = []
        self._text_parts: List[str] = []
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
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

    def extracted(self) -> Tuple[str, str, List[str]]:
        title = " ".join(self._title_parts).strip()
        text = " ".join(self._text_parts)
        text = unescape(re.sub(r"\s+", " ", text)).strip()
        return title, text, self.links


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
    ):
        self.user_agent = user_agent
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.min_chars = min_chars
        self.per_domain_limit = per_domain_limit
        self.request_timeout = request_timeout
        self.domain_delay_sec = domain_delay_sec
        self.save_raw = save_raw
        self.doc_id_mode = doc_id_mode

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

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            canonical = canonicalize_url(seed)
            if not canonical:
                continue
            item = QueueItem(priority=-3.0, depth=0, url=canonical, parent_url=None)
            heapq.heappush(self.frontier, item)
            self.seen_urls.add(canonical)

    def _next_doc_id(self) -> int:
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
        parser = self._robot_parser_for(url)
        try:
            return parser.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def _wait_if_needed(self, domain: str) -> None:
        now = time.time()
        elapsed = now - self.last_access_by_domain[domain]
        if elapsed < self.domain_delay_sec:
            time.sleep(self.domain_delay_sec - elapsed)
        self.last_access_by_domain[domain] = time.time()

    def _fetch_html(self, url: str) -> Optional[str]:
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

    @staticmethod
    def _relevance_score(title: str, text: str, url: str) -> float:
        blob = f"{title} {text[:5000]} {url}".lower()
        score = 0.0
        for kw in TECH_KEYWORDS:
            if kw in blob:
                score += 1.0
        return score

    def _extract(self, html: str) -> Tuple[str, str, List[str]]:
        parser = SimpleHTMLExtractor()
        parser.feed(html)
        return parser.extracted()

    def _iter_clean_links(self, current_url: str, raw_links: Iterable[str]) -> Iterable[str]:
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
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        target = self.raw_dir / f"{digest}.html"
        target.write_text(html, encoding="utf-8")

    def _append_record(self, record: Dict[str, object]) -> None:
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _webpage_doc_id(self, next_doc_id: int, url: str) -> object:
        if self.doc_id_mode == "hash":
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            return f"web_{digest}"
        return next_doc_id

    def crawl(self) -> None:
        next_doc_id = self._next_doc_id()

        while self.frontier and self.docs_written < self.max_pages:
            current = heapq.heappop(self.frontier)
            url = current.url
            depth = current.depth

            if url in self.visited:
                self.stats["already_visited"] += 1
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

            title, text, out_links_raw = self._extract(html)
            if len(text) < self.min_chars:
                self.stats["too_short"] += 1
                continue

            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in self.text_hashes:
                self.stats["duplicate_text"] += 1
                continue
            self.text_hashes.add(text_hash)

            relevance = self._relevance_score(title, text, url)
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
                "out_links": out_links,
                "relevance_score": relevance,
                "depth": depth,
                "parent_url": current.parent_url,
            }
            self._append_record(record)

            if self.doc_id_mode == "int":
                next_doc_id += 1
            self.docs_written += 1
            self.stats["written"] += 1

        self._write_report()

    def _write_report(self) -> None:
        lines = [
            f"generated_at: {utc_now_iso()}",
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
    parser = argparse.ArgumentParser(description="Focused technology crawler")
    parser.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--per-domain-limit", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--doc-id-mode",
        choices=["int", "hash"],
        default="int",
        help="int: compatible con indexador/GVSM actual; hash: formato tipo web_<hash>",
    )
    return parser


def main() -> None:
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
    )
    crawler.crawl()


if __name__ == "__main__":
    main()
