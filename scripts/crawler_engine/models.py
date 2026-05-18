from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, List, Optional, Set


DEFAULT_USER_AGENT = "IRS-TechCrawler/1.1 (+academic-project)"
DEFAULT_SEEDS_FILE = Path("scripts") / "tech_seeds.txt"
DEFAULT_OUTPUT = Path("data") / "extracted" / "webpages" / "webpages.jsonl"
DEFAULT_RAW_DIR = Path("data") / "raw" / "webpages"
DEFAULT_REPORT = Path("logs") / "crawl_report.txt"


@dataclass(order=True)
class QueueItem:
    """Priority queue element used by crawl frontier."""

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
    """Mutable runtime state and counters."""

    docs_written: int = 0
    frontier: List[QueueItem] = field(default_factory=list)
    visited_urls: Set[str] = field(default_factory=set)
    seen_urls: Set[str] = field(default_factory=set)
    existing_urls: Set[str] = field(default_factory=set)
    text_hashes: Set[str] = field(default_factory=set)
    domain_counts: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_access_by_domain: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    stats: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
