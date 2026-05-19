"""Configuration for web search module with automatic activation thresholds."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class WebSearchConfig:
    """Runtime configuration for web search fallback."""

    # Activation thresholds (combination: docs AND score)
    min_local_docs: int = 5  # If local docs < this, consider web search
    min_score_threshold: float = 0.3  # If avg score < this, consider web search
    
    # DuckDuckGo parameters
    ddg_max_results: int = 10  # Max results to fetch from DDG
    ddg_parallel_download: int = 5  # How many to download in parallel
    ddg_timeout: int = 12  # Timeout per URL
    
    # Storage and output
    output_jsonl: Path = Path("data") / "extracted" / "webpages" / "webpages.jsonl"
    save_to_jsonl: bool = True  # Persist results to main JSONL
    
    # Behavior
    user_agent: str = "IRS-WebSearch/1.0 (+academic-project)"
    min_chars: int = 300  # Minimum text length to accept
    delay: float = 0.5  # Delay between downloads (seconds)
    
    # Control
    enabled: bool = True  # Global on/off for web search
    fallback_only: bool = True  # If True, only use web search when local is insufficient
