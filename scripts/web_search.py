"""Web search orchestrator: manage fallback search, extraction, and indexing."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from web_search_engine import DDGScraper, WebSearchConfig


logger = logging.getLogger("web_search")


def utc_now_iso() -> str:
    """Return UTC timestamp in ISO 8601 format without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class WebSearchPipeline:
    """Orchestrate web search, indexing, and JSONL storage."""

    def __init__(self, config: Optional[WebSearchConfig] = None) -> None:
        self.config = config or WebSearchConfig()
        self.scraper = DDGScraper(
            max_results=self.config.ddg_max_results,
            parallel_download=self.config.ddg_parallel_download,
            timeout=self.config.ddg_timeout,
            user_agent=self.config.user_agent,
            delay=self.config.delay,
        )
        
        if self.config.save_to_jsonl:
            self.config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        
        self._next_doc_id = self._compute_next_doc_id()

    def _compute_next_doc_id(self) -> int:
        """Find the next available doc_id from the output JSONL."""
        if not self.config.output_jsonl.exists():
            return 1

        max_doc_id = 0
        try:
            with self.config.output_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        doc_id = record.get("doc_id")
                        if isinstance(doc_id, int) and doc_id > max_doc_id:
                            max_doc_id = doc_id
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            logger.warning("Error computing next_doc_id: %s", exc)

        return max_doc_id + 1

    def should_trigger_web_search(
        self,
        local_results_count: int,
        avg_score: float,
    ) -> bool:
        """Determine if web search should be triggered based on thresholds."""
        if not self.config.enabled or not self.config.fallback_only:
            return False

        insufficient_docs = local_results_count < self.config.min_local_docs
        low_score = avg_score < self.config.min_score_threshold

        triggered = insufficient_docs and low_score
        if triggered:
            logger.info(
                "Web search triggered: docs=%d (min=%d), avg_score=%.2f (min=%.2f)",
                local_results_count,
                self.config.min_local_docs,
                avg_score,
                self.config.min_score_threshold,
            )
        return triggered

    def search_and_index(self, query: str) -> List[dict]:
        """Execute web search and optionally save to JSONL. Return records."""
        logger.info("Starting web search for: %s", query)

        # Fetch results from DuckDuckGo
        records = self.scraper.search(query)
        if not records:
            logger.info("No results from web search for: %s", query)
            return []

        # Assign doc_ids and add metadata
        indexed_records = []
        total = max(1, len(records))
        for idx, record in enumerate(records):
            record["doc_id"] = self._next_doc_id
            record["crawl_date"] = utc_now_iso()
            record["score_hint"] = round(1.0 - (idx / total), 4)
            self._next_doc_id += 1
            indexed_records.append(record)

        # Save to JSONL if configured
        if self.config.save_to_jsonl:
            self._save_records_to_jsonl(indexed_records)

        logger.info("Indexed %d web search records", len(indexed_records))
        return indexed_records

    def search_with_fallback(
        self,
        query: str,
        local_results_count: int,
        avg_score: float,
    ) -> List[dict]:
        """Return indexed web documents only when local information is insufficient."""
        if not self.should_trigger_web_search(local_results_count, avg_score):
            return []
        return self.search_and_index(query)

    def _save_records_to_jsonl(self, records: List[dict]) -> None:
        """Append records to the JSONL file."""
        try:
            with self.config.output_jsonl.open("a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Saved %d web search records to %s", len(records), self.config.output_jsonl)
        except Exception as exc:
            logger.error("Failed to save records to JSONL: %s", exc)


def create_pipeline(config: Optional[WebSearchConfig] = None) -> WebSearchPipeline:
    """Factory function to create a web search pipeline."""
    return WebSearchPipeline(config)
