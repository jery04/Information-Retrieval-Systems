"""GVSM search pipeline: load indexes, retrieve candidates, and rank results.

This script wires dataset loading, Trie candidate retrieval, co-occurrence
cache handling, and GVSM tf-idf scoring into one integrated search engine.
It also includes a small main() routine to run a sample query end to end.
"""

import json                         # Provides JSON serialization and deserialization utilities
import os                           # Handles filesystem paths and directory operations
import sys
from typing import Dict, List, Optional, Tuple   # Type hints for structured and readable annotations
from engine import (                # Imports core IR components from the engine module
    CoOccurrenceIndex,              # Co-occurrence index for term-term statistics
    GeneralizedVectorSpaceModel,    # GVSM model for similarity with term correlations
)
from indexer import Index, PatriciaTrie   # Local indexing utilities: tokenizer and PatriciaTrie structure
from rag import RAGPipeline
from web_search import WebSearchPipeline, WebSearchConfig  # Web search fallback module

# Optional lightweight HTTP API (no external deps required for CORS-safe responses)
try:
    from flask import Flask, request, jsonify
except Exception:
    Flask = None

# Default paths for data files
DEFAULT_DATASET_PATH = os.path.join("data", "extracted", "webpages", "webpages.jsonl")
DEFAULT_TRIE_PATH = os.path.join("data", "processed", "inverted_index_trie.json")
DEFAULT_COOC_PATH = os.path.join("data", "processed", "cooccurrence_index.json")

class GVSMSearchEngine:
    """Full pipeline: Trie candidate retrieval plus GVSM re-ranking."""

    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        trie_path: str = DEFAULT_TRIE_PATH,
        cooc_cache_path: str = DEFAULT_COOC_PATH,
        min_cooc: int = 3,
        use_cosine: bool = True,
        force_rebuild_cooc: bool = False,
    ):
        """Initialize engine: load trie, documents, and co-occurrence index."""
        self.dataset_path = dataset_path
        self.trie_path = trie_path
        self.cooc_cache_path = cooc_cache_path
        self.min_cooc = min_cooc

        # load inverted trie from disk
        self.trie = PatriciaTrie(filepath=trie_path)
        self.trie.load()

        # load tokenized documents and co-occurrence index (cache or build)
        self.doc_tokens_by_id = self.load_tokenized_documents(dataset_path)
        # load full dataset records (used for API responses and snippets)
        self.records: Dict[int, Dict] = self.load_full_documents(dataset_path)
        self.cooc = self._load_or_build_cooc(force_rebuild=force_rebuild_cooc)

        # initialize GVSM model and compute idf
        self.model = GeneralizedVectorSpaceModel(
            cooc_index=self.cooc,
            use_cosine=use_cosine,
        )
        self.model.compute_idf()

        # precompute document vectors for scoring
        self.doc_vectors: Dict[int, Dict[str, float]] = {}
        for doc_id, doc_tokens in self.doc_tokens_by_id.items():
            doc_vector = self.model.get_document_vector(doc_tokens)
            if doc_vector:
                self.doc_vectors[doc_id] = doc_vector

    def load_tokenized_documents(self, dataset_path: str = DEFAULT_DATASET_PATH) -> Dict[int, List[str]]:
        """Load doc_id->tokens using the same tokenization as the Index."""
        documents: Dict[int, List[str]] = {}
        if not os.path.exists(dataset_path):
            return documents

        # read JSONL dataset and tokenize each document's text
        with open(dataset_path, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue

                try:
                    doc_id = int(record.get("doc_id"))
                except (TypeError, ValueError):
                    # skip records without a valid id
                    continue

                text = record.get("text", "")
                if not isinstance(text, str):
                    text = str(text)

                documents[doc_id] = Index.tokenize(text)

        return documents

    def load_full_documents(self, dataset_path: str = DEFAULT_DATASET_PATH) -> Dict[int, Dict]:
        """Load full JSONL records into a mapping doc_id -> record."""
        records: Dict[int, Dict] = {}
        if not os.path.exists(dataset_path):
            return records

        with open(dataset_path, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    doc_id = int(record.get("doc_id"))
                except (TypeError, ValueError):
                    continue
                records[doc_id] = record

        return records

    def build_cooccurrence_index(
        self,
        doc_tokens_by_id: Dict[int, List[str]],
        min_cooc: int = 3,
    ) -> CoOccurrenceIndex:
        """Build a CoOccurrenceIndex from a mapping of doc_id to tokens."""
        cooc_index = CoOccurrenceIndex(min_cooc=min_cooc)
        for doc_id, tokens in sorted(doc_tokens_by_id.items()):
            cooc_index.add_document(tokens)
        return cooc_index

    def _load_or_build_cooc(self, force_rebuild: bool = False) -> CoOccurrenceIndex:
        """Load cached co-occurrence index if compatible; otherwise rebuild and save."""
        if not force_rebuild:
            cached = CoOccurrenceIndex(min_cooc=self.min_cooc)
            cached._load_from_file(self.cooc_cache_path, min_cooc=self.min_cooc)
            if cached.total_docs == len(self.doc_tokens_by_id):
                return cached

        # build fresh index and persist it
        built = self.build_cooccurrence_index(self.doc_tokens_by_id, min_cooc=self.min_cooc)
        built.save(self.cooc_cache_path)
        return built

    def _resolve_min_match(self, query_terms: List[str], min_match: Optional[int]) -> int:
        """Adjust min_match to avoid queries with no candidates due to configuration."""
        unique_count = len(set(query_terms))
        if unique_count == 0:
            return 1

        if min_match is None:
            return 1 if unique_count == 1 else min(2, unique_count)

        return max(1, min(min_match, unique_count))

    def rank_candidates(
        self,
        query_vec: Dict[str, float],
        candidate_doc_ids: List[int],
        top_k: int = 20,
    ) -> List[Tuple[int, float]]:
        """Re-rank candidate doc ids using GVSM and return top_k results."""
        results: List[Tuple[int, float]] = []
        for doc_id in candidate_doc_ids:
            doc_vec = self.doc_vectors.get(doc_id)
            if not doc_vec:
                continue

            # score each candidate and collect positive scores
            score = self.model.similarity(query_vec, doc_vec)
            if score > 0.0:
                results.append((doc_id, score))

        results.sort(key=lambda pair: pair[1], reverse=True)
        return results[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 20,
        min_match: Optional[int] = None,
        max_candidates: int = 3000,
    ) -> List[Tuple[int, float]]:
        """Run search: retrieve candidates via Trie then rerank using GVSM."""
        query_terms = Index.tokenize(query)
        if not query_terms:
            return []

        resolved_min_match = self._resolve_min_match(query_terms, min_match)
        candidate_doc_ids = self.trie.get_parcial_AND(
            query_terms,
            min_match=resolved_min_match,
            max_candidates=max_candidates,
        )

        # if strict matching yields no candidates, relax requirement
        if not candidate_doc_ids and resolved_min_match > 1:
            candidate_doc_ids = self.trie.get_parcial_AND(
                query_terms,
                min_match=1,
                max_candidates=max_candidates,
            )

        if not candidate_doc_ids:
            return []

        query_vec = self.model.get_query_vector(query)
        if not query_vec:
            return []
        return self.rank_candidates(query_vec, candidate_doc_ids, top_k)


def _guess_file_type(record: Dict) -> str:
    """Heuristically determine a display file type for a record.

    Returns one of: 'PDF', 'IMAGEN', 'VIDEO', 'DOCUMENTO', 'OTROS'.
    """
    url = (record.get("url") or "").lower()
    stype = (record.get("source_type") or "").lower()

    if stype and "pdf" in stype:
        return "PDF"
    if url.endswith(".pdf") or ".pdf?" in url:
        return "PDF"
    for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"):
        if url.endswith(ext):
            return "IMAGEN"
    for ext in (".mp4", ".webm", ".mov", "youtube.com/watch"):
        if ext in url:
            return "VIDEO"
    # fallback for generic webpages
    return "DOCUMENTO"

def _make_api_app(engine: GVSMSearchEngine):
    """Create a small Flask app exposing a /search endpoint that returns JSON results."""
    if Flask is None:
        return None

    app = Flask(__name__)
    rag = RAGPipeline(engine)
    
    # Initialize web search pipeline for fallback
    web_search_config = WebSearchConfig()
    web_search_pipeline = WebSearchPipeline(web_search_config)

    @app.after_request
    def _add_cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.route("/search")
    def api_search():
        q = request.args.get("query", "").strip()
        try:
            top_k = int(request.args.get("top_k", "40"))
        except ValueError:
            top_k = 40

        if not q:
            return jsonify({"query": q, "total": 0, "results": []})

        # primary ranked results from GVSM
        ranked = engine.search(query=q, top_k=top_k)
        qlow = q.lower()

        # Candidate map: doc_id -> payload with score and stable tie-breakers
        candidates = {}

        for rank_pos, (doc_id, score) in enumerate(ranked):
            rec = engine.records.get(doc_id)
            if not rec:
                continue
            candidates[doc_id] = {
                "record": rec,
                "score": float(score),
                "source_rank": rank_pos,
                "source_type": rec.get("source_type", "webpage"),
            }

        # lightweight substring boost: scan titles/text for query matches
        for doc_id, rec in engine.records.items():
            if doc_id in candidates:
                continue
            title = (rec.get("title") or "").lower()
            text = (rec.get("text") or "").lower()
            if qlow in title or qlow in text:
                candidates[doc_id] = {
                    "record": rec,
                    "score": 0.15,
                    "source_rank": 10_000,
                    "source_type": rec.get("source_type", "webpage"),
                }

        # Check if web search should be triggered
        avg_score = sum(score for _, score in ranked) / len(ranked) if ranked else 0.0
        local_count = len(ranked)
        web_search_used = False

        web_records = web_search_pipeline.search_with_fallback(
            query=q,
            local_results_count=local_count,
            avg_score=avg_score,
        )
        if web_records:
            web_search_used = True
            for rec in web_records:
                doc_id = rec.get("doc_id")
                if not doc_id:
                    continue
                engine.records[doc_id] = rec
                web_score = float(rec.get("score_hint", 0.5))
                source_rank = int(rec.get("web_rank", 10_000))
                if doc_id in candidates:
                    candidates[doc_id]["score"] = max(candidates[doc_id]["score"], web_score)
                    candidates[doc_id]["record"] = rec
                    candidates[doc_id]["source_rank"] = min(candidates[doc_id]["source_rank"], source_rank)
                    candidates[doc_id]["source_type"] = rec.get("source_type", "web_search")
                else:
                    candidates[doc_id] = {
                        "record": rec,
                        "score": web_score,
                        "source_rank": source_rank,
                        "source_type": rec.get("source_type", "web_search"),
                    }

        ordered_candidates = sorted(
            candidates.items(),
            key=lambda item: (
                -float(item[1].get("score", 0.0)),
                int(item[1].get("source_rank", 10_000)),
                str(item[1]["record"].get("title") or "").lower(),
            ),
        )

        results = []
        for doc_id, payload in ordered_candidates[:top_k]:
            rec = payload["record"]
            score = float(payload.get("score", 0.0))
            file_type = _guess_file_type(rec)
            snippet = (rec.get("text") or "").replace("\n", " ")[:320].strip()
            source_type = payload.get("source_type", rec.get("source_type", "webpage"))

            results.append(
                {
                    "doc_id": int(doc_id) if isinstance(doc_id, int) else doc_id,
                    "title": rec.get("title") or "(sin título)",
                    "snippet": snippet,
                    "url": rec.get("url") or "",
                    "file_type": file_type,
                    "crawl_date": rec.get("crawl_date"),
                    "domain": rec.get("domain"),
                    "score": score,
                    "source_type": source_type,
                }
            )

        return jsonify({
            "query": q,
            "total": len(results),
            "results": results,
            "web_search_used": web_search_used,
            "local_results": local_count,
            "avg_local_score": avg_score,
        })

    @app.route("/rag")
    def api_rag():
        q = request.args.get("query", "").strip()
        try:
            top_k = int(request.args.get("top_k", "5"))
        except ValueError:
            top_k = 5
        try:
            max_sentences = int(request.args.get("max_sentences", "6"))
        except ValueError:
            max_sentences = 6
        try:
            max_chars = int(request.args.get("max_chars", "1200"))
        except ValueError:
            max_chars = 1200

        if not q:
            return jsonify({"query": q, "answer": "", "sources": [], "contexts": [], "total_sources": 0})

        payload = rag.answer(
            query=q,
            top_k=top_k,
            max_sentences=max_sentences,
            max_chars=max_chars,
        )
        return jsonify(payload)

    return app


if __name__ == "__main__":
    # allow running as: python main.py serve  -> start HTTP API
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        print("Inicializando motor y servidor API...")
        engine = GVSMSearchEngine()
        app = _make_api_app(engine)
        if app is None:
            print("Flask no está disponible. Instala Flask para usar la API (pip install flask)")
            sys.exit(1)
        app.run(host="127.0.0.1", port=5000, debug=False)
    else:
        print("ERROR!")
