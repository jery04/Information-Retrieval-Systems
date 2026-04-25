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
        ranked_map = {doc_id: score for doc_id, score in ranked}

        # lightweight substring boost: scan titles/text for query matches
        additional = []
        qlow = q.lower()
        for doc_id, rec in engine.records.items():
            if doc_id in ranked_map:
                continue
            title = (rec.get("title") or "").lower()
            text = (rec.get("text") or "").lower()
            if qlow in title or qlow in text:
                additional.append(doc_id)

        combined_ids = list(ranked_map.keys()) + additional

        results = []
        for doc_id in combined_ids:
            rec = engine.records.get(doc_id)
            if not rec:
                continue

            score = float(ranked_map.get(doc_id, 0.0))
            file_type = _guess_file_type(rec)
            snippet = (rec.get("text") or "").replace("\n", " ")[:320].strip()

            results.append(
                {
                    "doc_id": int(doc_id),
                    "title": rec.get("title") or "(sin título)",
                    "snippet": snippet,
                    "url": rec.get("url") or "",
                    "file_type": file_type,
                    "crawl_date": rec.get("crawl_date"),
                    "domain": rec.get("domain"),
                    "score": score,
                }
            )

        return jsonify({"query": q, "total": len(results), "results": results})

    @app.route("/set_model", methods=["POST"])
    def api_set_model():
        """Endpoint to change spaCy model used by Index and refresh engine document vectors.

        Expects JSON body: { "model": "en_core_web_sm" } or { "model": "es_core_news_sm" }
        """
        data = request.get_json(silent=True) or {}
        model = data.get("model")
        if not model:
            return jsonify({"ok": False, "error": "missing model"}), 400
        try:
            # update tokenizer/model used by Index
            Index.set_model(model)

            # refresh tokenized documents and recompute document vectors for the engine
            try:
                engine.doc_tokens_by_id = engine.load_tokenized_documents(engine.dataset_path)
                engine.model = GeneralizedVectorSpaceModel(
                    cooc_index=engine.cooc,
                    use_cosine=getattr(engine.model, "use_cosine", True),
                )
                engine.model.compute_idf()
                engine.doc_vectors = {}
                for doc_id, doc_tokens in engine.doc_tokens_by_id.items():
                    doc_vector = engine.model.get_document_vector(doc_tokens)
                    if doc_vector:
                        engine.doc_vectors[doc_id] = doc_vector
            except Exception:
                # non-fatal: setting the model succeeded but re-indexing failed
                pass

            return jsonify({"ok": True, "model": model})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

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
