"""GVSM search pipeline: load indexes, retrieve candidates, and rank results.

This script wires dataset loading, Trie candidate retrieval, co-occurrence
cache handling, and GVSM tf-idf scoring into one integrated search engine.
It also includes a small main() routine to run a sample query end to end.
"""

import json                         # Provides JSON serialization and deserialization utilities
import os                           # Handles filesystem paths and directory operations
import sys
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple   # Type hints for structured and readable annotations
from engine import (                # Imports core IR components from the engine module
    CoOccurrenceIndex,              # Co-occurrence index for term-term statistics
    GeneralizedVectorSpaceModel,    # GVSM model for similarity with term correlations
)
from chroma_store import ChromaVectorStore
from indexer import Index, PatriciaTrie   # Local indexing utilities: tokenizer and PatriciaTrie structure
from rag import RAGPipeline
from web_search import WebSearchPipeline, WebSearchConfig  # Web search fallback module

# Optional lightweight HTTP API (no external deps required for CORS-safe responses)
try:
    from flask import Flask, request, jsonify
except Exception:
    Flask = None
# Resolve repository root (scripts/ is one level under repo root) and use absolute paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Default paths for data files (absolute, repo-root relative)
DEFAULT_DATASET_PATH = os.path.join(REPO_ROOT, "data", "extracted", "webpages", "webpages.jsonl")
DEFAULT_TRIE_PATH = os.path.join(REPO_ROOT, "data", "processed", "inverted_index_trie.json")
DEFAULT_COOC_PATH = os.path.join(REPO_ROOT, "data", "processed", "cooccurrence_index.json")

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
        # Fusion parameters for semantic+lexical scoring
        alpha: float = 0.6,
        title_weight: float = 2.0,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        chroma_weight: float = 0.75,
    ):
        """Initialize engine: load trie, documents, and co-occurrence index."""
        self.dataset_path = dataset_path
        self.trie_path = trie_path
        self.cooc_cache_path = cooc_cache_path
        self.min_cooc = min_cooc
        # fusion / BM25 params
        self.alpha = float(alpha)
        self.title_weight = float(title_weight)
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self.chroma_weight = float(chroma_weight)

        # load inverted trie from disk
        print(f"  [1/7] Loading PatriciaTrie from {trie_path}...")
        self.trie = PatriciaTrie(filepath=trie_path)
        self.trie.load()
        print(f"  [1/7] ✓ Trie loaded: {len(self.trie.root.children)} root nodes")

        # load tokenized documents and co-occurrence index (cache or build)
        print(f"  [2/7] Loading tokenized documents from {dataset_path}...")
        self.doc_tokens_by_id = self.load_tokenized_documents(dataset_path)
        print(f"  [2/7] ✓ Loaded {len(self.doc_tokens_by_id)} documents")
        
        # load full dataset records (used for API responses and snippets)
        print(f"  [3/7] Loading full document records...")
        self.records: Dict[int, Dict] = self.load_full_documents(dataset_path)
        print(f"  [3/7] ✓ Loaded {len(self.records)} full records")
        
        print(f"  [4/7] Loading/building co-occurrence index...")
        self.cooc = self._load_or_build_cooc(force_rebuild=force_rebuild_cooc)
        print(f"  [4/7] ✓ Co-occurrence index ready")

        # initialize GVSM model and compute idf
        print(f"  [5/7] Initializing GeneralizedVectorSpaceModel...")
        self.model = GeneralizedVectorSpaceModel(
            cooc_index=self.cooc,
            use_cosine=use_cosine,
        )
        
        print(f"  [6/7] Computing IDF scores...")
        self.model.compute_idf()
        print(f"  [6/7] ✓ IDF computed")

        # precompute document vectors for scoring
        print(f"  [7/7] Precomputing document vectors for {len(self.doc_tokens_by_id)} docs...")
        self.doc_vectors: Dict[int, Dict[str, float]] = {}
        for doc_id, doc_tokens in self.doc_tokens_by_id.items():
            doc_vector = self.model.get_document_vector(doc_tokens)
            if doc_vector:
                self.doc_vectors[doc_id] = doc_vector
        print(f"  [7/7] ✓ Precomputed {len(self.doc_vectors)} document vectors")
        # Build BM25 lexical index structures for lexical scoring
        print("  [8/8] Building BM25 lexical index...")
        self._build_bm25_index()
        print("  [8/8] ✓ BM25 index ready")
        print("  [9/9] Building Chroma vector index...")
        self.chroma_store = ChromaVectorStore(
            persist_directory=os.path.join(REPO_ROOT, "data", "processed", "chroma_db"),
            collection_name="webpages",
        )
        if self.chroma_store.enabled:
            chroma_count = self.chroma_store.upsert_documents(self.records)
            print(f"  [9/9] ✓ Chroma index ready ({chroma_count} documents upserted)")
        else:
            print("  [9/9] ⚠ ChromaDB no está disponible; se omite la capa vectorial")
        print("[init] ✓ GVSMSearchEngine fully initialized")

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

    def _record_has_lexical_match(self, record: Dict, query_terms: List[str]) -> bool:
        """Return True when the record contains at least one query term.

        This is intentionally lightweight: it checks title, body text and URL,
        which is enough to block unrelated Chroma-only candidates for short queries.
        """
        if not query_terms:
            return False

        title = (record.get("title") or "")
        text = (record.get("text") or "")
        url = (record.get("url") or "")

        title_terms = set(Index.tokenize(title))
        text_terms = set(Index.tokenize(text))
        url_low = url.lower()

        for term in set(query_terms):
            if term in title_terms or term in text_terms or term in url_low:
                return True
        return False

    def _build_bm25_index(self) -> None:
        """Precompute term frequencies, document lengths and IDF values for BM25.

        We treat title tokens with higher weight by keeping a separate title TF map
        and combining them at scoring time with `self.title_weight`.
        """
        # term -> document frequency (number of docs containing term)
        self.term_doc_freq: Dict[str, int] = {}
        # per-doc term frequencies for body and title
        self.doc_tf_body: Dict[int, Counter] = {}
        self.doc_tf_title: Dict[int, Counter] = {}
        # per-doc length (body only) and avgdl
        self.doc_len: Dict[int, int] = {}

        N = 0
        for doc_id, tokens in self.doc_tokens_by_id.items():
            N += 1
            tf_body = Counter(tokens)
            self.doc_tf_body[doc_id] = tf_body
            self.doc_len[doc_id] = sum(tf_body.values())

            # title tokens
            title = (self.records.get(doc_id, {}).get("title") or "")
            title_tokens = Index.tokenize(title)
            tf_title = Counter(title_tokens)
            self.doc_tf_title[doc_id] = tf_title

        # compute document frequencies
        df: Dict[str, int] = {}
        for doc_id in self.doc_tf_body:
            unique_terms = set(self.doc_tf_body[doc_id].keys()) | set(self.doc_tf_title[doc_id].keys())
            for t in unique_terms:
                df[t] = df.get(t, 0) + 1

        self.term_doc_freq = df
        self.N = max(1, N)
        # average document length (body only)
        total_len = sum(self.doc_len.values()) if self.doc_len else 0
        self.avgdl = (total_len / self.N) if self.N else 0.0

        # precompute idf with BM25's common formula
        self.idf: Dict[str, float] = {}
        for term, freq in self.term_doc_freq.items():
            # BM25 idf: log((N - n + 0.5) / (n + 0.5) + 1)
            n = freq
            self.idf[term] = math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def _bm25_score(self, query_terms: List[str], doc_id: int) -> float:
        """Compute BM25 score for `doc_id` given tokenized `query_terms`.

        Title tokens are weighted by `self.title_weight`.
        """
        if not query_terms:
            return 0.0

        tf_body = self.doc_tf_body.get(doc_id, Counter())
        tf_title = self.doc_tf_title.get(doc_id, Counter())
        dl = float(self.doc_len.get(doc_id, 0))

        score = 0.0
        for term in query_terms:
            # effective term frequency: body + title_weight * title
            f = tf_body.get(term, 0) + self.title_weight * tf_title.get(term, 0)
            if f <= 0:
                continue
            idf = self.idf.get(term, 0.0)
            denom = f + self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * (dl / (self.avgdl or 1.0)))
            score += idf * (f * (self.bm25_k1 + 1.0)) / denom

        return float(score)

    def rank_candidates(
        self,
        query_vec: Dict[str, float],
        candidate_doc_ids: List[int],
        query_terms: Optional[List[str]] = None,
        chroma_scores: Optional[Dict[int, float]] = None,
        top_k: int = 20,
    ) -> List[Tuple[int, float]]:
        """Re-rank candidate doc ids by fusing GVSM (semantic) and BM25 (lexical).

        Both scores are normalized across the candidate set to [0,1] before
        applying the linear combination with `self.alpha`.
        """
        if query_terms is None:
            query_terms = list(query_vec.keys()) if query_vec else []
        if chroma_scores is None:
            chroma_scores = {}

        # filter out doc ids that point to empty records (no title, no text, no url)
        filtered_candidates: List[int] = []
        for doc_id in candidate_doc_ids:
            rec = self.records.get(doc_id, {})
            title = (rec.get("title") or "").strip()
            text = (rec.get("text") or "").strip()
            url = (rec.get("url") or "").strip()
            if title or text or url:
                filtered_candidates.append(doc_id)

        candidate_doc_ids = filtered_candidates

        sem_scores: Dict[int, float] = {}
        lex_scores: Dict[int, float] = {}

        # compute raw scores
        for doc_id in candidate_doc_ids:
            # semantic score (GVSM)
            doc_vec = self.doc_vectors.get(doc_id)
            sem = 0.0
            if doc_vec and query_vec:
                sem = float(self.model.similarity(query_vec, doc_vec) or 0.0)
            sem_scores[doc_id] = sem

            # lexical score (BM25) - always computable if doc present in index
            lex = self._bm25_score(query_terms, doc_id)
            lex_scores[doc_id] = lex

        # normalization: map to [0,1] dividing by max (avoid division by zero)
        max_sem = max(sem_scores.values()) if sem_scores else 0.0
        max_lex = max(lex_scores.values()) if lex_scores else 0.0
        max_chroma = max(chroma_scores.values()) if chroma_scores else 0.0

        combined: List[Tuple[int, float]] = []
        for doc_id in candidate_doc_ids:
            sem_norm = (sem_scores.get(doc_id, 0.0) / max_sem) if max_sem > 0 else 0.0
            lex_norm = (lex_scores.get(doc_id, 0.0) / max_lex) if max_lex > 0 else 0.0
            base = self.alpha * sem_norm + (1.0 - self.alpha) * lex_norm
            chroma_norm = (chroma_scores.get(doc_id, 0.0) / max_chroma) if max_chroma > 0 else 0.0
            final = (1.0 - self.chroma_weight) * base + self.chroma_weight * chroma_norm
            combined.append((doc_id, float(final)))

        combined.sort(key=lambda pair: pair[1], reverse=True)
        return combined[:top_k]

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

        chroma_hits: List[Tuple[int, float]] = []
        if getattr(self, "chroma_store", None) and self.chroma_store.enabled:
            chroma_hits = self.chroma_store.search(query, top_k=max_candidates)
            if len(query_terms) == 1:
                chroma_hits = [
                    (doc_id, score)
                    for doc_id, score in chroma_hits
                    if self._record_has_lexical_match(self.records.get(doc_id, {}), query_terms)
                ]
        chroma_scores = {doc_id: score for doc_id, score in chroma_hits}

        # combine trie and Chroma candidates (union)
        if chroma_hits:
            candidate_doc_ids = list(dict.fromkeys(candidate_doc_ids + [doc_id for doc_id, _ in chroma_hits]))

        # if strict matching yields no candidates, relax requirement
        if not candidate_doc_ids and resolved_min_match > 1:
            candidate_doc_ids = self.trie.get_parcial_AND(
                query_terms,
                min_match=1,
                max_candidates=max_candidates,
            )

            if chroma_hits:
                candidate_doc_ids = list(dict.fromkeys(candidate_doc_ids + [doc_id for doc_id, _ in chroma_hits]))

        if not candidate_doc_ids:
            return chroma_hits[:top_k] if chroma_hits else []

        query_vec = self.model.get_query_vector(query)
        if not query_vec:
            # still allow BM25-only fallback: build query vec empty but compute lexical ranks
            # represent query_terms explicitly
            query_terms = Index.tokenize(query)
            # compute BM25-only ranking
            return self.rank_candidates({}, candidate_doc_ids, query_terms, chroma_scores, top_k)

        # pass tokenized query terms into ranking for BM25
        query_terms = Index.tokenize(query)
        return self.rank_candidates(query_vec, candidate_doc_ids, query_terms, chroma_scores, top_k)

    def ingest_records(self, records: List[dict]) -> int:
        """Ingest externally fetched records (e.g., from web search) into the engine.

        This updates `self.records`, token lists, GVSM vectors, and BM25 structures.
        It will also upsert to the Chroma store when enabled.
        Returns number of records ingested.
        """
        count = 0
        for rec in records:
            try:
                doc_id = int(rec.get("doc_id"))
            except Exception:
                continue

            # basic sanity checks
            title = (rec.get("title") or "").strip()
            text = (rec.get("text") or "").strip()
            url = (rec.get("url") or "").strip()
            if not (title or text or url):
                continue

            # store record
            self.records[doc_id] = rec

            # tokenize and add
            tokens = Index.tokenize(text)
            self.doc_tokens_by_id[doc_id] = tokens

            # compute and store document vector
            try:
                doc_vector = self.model.get_document_vector(tokens)
                if doc_vector:
                    self.doc_vectors[doc_id] = doc_vector
            except Exception:
                pass

            count += 1

        if count > 0:
            # recompute idf/bm25 structures conservatively
            try:
                self.model.compute_idf()
            except Exception:
                pass
            try:
                self._build_bm25_index()
            except Exception:
                pass

            # upsert into Chroma if available
            try:
                if getattr(self, "chroma_store", None) and self.chroma_store.enabled:
                    # prepare mapping of new records
                    rec_map = {int(r.get("doc_id")): r for r in records if r.get("doc_id")}
                    self.chroma_store.upsert_documents(rec_map)
            except Exception:
                pass

        return count


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

    print("[init] Creating Flask app")
    app = Flask(__name__)

    print("[init] Initializing RAGPipeline")
    rag = RAGPipeline(engine)

    # Initialize web search pipeline for fallback
    print("[init] Initializing WebSearchPipeline")
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

        print(f"[/search] Query: {q[:50]}... | top_k={top_k}")

        if not q:
            print("[/search] Empty query, returning empty results")
            return jsonify({"query": q, "total": 0, "results": []})

        # primary ranked results from GVSM
        print("[/search] Calling engine.search()")
        ranked = engine.search(query=q, top_k=top_k)
        print(f"[/search] ✓ Got {len(ranked)} results from GVSM")
        qlow = q.lower()

        # Candidate map: doc_id -> payload with score and stable tie-breakers
        candidates = {}

        for rank_pos, (doc_id, score) in enumerate(ranked):
            rec = engine.records.get(doc_id)
            if not rec:
                continue
            # skip documents that are essentially empty (no title, no text, no url)
            title = (rec.get("title") or "").strip()
            text = (rec.get("text") or "").strip()
            url = (rec.get("url") or "").strip()
            if not (title or text or url):
                continue

            candidates[doc_id] = {
                "record": rec,
                "score": float(score),
                "source_rank": rank_pos,
                "source_type": rec.get("source_type", "webpage"),
            }

        # lightweight substring boost: scan titles/text for query matches
        # lightweight substring boost: scan titles/text for query matches
        for doc_id, rec in engine.records.items():
            if doc_id in candidates:
                continue
            title = (rec.get("title") or "").lower()
            text = (rec.get("text") or "").lower()
            # skip empty records
            if not (title or text or (rec.get("url") or "")):
                continue
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

        print("[/search] Activating web_search_pipeline.search_with_fallback()")
        web_records = web_search_pipeline.search_with_fallback(
            query=q,
            local_results_count=local_count,
            avg_score=avg_score,
        )
        print(f"[/search] ✓ Web search returned {len(web_records) if web_records else 0} records")
        if web_records:
            web_search_used = True
            for rec in web_records:
                doc_id = rec.get("doc_id")
                if not doc_id:
                    continue
                # skip web results that are empty
                title = (rec.get("title") or "").strip()
                text = (rec.get("text") or "").strip()
                url = (rec.get("url") or "").strip()
                if not (title or text or url):
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

        print(f"[/rag] Query: {q[:50]}... | top_k={top_k} max_chars={max_chars}")

        if not q:
            print("[/rag] Empty query, returning empty response")
            return jsonify({"query": q, "answer": "", "sources": [], "contexts": [], "total_sources": 0})

        print("[/rag] Calling rag.answer()...")
        payload = rag.answer(
            query=q,
            top_k=top_k,
            max_sentences=max_sentences,
            max_chars=max_chars,
            web_search_pipeline=web_search_pipeline,
        )
        print(f"[/rag] ✓ Completed: answer_len={len(payload.get('answer') or '')} sources={len(payload.get('sources') or [])}")
        return jsonify(payload)

    return app


if __name__ == "__main__":
    # allow running as: python main.py serve  -> start HTTP API
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        print("=" * 70)
        print("Inicializando Sistema de Recuperación de Información")
        print("=" * 70)
        print()
        
        print("[startup] Creando GVSMSearchEngine...")
        try:
            engine = GVSMSearchEngine()
        except Exception as e:
            print(f"[ERROR] GVSMSearchEngine falló: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        print()
        print("[startup] Creando Flask app...")
        try:
            app = _make_api_app(engine)
        except Exception as e:
            print(f"[ERROR] Flask app falló: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        if app is None:
            print("[ERROR] Flask no está disponible. Instala: pip install flask")
            sys.exit(1)
        
        print()
        print("=" * 70)
        print("✓ Sistema inicializado correctamente")
        print("=" * 70)
        print()
        print("🚀 Iniciando servidor en http://127.0.0.1:5000")
        print("   Endpoint /search - para búsqueda GVSM")
        print("   Endpoint /rag    - para generación RAG")
        print()
        print("Presiona CTRL+C para salir")
        print()
        
        try:
            app.run(host="127.0.0.1", port=5000, debug=False)
        except KeyboardInterrupt:
            print("\n\n[shutdown] Servidor detenido por el usuario")
            sys.exit(0)
    else:
        print("ERROR! Uso: python3 scripts/main.py serve")
