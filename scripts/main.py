"""GVSM search pipeline: load indexes, retrieve candidates, and rank results.

This script wires dataset loading, Trie candidate retrieval, co-occurrence
cache handling, and GVSM tf-idf scoring into one integrated search engine.
It also includes a small main() routine to run a sample query end to end.
"""

import json                         # Provides JSON serialization and deserialization utilities
import os                           # Handles filesystem paths and directory operations
from typing import Dict, List, Optional, Tuple   # Type hints for structured and readable annotations
from engine import (                # Imports core IR components from the engine module
    CoOccurrenceIndex,              # Co-occurrence index for term-term statistics
    GeneralizedVectorSpaceModel,    # GVSM model for similarity with term correlations
)
from indexer import Index, PatriciaTrie   # Local indexing utilities: tokenizer and PatriciaTrie structure

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
        self.cooc = self._load_or_build_cooc(force_rebuild=force_rebuild_cooc)

        # initialize GVSM model and compute idf
        self.model = GeneralizedVectorSpaceModel(
            trie=self.trie,
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

    def build_cooccurrence_index(
        self,
        doc_tokens_by_id: Dict[int, List[str]],
        min_cooc: int = 3,
    ) -> CoOccurrenceIndex:
        """Build a CoOccurrenceIndex from a mapping of doc_id to tokens."""
        cooc_index = CoOccurrenceIndex(min_cooc=min_cooc)
        for doc_id, tokens in sorted(doc_tokens_by_id.items()):
            cooc_index.add_document(tokens, doc_id)
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


def main() -> None:
    """Small end-to-end test to validate the integration."""
    engine = GVSMSearchEngine()

    query = "Blog Technology"
    results = engine.search(query=query, top_k=10)

    print(f"Consulta: {query}")
    if not results:
        print("Sin resultados")
        return

    for doc_id, score in results:
        print(f"doc_id={doc_id} score={score:.6f}")


if __name__ == "__main__":
    main()
