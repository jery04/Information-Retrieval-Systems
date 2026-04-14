import json                         # Handle JSON file operations (save/load index)
import os                           # File and directory path management
from collections import defaultdict # Dictionary with default values
from typing import Dict, List, Optional, Tuple   # Type hints for better code readability
import numpy as np                  # Numerical computing and linear algebra operations
from indexer import Index, PatriciaTrie   # Local project modules (Index and PatriciaTrie structures)

DEFAULT_DATASET_PATH = os.path.join("data", "extracted", "webpages", "webpages.jsonl")
DEFAULT_TRIE_PATH = os.path.join("data", "processed", "inverted_index_trie.json")
DEFAULT_COOC_PATH = os.path.join("data", "processed", "cooccurrence_index.json")

class CoOccurrenceIndex:
    """Structure for term-term co-occurrences plus document frequency."""

    def __init__(self, min_cooc: int = 3):
        """Initialize co-occurrence index with a minimum co-occurrence threshold."""
        self.min_cooc = min_cooc
        self.cooc: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.df: Dict[str, int] = {}
        self.total_docs: int = 0
        # storage: nested co-occurrence counts, document frequencies, and document counter

    def add_document(self, tokens: List[str], _doc_id: int) -> None:
        """Update document frequencies and co-occurrences from a document's tokens."""
        # prepare a sorted list of unique tokens from the document
        unique_tokens = sorted(set(tokens))
        if not unique_tokens:
            # count empty documents but do not update df/cooc
            self.total_docs += 1
            return

        # update document frequency for each unique term
        for term in unique_tokens:
            self.df[term] = self.df.get(term, 0) + 1

        # increment the total document counter
        self.total_docs += 1

        # update pairwise co-occurrence counts (upper triangle only)
        n_tokens = len(unique_tokens)
        for i in range(n_tokens):
            term_a = unique_tokens[i]
            row = self.cooc[term_a]
            for j in range(i + 1, n_tokens):
                term_b = unique_tokens[j]
                row[term_b] = row.get(term_b, 0) + 1

    def get_correlation(self, term1: str, term2: str, method: str = "cosine") -> float:
        """Return correlation between two terms, applying the min_cooc threshold."""
        # identical terms are perfectly correlated
        if term1 == term2:
            return 1.0

        # normalize ordering since cooc stores pairs with term_a < term_b
        if term1 > term2:
            term1, term2 = term2, term1

        # get co-occurrence count and enforce threshold
        count = self.cooc.get(term1, {}).get(term2, 0)
        if count < self.min_cooc:
            return 0.0

        # require non-zero document frequencies
        df1 = self.df.get(term1, 0)
        df2 = self.df.get(term2, 0)
        if df1 == 0 or df2 == 0:
            return 0.0

        # compute similarity according to chosen method
        if method == "cosine":
            return count / (df1 ** 0.5 * df2 ** 0.5)
        if method == "jaccard":
            union = df1 + df2 - count
            return count / union if union > 0 else 0.0
        if method == "dice":
            return (2 * count) / (df1 + df2)
        return 0.0

    def to_dict(self) -> Dict[str, object]:
        """Serialize the index to a dictionary suitable for JSON saving."""
        # compact representation of the index state
        return {
            "min_cooc": self.min_cooc,
            "df": self.df,
            "total_docs": self.total_docs,
            "cooc": self.cooc,
        }
    
    def _from_dict(self, data: Dict[str, object]) -> None:
        """Load index fields from a dictionary into this instance."""
        # read min_cooc with a safe fallback
        self.min_cooc = int(data.get("min_cooc", self.min_cooc))
        # load document frequencies
        raw_df = data.get("df", {})
        if isinstance(raw_df, dict):
            self.df = {str(term): int(df) for term, df in raw_df.items()}
        else:
            self.df = {}

        # load total document count
        raw_total_docs = data.get("total_docs", 0)
        self.total_docs = int(raw_total_docs) if isinstance(raw_total_docs, (int, float)) else 0

        # load nested co-occurrence mapping safely
        raw_cooc = data.get("cooc", {})
        if isinstance(raw_cooc, dict):
            self.cooc = defaultdict(
                dict,
                {
                    str(term_a): {str(term_b): int(cnt) for term_b, cnt in pairs.items()}
                    for term_a, pairs in raw_cooc.items()
                    if isinstance(pairs, dict)
                },
            )
        else:
            self.cooc = defaultdict(dict)

    def save(self, filepath: str) -> None:
        """Save the co-occurrence index to a JSON file."""
        # ensure target directory exists then write compact JSON
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, separators=("," , ":"))
    
    def _load_from_file(self, filepath: str, min_cooc: int = 3) -> None:
        """Load index data from a JSON file; leave empty if file is missing."""
        # if file missing, preserve defaults but set min_cooc
        if not os.path.exists(filepath):
            self.min_cooc = min_cooc
            return

        # read and populate fields
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        self._from_dict(data)
        self.min_cooc = min_cooc

class GeneralizedVectorSpaceModel:
    """Generalized Vector Space Model backed by PatriciaTrie and CoOccurrenceIndex."""

    def __init__(
        self,
        trie: "PatriciaTrie",
        cooc_index: "CoOccurrenceIndex",
        use_cosine: bool = True,
    ):
        """Initialize GVSM with trie, co-occurrence index, and metric choice."""
        self.trie = trie
        self.cooc = cooc_index
        self.use_cosine = use_cosine
        self.idf: Dict[str, float] = {}
        self.total_docs: int = cooc_index.total_docs
        # idf values computed by compute_idf()

    def compute_idf(self) -> None:
        """Compute IDF for all known terms from the co-occurrence index."""
        # refresh document count and reset idf cache
        self.total_docs = self.cooc.total_docs
        self.idf.clear()

        if self.total_docs == 0:
            # no documents to compute idf from
            return

        # smoothed idf calculation
        for term, df in self.cooc.df.items():
            self.idf[term] = np.log((self.total_docs + 1) / (df + 1)) + 1

    def _get_term_weight(self, term: str, tf: float) -> float:
        """Compute tf-idf weight for a term given normalized tf."""
        # multiply normalized tf by idf (0 if term unknown)
        return tf * self.idf.get(term, 0.0)

    def get_query_vector(self, query: str) -> Dict[str, float]:
        """Convert a query string into a sparse term->weight vector."""
        tokens = Index.tokenize(query)
        if not tokens:
            return {}
        
        # compute term frequencies and find max for normalization
        term_freq: Dict[str, int] = defaultdict(int)
        max_tf = 1
        for token in tokens:
            term_freq[token] += 1
            if term_freq[token] > max_tf:
                max_tf = term_freq[token]

        # build weighted vector using normalized tf and idf
        query_vector: Dict[str, float] = {}
        for term, tf in term_freq.items():
            if term not in self.cooc.df:
                # skip unknown terms
                continue
            normalized_tf = tf / max_tf
            weight = self._get_term_weight(term, normalized_tf)
            if weight > 0:
                query_vector[term] = weight
        return query_vector

    def get_document_vector(self, doc_tokens: List[str]) -> Dict[str, float]:
        """Convert document tokens into a sparse term->weight vector."""
        if not doc_tokens:
            return {}
        
        # compute term frequencies and normalization factor
        term_freq: Dict[str, int] = defaultdict(int)
        max_tf = 1
        for token in doc_tokens:
            term_freq[token] += 1
            if term_freq[token] > max_tf:
                max_tf = term_freq[token]

        # build weighted document vector
        doc_vector: Dict[str, float] = {}
        for term, tf in term_freq.items():
            if term not in self.cooc.df:
                continue
            normalized_tf = tf / max_tf
            weight = self._get_term_weight(term, normalized_tf)
            if weight > 0:
                doc_vector[term] = weight
        return doc_vector

    def _get_correlation(self, term1: str, term2: str) -> float:
        """Get term-term correlation from the CoOccurrenceIndex."""
        if term1 == term2:
            return 1.0
        method = "cosine" if self.use_cosine else "jaccard"
        return self.cooc.get_correlation(term1, term2, method=method)

    def similarity(self, query_vec: Dict[str, float], doc_vec: Dict[str, float]) -> float:
        """
        Compute GVSM similarity:
        sum_i sum_j (w_iq * w_jd * corr(ti, tj)) / (||q|| * ||d||)
        """
        if not query_vec or not doc_vec:
            return 0.0

        # accumulate cross-term weighted correlations
        score = 0.0
        for q_term, q_weight in query_vec.items():
            for d_term, d_weight in doc_vec.items():
                corr = self._get_correlation(q_term, d_term)
                if corr == 0.0:
                    continue
                score += q_weight * d_weight * corr

        # compute squared norms
        norm_q = sum(weight * weight for weight in query_vec.values())
        norm_d = sum(weight * weight for weight in doc_vec.values())
        if norm_q == 0.0 or norm_d == 0.0:
            return 0.0

        # normalize by product of norms
        return score / np.sqrt(norm_q * norm_d)

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

    query = "Entrada de blog sobre nuevas herramientas"
    results = engine.search(query=query, top_k=10)

    print(f"Consulta: {query}")
    if not results:
        print("Sin resultados")
        return

    for doc_id, score in results:
        print(f"doc_id={doc_id} score={score:.6f}")

if __name__ == "__main__":
    main()