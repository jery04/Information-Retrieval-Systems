"""Information retrieval core: co-occurrence indexing and GVSM scoring.

This module defines structures to track term document frequencies and
term-term co-occurrence counts, plus a Generalized Vector Space Model
for tf-idf vectorization and correlation-aware query-document similarity.
"""

import json                         # Handle JSON file operations (save/load index)
import os                           # File and directory path management
from collections import defaultdict # Dictionary with default values
from itertools import combinations  # Efficient pair generation for co-occurrence updates
from typing import Dict, List, Optional, Tuple   # Type hints for better code readability
import numpy as np                  # Numerical computing and linear algebra operations
from scipy.sparse import csr_matrix # Sparse matrix for memory-efficient co-occurrence storage
from indexer import Index           # Local tokenizer utility

class CoOccurrenceIndex:
    """Sparse co-occurrence index plus document frequency statistics."""

    def __init__(self, min_cooc: int = 3):
        """Initialize co-occurrence index with a minimum co-occurrence threshold."""
        self.min_cooc = min_cooc
        self.df: Dict[str, int] = {}
        self.total_docs: int = 0

        # term-index mappings for sparse matrix coordinates
        self.term_to_idx: Dict[str, int] = {}
        self.idx_to_term: List[str] = []

        # upper-triangle co-occurrence matrix (i < j) in CSR format
        self.cooc: csr_matrix = csr_matrix((0, 0), dtype=np.int32)

        # mutable builder map used while ingesting docs; released after matrix materialization
        self._pair_counts: Optional[Dict[Tuple[int, int], int]] = defaultdict(int)
        self._matrix_dirty: bool = False

    def _get_or_create_term_idx(self, term: str) -> int:
        """Map a term to its sparse-matrix index, creating it if needed."""
        idx = self.term_to_idx.get(term)
        if idx is not None:
            return idx

        idx = len(self.idx_to_term)
        self.term_to_idx[term] = idx
        self.idx_to_term.append(term)
        # matrix shape must be refreshed when vocabulary grows
        self._matrix_dirty = True
        return idx

    def _ensure_pair_counts(self) -> Dict[Tuple[int, int], int]:
        """Ensure mutable pair counts exist, rebuilding from CSR if necessary."""
        if self._pair_counts is None:
            rebuilt: Dict[Tuple[int, int], int] = defaultdict(int)
            if self.cooc.nnz > 0:
                coo = self.cooc.tocoo()
                for row, col, value in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
                    rebuilt[(int(row), int(col))] = int(value)
            self._pair_counts = rebuilt
        return self._pair_counts

    def _ensure_sparse_matrix(self) -> None:
        """Materialize pair counts into CSR matrix and release builder storage."""
        vocab_size = len(self.idx_to_term)
        if not self._matrix_dirty and self.cooc.shape == (vocab_size, vocab_size):
            return

        counts = self._pair_counts or {}
        if vocab_size == 0:
            self.cooc = csr_matrix((0, 0), dtype=np.int32)
        elif not counts:
            self.cooc = csr_matrix((vocab_size, vocab_size), dtype=np.int32)
        else:
            pair_count = len(counts)
            rows = np.fromiter((pair[0] for pair in counts), dtype=np.int32, count=pair_count)
            cols = np.fromiter((pair[1] for pair in counts), dtype=np.int32, count=pair_count)
            data = np.fromiter((int(value) for value in counts.values()), dtype=np.int32, count=pair_count)
            self.cooc = csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size), dtype=np.int32)

        self._matrix_dirty = False
        # keep only compact CSR representation until new documents are added
        self._pair_counts = None

    def _get_pair_count(self, term1: str, term2: str) -> int:
        """Fetch co-occurrence count for a pair from the sparse matrix."""
        idx1 = self.term_to_idx.get(term1)
        idx2 = self.term_to_idx.get(term2)
        if idx1 is None or idx2 is None:
            return 0

        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        self._ensure_sparse_matrix()
        return int(self.cooc[idx1, idx2])

    def add_document(self, tokens: List[str]) -> None:
        """Update document frequencies and co-occurrences from a document's tokens."""
        # prepare a sorted list of unique tokens from the document
        unique_tokens = sorted(set(tokens))
        if not unique_tokens:
            # count empty documents but do not update df/cooc
            self.total_docs += 1
            return

        # update document frequency for each unique term
        term_indices: List[int] = []
        for term in unique_tokens:
            self.df[term] = self.df.get(term, 0) + 1
            term_indices.append(self._get_or_create_term_idx(term))

        # increment the total document counter
        self.total_docs += 1

        # update pairwise co-occurrence counts (upper triangle only)
        if len(term_indices) >= 2:
            term_indices.sort()
            pair_counts = self._ensure_pair_counts()
            for term_a_idx, term_b_idx in combinations(term_indices, 2):
                key = (term_a_idx, term_b_idx)
                pair_counts[key] += 1
            self._matrix_dirty = True

    def get_correlation(self, term1: str, term2: str, method: str = "cosine") -> float:
        """Return correlation between two terms, applying the min_cooc threshold."""
        # identical terms are perfectly correlated
        if term1 == term2:
            return 1.0

        # get co-occurrence count and enforce threshold
        count = self._get_pair_count(term1, term2)
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
        self._ensure_sparse_matrix()
        coo = self.cooc.tocoo()

        # compact sparse representation of the index state
        return {
            "min_cooc": self.min_cooc,
            "df": self.df,
            "total_docs": self.total_docs,
            "terms": self.idx_to_term,
            "cooc_rows": coo.row.tolist(),
            "cooc_cols": coo.col.tolist(),
            "cooc_data": [int(value) for value in coo.data.tolist()],
            "cooc_shape": [int(coo.shape[0]), int(coo.shape[1])],
            "cooc_format": "csr_upper_triangle_v2",
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

        # reset sparse structures
        self.term_to_idx = {}
        self.idx_to_term = []
        self.cooc = csr_matrix((0, 0), dtype=np.int32)
        self._pair_counts = defaultdict(int)
        self._matrix_dirty = False

        # first try sparse v2 format
        raw_terms = data.get("terms", [])
        raw_rows = data.get("cooc_rows", [])
        raw_cols = data.get("cooc_cols", [])
        raw_data = data.get("cooc_data", [])
        raw_shape = data.get("cooc_shape", [])

        has_sparse_payload = (
            "cooc_rows" in data
            and "cooc_cols" in data
            and "cooc_data" in data
        )
        if (
            has_sparse_payload
            and isinstance(raw_terms, list)
            and isinstance(raw_rows, list)
            and isinstance(raw_cols, list)
            and isinstance(raw_data, list)
            and len(raw_rows) == len(raw_cols) == len(raw_data)
        ):
            self.idx_to_term = [str(term) for term in raw_terms]
            self.term_to_idx = {term: idx for idx, term in enumerate(self.idx_to_term)}

            # include any df term missing from serialized term list
            for term in sorted(self.df.keys()):
                if term not in self.term_to_idx:
                    self.term_to_idx[term] = len(self.idx_to_term)
                    self.idx_to_term.append(term)

            try:
                rows = np.asarray([int(value) for value in raw_rows], dtype=np.int32)
                cols = np.asarray([int(value) for value in raw_cols], dtype=np.int32)
                values = np.asarray([int(value) for value in raw_data], dtype=np.int32)

                if isinstance(raw_shape, (list, tuple)) and len(raw_shape) == 2:
                    shape_0 = int(raw_shape[0])
                    shape_1 = int(raw_shape[1])
                else:
                    shape_0 = len(self.idx_to_term)
                    shape_1 = len(self.idx_to_term)

                target_size = len(self.idx_to_term)
                shape_0 = max(shape_0, target_size)
                shape_1 = max(shape_1, target_size)

                self.cooc = csr_matrix((values, (rows, cols)), shape=(shape_0, shape_1), dtype=np.int32)
                self._pair_counts = None
                self._matrix_dirty = False
                return
            except (TypeError, ValueError):
                # fall back to legacy nested-dict format
                pass

        # fallback: load legacy nested co-occurrence mapping and convert to sparse
        raw_cooc = data.get("cooc", {})
        all_terms = set(self.df.keys())

        if isinstance(raw_cooc, dict):
            for term_a, pairs in raw_cooc.items():
                all_terms.add(str(term_a))
                if isinstance(pairs, dict):
                    for term_b in pairs.keys():
                        all_terms.add(str(term_b))

        self.idx_to_term = sorted(all_terms)
        self.term_to_idx = {term: idx for idx, term in enumerate(self.idx_to_term)}

        pair_counts = defaultdict(int)
        if isinstance(raw_cooc, dict):
            for term_a, pairs in raw_cooc.items():
                if not isinstance(pairs, dict):
                    continue
                idx_a = self.term_to_idx.get(str(term_a))
                if idx_a is None:
                    continue

                for term_b, raw_count in pairs.items():
                    idx_b = self.term_to_idx.get(str(term_b))
                    if idx_b is None or idx_a == idx_b:
                        continue
                    try:
                        count = int(raw_count)
                    except (TypeError, ValueError):
                        continue
                    if count <= 0:
                        continue

                    row_idx, col_idx = idx_a, idx_b
                    if row_idx > col_idx:
                        row_idx, col_idx = col_idx, row_idx
                    pair_counts[(row_idx, col_idx)] += count

        self._pair_counts = pair_counts
        self._matrix_dirty = True
        self._ensure_sparse_matrix()

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
    """Generalized Vector Space Model backed by CoOccurrenceIndex statistics."""

    def __init__(
        self,
        cooc_index: "CoOccurrenceIndex",
        use_cosine: bool = True,
    ):
        """Initialize GVSM with co-occurrence index and metric choice."""
        self.cooc = cooc_index
        self.use_cosine = use_cosine
        self.idf: Dict[str, float] = {}
        self.total_docs: int = cooc_index.total_docs
        self._corr_matrix: Optional[csr_matrix] = None
        self._corr_method: Optional[str] = None
        self._corr_min_cooc: Optional[int] = None
        # idf values computed by compute_idf()

    def _resolve_correlation_method(self) -> str:
        """Return active correlation method name based on model settings."""
        return "cosine" if self.use_cosine else "jaccard"

    def _ensure_correlation_matrix(self) -> csr_matrix:
        """Build and cache sparse term-term correlation matrix used by GVSM scoring."""
        method = self._resolve_correlation_method()
        if (
            self._corr_matrix is not None
            and self._corr_method == method
            and self._corr_min_cooc == self.cooc.min_cooc
        ):
            return self._corr_matrix

        self.cooc._ensure_sparse_matrix()
        cooc_matrix = self.cooc.cooc
        vocab_size = int(cooc_matrix.shape[0])

        if vocab_size == 0:
            self._corr_matrix = csr_matrix((0, 0), dtype=np.float64)
            self._corr_method = method
            self._corr_min_cooc = self.cooc.min_cooc
            return self._corr_matrix

        df_values = np.zeros(vocab_size, dtype=np.float64)
        for term, idx in self.cooc.term_to_idx.items():
            if 0 <= idx < vocab_size:
                df_values[idx] = float(self.cooc.df.get(term, 0))

        coo = cooc_matrix.tocoo()
        rows = coo.row.astype(np.int32)
        cols = coo.col.astype(np.int32)
        counts = coo.data.astype(np.float64)

        # upper-triangle counts are thresholded and mapped to correlations.
        if counts.size > 0:
            mask = (counts >= float(self.cooc.min_cooc)) & (rows != cols)
            rows = rows[mask]
            cols = cols[mask]
            counts = counts[mask]
        if counts.size == 0:
            corr_matrix = csr_matrix((vocab_size, vocab_size), dtype=np.float64)
            corr_matrix.setdiag(1.0)
            self._corr_matrix = corr_matrix
            self._corr_method = method
            self._corr_min_cooc = self.cooc.min_cooc
            return self._corr_matrix

        if method == "cosine":
            denom = np.sqrt(df_values[rows] * df_values[cols])
            corr_values = np.divide(
                counts,
                denom,
                out=np.zeros_like(counts, dtype=np.float64),
                where=denom > 0,
            )
        elif method == "jaccard":
            union = df_values[rows] + df_values[cols] - counts
            corr_values = np.divide(
                counts,
                union,
                out=np.zeros_like(counts, dtype=np.float64),
                where=union > 0,
            )
        else:
            denom = df_values[rows] + df_values[cols]
            corr_values = np.divide(
                2.0 * counts,
                denom,
                out=np.zeros_like(counts, dtype=np.float64),
                where=denom > 0,
            )

        valid = corr_values > 0
        rows = rows[valid]
        cols = cols[valid]
        corr_values = corr_values[valid]

        sym_rows = np.concatenate((rows, cols))
        sym_cols = np.concatenate((cols, rows))
        sym_data = np.concatenate((corr_values, corr_values))

        corr_matrix = csr_matrix(
            (sym_data, (sym_rows, sym_cols)),
            shape=(vocab_size, vocab_size),
            dtype=np.float64,
        )
        corr_matrix.setdiag(1.0)
        corr_matrix.eliminate_zeros()
        corr_matrix.sum_duplicates()

        self._corr_matrix = corr_matrix
        self._corr_method = method
        self._corr_min_cooc = self.cooc.min_cooc
        return self._corr_matrix

    def _vector_to_sparse_arrays(self, vector: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Map a sparse term->weight dict to aligned index/weight arrays."""
        indices: List[int] = []
        weights: List[float] = []

        for term, weight in vector.items():
            if weight == 0.0:
                continue
            idx = self.cooc.term_to_idx.get(term)
            if idx is None:
                continue
            indices.append(idx)
            weights.append(float(weight))

        if not indices:
            return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float64)

        idx_array = np.asarray(indices, dtype=np.int32)
        weight_array = np.asarray(weights, dtype=np.float64)
        order = np.argsort(idx_array)
        return idx_array[order], weight_array[order]

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

    def similarity(self, query_vec: Dict[str, float], doc_vec: Dict[str, float]) -> float:
        """
        Compute GVSM similarity:
        sum_i sum_j (w_iq * w_jd * corr(ti, tj)) / (||q|| * ||d||)
        """
        if not query_vec or not doc_vec:
            return 0.0

        q_indices, q_weights = self._vector_to_sparse_arrays(query_vec)
        d_indices, d_weights = self._vector_to_sparse_arrays(doc_vec)
        if q_indices.size == 0 or d_indices.size == 0:
            return 0.0

        corr_matrix = self._ensure_correlation_matrix()
        if corr_matrix.shape == (0, 0):
            return 0.0

        # score = q^T * A_sub * d, where A_sub selects rows(Q) and cols(D).
        corr_sub = corr_matrix[q_indices][:, d_indices]
        if corr_sub.nnz == 0:
            return 0.0

        transformed_doc = corr_sub.dot(d_weights)
        score = float(np.dot(q_weights, transformed_doc))
        if score == 0.0:
            return 0.0

        norm_q = float(np.linalg.norm(q_weights))
        norm_d = float(np.linalg.norm(d_weights))
        if norm_q == 0.0 or norm_d == 0.0:
            return 0.0

        return score / (norm_q * norm_d)
    