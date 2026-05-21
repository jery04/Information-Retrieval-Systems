"""Fallback extractive RAG generator (improved SimpleRAGGenerator)."""

import re
from typing import Dict, List, Optional, Tuple


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part and part.strip()]


def _make_snippet(text: str, max_chars: int = 320) -> str:
    """Normalize whitespace and truncate text for snippets."""
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact[:max_chars].strip()


def _tokenize_simple(text: str) -> List[str]:
    """Simple word tokenization (no dependencies on indexer)."""
    if not text:
        return []
    # Convert to lowercase and split on word boundaries
    words = re.findall(r'\b\w+\b', text.lower())
    return words


def _score_sentence(query_terms: set, sentence: str) -> float:
    """Score sentence based on query term overlap (improved)."""
    if not sentence or not query_terms:
        return 0.0
    
    tokens = set(_tokenize_simple(sentence.lower()))
    if not tokens:
        return 0.0
    
    overlap = len(query_terms & tokens)
    if overlap == 0:
        return 0.0
    
    # Improved scoring: consider position and density
    # Penalize very short sentences, reward high overlap
    sentence_length = max(5, len(tokens))  # Minimum 5 to avoid penalizing short sentences too much
    overlap_ratio = overlap / sentence_length
    
    return overlap_ratio


class ImprovedRAGGenerator:
    """Improved extractive RAG generator with better heuristics."""
    
    def __init__(
        self,
        max_sentences: int = 6,
        max_chars: int = 1200,
        max_per_doc: int = 2,
        sentence_diversity: bool = True,
    ) -> None:
        """
        Initialize improved generator.
        
        Args:
            max_sentences: Maximum number of sentences to extract
            max_chars: Maximum characters in final answer
            max_per_doc: Maximum sentences per source document
            sentence_diversity: If True, try to select from different documents
        """
        self.max_sentences = max(1, int(max_sentences))
        self.max_chars = max(200, int(max_chars))
        self.max_per_doc = max(1, int(max_per_doc))
        self.sentence_diversity = sentence_diversity
    
    def generate(
        self,
        query: str,
        documents: List[Dict[str, object]],
        max_sentences: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> Tuple[str, List[int]]:
        """
        Generate an extractive answer from top-ranked documents.
        
        Args:
            query: User query
            documents: List of retrieved documents with {doc_id, score, title, url, text, snippet}
            max_sentences: Override max_sentences
            max_chars: Override max_chars
        
        Returns:
            (answer_text, doc_ids_used)
        """
        query_terms = set(_tokenize_simple(query.lower()))
        if not query_terms or not documents:
            return "", []
        
        sentence_budget = max_sentences if max_sentences is not None else self.max_sentences
        char_budget = max_chars if max_chars is not None else self.max_chars
        
        # Collect candidate sentences from all documents
        candidates: List[Tuple[float, int, str]] = []
        doc_sentence_count: Dict[int, int] = {}
        
        for doc in documents:
            doc_id = int(doc.get("doc_id", -1))
            text = str(doc.get("text") or "")
            if not text:
                continue
            
            sentences = _split_sentences(text)
            if not sentences:
                continue
            
            doc_score = float(doc.get("score", 0.0))
            doc_sentence_count[doc_id] = 0
            
            # Score all sentences in this document
            scored: List[Tuple[float, str]] = []
            for sentence in sentences:
                local_score = _score_sentence(query_terms, sentence)
                if local_score <= 0.0:
                    continue
                
                # Combine sentence relevance with document relevance
                combined = local_score * (1.0 + min(1.0, doc_score * 0.5))
                scored.append((combined, sentence))
            
            if not scored:
                continue
            
            # Sort by score and take top N from this doc
            scored.sort(key=lambda x: x[0], reverse=True)
            
            for combined, sentence in scored[: self.max_per_doc]:
                candidates.append((combined, doc_id, sentence))
        
        if not candidates:
            # Fallback: use first document's first paragraph
            if documents:
                first = documents[0]
                fallback = _make_snippet(
                    str(first.get("text") or ""), 
                    max_chars=char_budget
                )
                return fallback, [int(first.get("doc_id", -1))]
            return "", []
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Select final sentences with diversity and budget constraints
        chosen: List[str] = []
        used_docs: List[int] = []
        seen_sentences = set()
        current_len = 0
        selected_docs_count = 0
        
        for score, doc_id, sentence in candidates:
            # Skip duplicate sentences
            if sentence in seen_sentences:
                continue
            
            # Hard limits
            if len(chosen) >= sentence_budget:
                break
            
            added_len = len(sentence) + (1 if chosen else 0)
            if current_len + added_len > char_budget:
                break
            
            # Diversity: if enabled, don't overload one document
            if self.sentence_diversity:
                if doc_sentence_count.get(doc_id, 0) >= self.max_per_doc:
                    continue
            
            chosen.append(sentence)
            seen_sentences.add(sentence)
            current_len += added_len
            
            if doc_id not in used_docs:
                used_docs.append(doc_id)
                selected_docs_count += 1
            
            doc_sentence_count[doc_id] = doc_sentence_count.get(doc_id, 0) + 1
        
        answer = " ".join(chosen).strip()
        
        if not answer:
            # Final fallback: snippet from highest-scored document
            if documents:
                first = documents[0]
                fallback = _make_snippet(
                    str(first.get("text") or ""), 
                    max_chars=char_budget
                )
                return fallback, [int(first.get("doc_id", -1))]
            return "", []
        
        return answer, used_docs
