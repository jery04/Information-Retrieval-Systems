"""Simple RAG pipeline that enriches answers with retrieved document content."""

import re                      # Provides regular expressions for text pattern matching
from typing import Dict, List, Optional, Tuple   # Type hints for dictionaries, lists, optionals, and tuples
from indexer import Index      # Imports the Index class for tokenizing and indexing documents

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using a lightweight regex."""
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part and part.strip()]

def _make_snippet(text: str, max_chars: int = 320) -> str:
    """Normalize whitespace and truncate text for short previews."""
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact[:max_chars].strip()

def _score_sentence(query_terms: set, sentence: str) -> float:
    """Score a sentence based on query-term overlap."""
    if not sentence or not query_terms:
        return 0.0
    tokens = Index.tokenize(sentence)
    if not tokens:
        return 0.0
    overlap = len(set(tokens) & query_terms)
    if overlap == 0:
        return 0.0
    return overlap / (len(tokens) + 3)

class SimpleRAGGenerator:
    """Extractive generator that selects relevant sentences from sources."""

    def __init__(
        self,
        max_sentences: int = 6,
        max_chars: int = 1200,
        max_per_doc: int = 2,
    ) -> None:
        self.max_sentences = max(1, int(max_sentences))
        self.max_chars = max(200, int(max_chars))
        self.max_per_doc = max(1, int(max_per_doc))

    def generate(
        self,
        query: str,
        documents: List[Dict[str, object]],
        max_sentences: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> Tuple[str, List[int]]:
        """Generate an answer using top sentences from retrieved documents."""
        query_terms = set(Index.tokenize(query))
        if not query_terms or not documents:
            return "", []

        sentence_budget = max_sentences if max_sentences is not None else self.max_sentences
        char_budget = max_chars if max_chars is not None else self.max_chars

        candidates: List[Tuple[float, int, str]] = []
        for doc in documents:
            doc_id = int(doc.get("doc_id", -1))
            text = str(doc.get("text") or "")
            if not text:
                continue

            sentences = _split_sentences(text)
            if not sentences:
                continue

            doc_score = float(doc.get("score", 0.0))
            scored: List[Tuple[float, str]] = []
            for sentence in sentences:
                local_score = _score_sentence(query_terms, sentence)
                if local_score <= 0.0:
                    continue
                combined = local_score * (1.0 + min(1.0, doc_score))
                scored.append((combined, sentence))

            if not scored:
                continue

            scored.sort(key=lambda pair: pair[0], reverse=True)
            for combined, sentence in scored[: self.max_per_doc]:
                candidates.append((combined, doc_id, sentence))

        if not candidates:
            first = documents[0]
            fallback = _make_snippet(str(first.get("text") or ""), max_chars=char_budget)
            return fallback, [int(first.get("doc_id", -1))]

        candidates.sort(key=lambda item: item[0], reverse=True)
        chosen: List[str] = []
        used_docs: List[int] = []
        seen_sentences = set()
        current_len = 0

        for _, doc_id, sentence in candidates:
            if sentence in seen_sentences:
                continue
            if len(chosen) >= sentence_budget:
                break
            added_len = len(sentence) + (1 if chosen else 0)
            if current_len + added_len > char_budget:
                break
            chosen.append(sentence)
            seen_sentences.add(sentence)
            current_len += added_len
            if doc_id not in used_docs:
                used_docs.append(doc_id)

        answer = " ".join(chosen).strip()
        if not answer:
            first = documents[0]
            fallback = _make_snippet(str(first.get("text") or ""), max_chars=char_budget)
            return fallback, [int(first.get("doc_id", -1))]

        return answer, used_docs

class RAGPipeline:
    """Retrieval + generation wrapper for simple RAG."""

    def __init__(self, retriever, generator: Optional[SimpleRAGGenerator] = None) -> None:
        self.retriever = retriever
        self.generator = generator or SimpleRAGGenerator()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        """Retrieve top documents and normalize fields for generation."""
        ranked = self.retriever.search(query=query, top_k=top_k)
        documents: List[Dict[str, object]] = []
        for doc_id, score in ranked:
            record = self.retriever.records.get(doc_id)
            if not record:
                continue
            text = str(record.get("text") or "")
            documents.append(
                {
                    "doc_id": int(doc_id),
                    "score": float(score),
                    "title": record.get("title") or "(sin titulo)",
                    "url": record.get("url") or "",
                    "text": text,
                    "snippet": _make_snippet(text),
                }
            )
        return documents

    def answer(
        self,
        query: str,
        top_k: int = 5,
        max_sentences: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> Dict[str, object]:
        """Retrieve documents and generate an enriched answer."""
        documents = self.retrieve(query=query, top_k=top_k)
        if not documents:
            return {
                "query": query,
                "answer": "",
                "sources": [],
                "contexts": [],
                "total_sources": 0,
            }

        answer, used_doc_ids = self.generator.generate(
            query=query,
            documents=documents,
            max_sentences=max_sentences,
            max_chars=max_chars,
        )

        used_set = set(used_doc_ids)
        sources = [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "url": doc["url"],
                "score": doc["score"],
            }
            for doc in documents
            if doc["doc_id"] in used_set
        ]
        contexts = [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "snippet": doc["snippet"],
                "url": doc["url"],
            }
            for doc in documents
        ]

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "total_sources": len(sources),
        }
