"""
RAG Pipeline - Retrieval-Augmented Generation.

Main public interface that maintains backward compatibility with main.py.
Internally uses the modular rag_engine for Cerebras LLM generation with extractive fallback.
"""

import logging
from typing import Dict, List, Optional

from rag_engine import CerebrasRAGGenerator, RAGConfig

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Retrieval + generation wrapper for Cerebras-powered RAG.
    
    Maintains the exact same interface as the original implementation for backward
    compatibility with main.py, while internally using the new rag_engine module.
    """

    def __init__(self, retriever, generator: Optional[CerebrasRAGGenerator] = None) -> None:
        """
        Initialize RAG Pipeline.
        
        Args:
            retriever: Search engine (e.g., GVSMSearchEngine)
            generator: Optional CerebrasRAGGenerator instance (creates new if None)
        """
        self.retriever = retriever
        
        # Initialize generator with Cerebras + fallback
        if generator is None:
            try:
                config = RAGConfig()
                self.generator = CerebrasRAGGenerator(
                    config=config,
                    max_chars=1200,
                    enable_fallback=True,  # Always keep fallback for robustness
                )
                logger.info("Initialized RAGPipeline with Cerebras LLM backend")
            except Exception as e:
                logger.error(f"Failed to initialize Cerebras generator: {e}")
                logger.warning("RAG Pipeline may not work properly without Cerebras API")
                raise
        else:
            self.generator = generator

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        """
        Retrieve top documents and normalize fields for generation.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
        
        Returns:
            List of documents with normalized fields
        """
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
                    "snippet": self._make_snippet(text),
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
        """
        Retrieve documents and generate an enriched answer using Cerebras LLM.
        
        This is the main public interface - maintains exact compatibility with original RAG.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            max_sentences: Ignored (LLM decides structure)
            max_chars: Maximum characters in response
        
        Returns:
            Dict with query, answer, sources, contexts, total_sources
        """
        documents = self.retrieve(query=query, top_k=top_k)
        
        if not documents:
            return {
                "query": query,
                "answer": "",
                "sources": [],
                "contexts": [],
                "total_sources": 0,
            }

        # Generate answer with Cerebras LLM (or fallback)
        answer, used_doc_ids = self.generator.generate(
            query=query,
            documents=documents,
            max_sentences=max_sentences,
            max_chars=max_chars,
        )

        used_set = set(used_doc_ids)
        
        # Build sources list (documents actually used in answer)
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
        
        # Build contexts list (all retrieved documents)
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

    @staticmethod
    def _make_snippet(text: str, max_chars: int = 320) -> str:
        """Normalize whitespace and truncate text for snippets."""
        if not text:
            return ""
        compact = " ".join(text.split())
        return compact[:max_chars].strip()

