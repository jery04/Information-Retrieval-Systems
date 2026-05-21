"""Cerebras-based RAG generator (generative approach)."""

import logging
from typing import Dict, List, Optional, Tuple

from .config import RAGConfig
from .cerebras_client import CerebrasClient
from .fallback_generator import ImprovedRAGGenerator
from .utils import build_rag_prompt, extract_used_doc_ids, truncate_answer

logger = logging.getLogger(__name__)


class CerebrasRAGGenerator:
    """Generative RAG using Cerebras LLM with fallback to extractive."""
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        max_chars: int = 1200,
        enable_fallback: bool = True,
    ):
        """
        Initialize Cerebras RAG generator.
        
        Args:
            config: RAGConfig instance (creates new if None)
            max_chars: Maximum characters in response
            enable_fallback: If True, use ImprovedRAGGenerator as fallback
        """
        self.config = config or RAGConfig()
        self.max_chars = max(200, int(max_chars))
        self.enable_fallback = enable_fallback
        
        # Initialize Cerebras client
        try:
            self.cerebras_client = CerebrasClient(self.config)
            logger.info("CerebrasRAGGenerator initialized with LLM backend")
        except Exception as e:
            logger.error(f"Failed to initialize Cerebras client: {e}")
            self.cerebras_client = None
            if not enable_fallback:
                raise
            logger.warning("Falling back to extractive generation")
        
        # Initialize fallback generator
        if self.enable_fallback:
            self.fallback_generator = ImprovedRAGGenerator(
                max_sentences=6,
                max_chars=max_chars,
                max_per_doc=2,
                sentence_diversity=True,
            )
        else:
            self.fallback_generator = None
    
    def generate(
        self,
        query: str,
        documents: List[Dict[str, object]],
        max_sentences: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> Tuple[str, List[int]]:
        """
        Generate an enriched answer using Cerebras LLM.
        
        Falls back to extractive generation if LLM fails.
        
        Args:
            query: User query
            documents: List of retrieved documents
            max_sentences: Ignored (LLM decides structure)
            max_chars: Override max_chars for truncation
        
        Returns:
            (answer_text, doc_ids_used)
        """
        char_budget = max_chars if max_chars is not None else self.max_chars
        
        # Handle empty inputs
        if not query or not documents:
            return "", []
        
        # Try LLM generation
        if self.cerebras_client:
            logger.info(f"Generating answer via Cerebras for query: {query[:50]}...")
            answer, used_ids = self._generate_with_llm(query, documents, char_budget)
            
            if answer:
                return answer, used_ids
            
            # LLM failed, try fallback
            logger.info("LLM generation failed, attempting fallback...")
        
        # Fallback to extractive generation
        if self.fallback_generator:
            logger.info("Using improved extractive RAG as fallback")
            answer, used_ids = self.fallback_generator.generate(
                query=query,
                documents=documents,
                max_sentences=6,
                max_chars=char_budget,
            )
            return answer, used_ids
        
        # No fallback available
        logger.error("No generation method available")
        return "", []
    
    def _generate_with_llm(
        self,
        query: str,
        documents: List[Dict[str, object]],
        max_chars: int,
    ) -> Tuple[str, List[int]]:
        """
        Generate answer using Cerebras LLM.
        
        Args:
            query: User query
            documents: Retrieved documents
            max_chars: Maximum characters
        
        Returns:
            (answer, doc_ids_used) or ("", []) if failed
        """
        try:
            # Build prompt
            prompt = build_rag_prompt(query, documents)
            
            # Generate with Cerebras
            answer, success = self.cerebras_client.generate(prompt)
            
            if not success or not answer:
                logger.warning("Cerebras generation was not successful")
                return "", []
            
            # Truncate to budget
            answer = truncate_answer(answer, max_chars)
            
            # Extract which documents were used
            doc_titles = [str(doc.get("title", "")) for doc in documents]
            doc_ids = [int(doc.get("doc_id", -1)) for doc in documents]
            
            used_ids = extract_used_doc_ids(answer, doc_titles, doc_ids)
            
            logger.info(f"Generated answer using {len(used_ids)} sources")
            return answer, used_ids
        
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}", exc_info=True)
            return "", []
