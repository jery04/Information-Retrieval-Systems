"""RAG Engine module - Retrieval-Augmented Generation with Cerebras LLM."""

from .config import RAGConfig
from .generator import CerebrasRAGGenerator
from .fallback_generator import ImprovedRAGGenerator
from .cerebras_client import CerebrasClient
from .utils import build_rag_prompt, extract_used_doc_ids, truncate_answer

__all__ = [
    "RAGConfig",
    "CerebrasRAGGenerator",
    "ImprovedRAGGenerator",
    "CerebrasClient",
    "build_rag_prompt",
    "extract_used_doc_ids",
    "truncate_answer",
]
