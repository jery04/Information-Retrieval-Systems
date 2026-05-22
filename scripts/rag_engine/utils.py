"""Utility functions for RAG engine."""

import re
from typing import Dict, List, Set


def build_rag_prompt(query: str, documents: List[Dict[str, object]]) -> str:
    """
    Build a comprehensive prompt for Cerebras LLM.
    
    Args:
        query: User query
        documents: List of top_k documents with {doc_id, title, text, url, snippet}
    
    Returns:
        Formatted prompt string
    """
    if not documents:
        return f"User query: {query}\n\nNo documents available."
    
    # Build document context
    doc_context = "Documents Retrieved:\n"
    doc_context += "-" * 50 + "\n"
    
    def _clean_text(s: str) -> str:
        if not s:
            return ""
        # remove simple HTML tags and excessive whitespace
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    for i, doc in enumerate(documents, 1):
        title = (doc.get("title") or "(sin título)")
        url = doc.get("url") or ""
        # prefer explicit snippet, otherwise use full text
        raw_snip = doc.get("snippet") or doc.get("text") or ""
        snippet = _clean_text(str(raw_snip))[:1000]
        doc_id = doc.get("doc_id", "?")

        doc_context += f"\n[Document {i} - ID: {doc_id}]\n"
        doc_context += f"Title: {title}\n"
        if url:
            doc_context += f"URL: {url}\n"
        doc_context += f"Content: {snippet}\n"
        doc_context += "-" * 50 + "\n"
    
    # Build main prompt
    prompt = f"""You are a helpful AI assistant answering user queries using provided documents.

User Query: {query}

{doc_context}

Instructions:
1. Generate a comprehensive answer that integrates information from the provided documents
2. Cite implicitly by mentioning information from the documents (you don't need explicit citations)
3. If information is not in the documents, say so clearly
4. Provide a balanced, informative answer between 300-1200 characters
5. Write in the same language as the query

Output format requirement:
- On the very first line of your response, output a single JSON object with a boolean key "sufficient" indicating whether the provided documents are sufficient to answer the query (true or false). Example: {"sufficient": false}
- After that JSON line, provide a blank line, then the human-readable answer text.

Answer:"""
    
    return prompt


def extract_used_doc_ids(
    answer: str, 
    document_titles: List[str],
    document_ids: List[int]
) -> List[int]:
    """
    Attempt to extract which documents were likely used in the LLM response.
    
    This is a heuristic approach: checks if document titles or key terms appear in the answer.
    
    Args:
        answer: The generated answer from LLM
        document_titles: List of document titles in retrieval order
        document_ids: List of corresponding document IDs
    
    Returns:
        List of doc_ids that were likely used
    """
    if not answer or not document_titles:
        return []
    
    used_ids: Set[int] = set()
    answer_lower = answer.lower()
    
    for title, doc_id in zip(document_titles, document_ids):
        if not title:
            continue
        
        # Check if title words appear in answer
        title_words = [w.lower() for w in title.split() if len(w) > 3]
        
        # If at least 2 significant words from title appear in answer, mark as used
        matching_words = sum(1 for word in title_words if word in answer_lower)
        
        if matching_words >= min(2, len(title_words)):
            used_ids.add(doc_id)
    
    # If no documents detected via titles, assume the first one was used (fallback)
    if not used_ids and document_ids:
        used_ids.add(document_ids[0])
    
    return sorted(list(used_ids))


def truncate_answer(answer: str, max_chars: int = 1200) -> str:
    """
    Truncate answer to max_chars while preserving sentence boundaries.
    
    Args:
        answer: Generated answer
        max_chars: Maximum character count
    
    Returns:
        Truncated answer
    """
    if len(answer) <= max_chars:
        return answer
    
    # Try to cut at sentence boundary
    truncated = answer[:max_chars]
    
    # Find last sentence-ending punctuation
    for punct in ['. ', '! ', '? ']:
        last_idx = truncated.rfind(punct)
        if last_idx > max_chars * 0.8:  # At least 80% of max_chars
            return truncated[:last_idx + 1].strip()
    
    # If no good sentence boundary, cut at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:
        return truncated[:last_space].strip() + "..."
    
    return truncated.strip() + "..."
