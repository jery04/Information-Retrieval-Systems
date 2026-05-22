"""Utility functions for RAG engine."""

import json
import re
from typing import Dict, List, Set, Tuple


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
        return f"Consulta del usuario: {query}\n\nNo hay documentos disponibles."

    # Build document context with a total budget so the prompt does not grow
    # unbounded when many long documents are retrieved.
    doc_context = "Documentos recuperados:\n"
    doc_context += "-" * 50 + "\n"
    total_budget = 2400
    overhead_budget = 600
    usable_budget = max(800, total_budget - overhead_budget)
    per_doc_budget = max(160, usable_budget // max(1, len(documents)))
    
    def _clean_text(s: str) -> str:
        if not s:
            return ""
        # remove simple HTML tags and excessive whitespace
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _truncate_to_budget(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        clipped = text[:max_chars]
        last_space = clipped.rfind(" ")
        if last_space > max_chars * 0.7:
            clipped = clipped[:last_space]
        return clipped.strip()

    for i, doc in enumerate(documents, 1):
        title = (doc.get("title") or "(sin título)")
        url = doc.get("url") or ""
        # prefer explicit snippet, otherwise use full text
        raw_snip = doc.get("snippet") or doc.get("text") or ""
        snippet = _truncate_to_budget(_clean_text(str(raw_snip)), per_doc_budget)
        doc_id = doc.get("doc_id", "?")

        doc_context += f"\n[Document {i} - ID: {doc_id}]\n"
        doc_context += f"Title: {title}\n"
        if url:
            doc_context += f"URL: {url}\n"
        doc_context += f"Content: {snippet}\n"
        doc_context += "-" * 50 + "\n"
    
    # Build main prompt in Spanish to reduce instruction-language mixing.
    prompt = f"""Eres un asistente útil que responde usando únicamente los documentos proporcionados.

Consulta del usuario: {query}

{doc_context}

Instrucciones:
1. Responde de forma clara y directa usando la información de los documentos.
2. Si los documentos no alcanzan para responder bien, dilo explícitamente.
3. No inventes datos ni rellenes con contenido no respaldado por los documentos.
4. Mantén la respuesta breve y útil, idealmente entre 300 y 1200 caracteres.
5. Responde en el mismo idioma de la consulta.

Formato de salida obligatorio:
- En la primera línea de tu respuesta, escribe un único objeto JSON con la clave booleana "sufficient" para indicar si los documentos alcanzan para responder la consulta (true o false). Ejemplo: {{"sufficient": false}}
- Después de esa línea JSON, deja una línea en blanco y luego escribe la respuesta en texto natural.

Respuesta:"""
    
    return prompt


def parse_sufficient_flag(answer: str) -> Tuple[bool, str]:
    """Parse a leading JSON sufficiency marker from the model output.

    Returns a tuple of (sufficient, cleaned_answer). If the marker is missing
    or malformed, defaults to sufficient=True and returns the original answer.
    """
    if not answer:
        return True, ""

    lines = answer.splitlines()
    if not lines:
        return True, answer

    first_line = lines[0].strip()
    if first_line.startswith("{") and first_line.endswith("}"):
        try:
            parsed = json.loads(first_line)
            if isinstance(parsed, dict) and "sufficient" in parsed:
                sufficient = bool(parsed.get("sufficient"))
                cleaned = "\n".join(lines[1:]).lstrip()
                return sufficient, cleaned
        except Exception:
            pass

    return True, answer


def extract_used_doc_ids(
    answer: str, 
    document_titles: List[str],
    document_ids: List[int]
) -> List[int]:
    """
    Attempt to extract which documents were likely used in the LLM response.
    
    This is a conservative heuristic: it only returns documents with a clear
    title overlap in the answer. If nothing matches, it returns an empty list
    instead of guessing the first document.
    
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
    
    # Do not guess: if nothing matches, return an empty list.
    # The caller can treat this as "source attribution unavailable".
    
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
