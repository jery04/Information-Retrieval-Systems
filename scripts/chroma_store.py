"""Lightweight ChromaDB wrapper for vector retrieval.

This module keeps the integration offline-friendly by generating deterministic
embeddings locally with a hashing trick. ChromaDB is used as the persistent
vector store; the embedding model can be swapped later without changing the
search pipeline.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from indexer import Index

# Disable Chroma anonymized telemetry to avoid noisy warnings in the console.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")


def _disable_posthog_telemetry() -> None:
    """Make PostHog telemetry a no-op when available.

    Some Chroma/PostHog combinations emit warnings because the telemetry client
    tries to call a mismatched `capture()` implementation. Turning the call
    into a no-op avoids the warning without affecting search functionality.
    """
    try:
        import posthog  # type: ignore
    except Exception:  # pragma: no cover - optional dependency surface
        return

    try:
        posthog.disabled = True
        posthog.capture = lambda *args, **kwargs: None  # type: ignore[assignment]
    except Exception:
        return

try:
    import chromadb
    try:
        from chromadb.config import Settings
    except Exception:  # pragma: no cover - optional dependency surface
        Settings = None
except Exception:  # pragma: no cover - optional dependency
    chromadb = None
    Settings = None


_disable_posthog_telemetry()


class ChromaVectorStore:
    """Persistent ChromaDB-backed vector store."""

    def __init__(
        self,
        persist_directory: str = "/home/alex/Documentos/Information-Retrieval-Systems/data",
        collection_name: str = "webpages",
        dimension: int = 384,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.dimension = int(dimension)
        self.enabled = chromadb is not None
        self.client = None
        self.collection = None

        if not self.enabled:
            return

        os.makedirs(self.persist_directory, exist_ok=True)
        client_kwargs = {"path": self.persist_directory}
        if Settings is not None:
            client_kwargs["settings"] = Settings(anonymized_telemetry=False)

        self.client = chromadb.PersistentClient(**client_kwargs)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_text(self, text: str) -> List[float]:
        """Generate a deterministic dense vector using a hashing trick."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = Index.tokenize(text or "")
        if not tokens:
            return vec.tolist()

        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little", signed=False) % self.dimension
            sign = 1.0 if (digest[4] & 1) else -1.0
            vec[idx] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.tolist()

    def _normalize_record_text(self, record: Dict) -> Tuple[str, Dict]:
        title = (record.get("title") or "").strip()
        text = (record.get("text") or "").strip()
        url = (record.get("url") or "").strip()
        combined = "\n".join(part for part in (title, text) if part)
        metadata = {
            "title": title,
            "url": url,
            "source_type": record.get("source_type") or "webpage",
            "doc_id": record.get("doc_id"),
        }
        return combined, metadata

    def upsert_documents(self, records: Dict[int, Dict]) -> int:
        """Insert or update documents into the Chroma collection."""
        if not self.enabled or self.collection is None:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []
        embeddings: List[List[float]] = []

        for doc_id, record in records.items():
            combined_text, metadata = self._normalize_record_text(record)
            if not combined_text.strip() and not metadata.get("url"):
                continue

            ids.append(str(doc_id))
            documents.append(combined_text[:40000])
            metadatas.append(metadata)
            embeddings.append(self._embed_text(combined_text or metadata.get("url") or ""))

        if ids:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        return len(ids)

    def count(self) -> int:
        if not self.enabled or self.collection is None:
            return 0
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Return `(doc_id, score)` pairs ordered by vector similarity."""
        if not self.enabled or self.collection is None:
            return []

        query_text = (query or "").strip()
        if not query_text:
            return []

        embedding = self._embed_text(query_text)
        if not any(embedding):
            return []

        try:
            result = self.collection.query(
                query_embeddings=[embedding],
                n_results=max(1, int(top_k)),
                include=["distances"],
            )
        except Exception:
            return []

        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        ranked: List[Tuple[int, float]] = []
        for raw_id, distance in zip(ids, distances):
            try:
                doc_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            score = 1.0 - float(distance or 0.0)
            score = max(0.0, min(1.0, score))
            ranked.append((doc_id, score))

        return ranked