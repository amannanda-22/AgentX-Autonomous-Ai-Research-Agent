"""
MemoryManager — Singleton wrapper around ChromaDB for persistent vector storage.
Uses OpenAI embeddings when the key is available; falls back to ChromaDB's
default sentence-transformers embedding function.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "agentx_research"


def _get_embedding_function():
    """Return the best available embedding function."""
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key != "your_openai_api_key_here":
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

            return OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small",
            )
        except Exception as exc:
            logger.warning("OpenAI embeddings unavailable: %s. Using default.", exc)

    # ChromaDB default — sentence-transformers all-MiniLM-L6-v2
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()


class MemoryManager:
    """
    Thread-safe singleton that manages one ChromaDB persistent collection.
    All read/write operations go through this class.
    """

    _instance: Optional["MemoryManager"] = None

    def __new__(cls) -> "MemoryManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
        os.makedirs(persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self._ef = _get_embedding_function()

        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialised. Collection '%s' has %d documents.",
            _COLLECTION_NAME,
            self._collection.count(),
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert a document into the collection."""
        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Semantic similarity search."""
        count = self._collection.count()
        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        # ChromaDB errors if n_results > collection size
        n = min(n_results, count)
        kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        return self._collection.query(**kwargs)

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by its ID."""
        result = self._collection.get(ids=[doc_id], include=["documents", "metadatas"])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        if docs and docs[0]:
            return {"document": docs[0], "metadata": metas[0] if metas else {}}
        return None

    def list_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return metadata for the N most recent documents, sorted by timestamp."""
        result = self._collection.get(include=["metadatas", "ids"])
        ids = result.get("ids", [])
        metas = result.get("metadatas", [])

        combined = list(zip(ids, metas))
        # Sort by timestamp descending
        combined.sort(
            key=lambda x: x[1].get("timestamp", ""),
            reverse=True,
        )
        return [
            {
                "task_id": id_,
                "topic": m.get("topic", "Unknown"),
                "timestamp": m.get("timestamp", ""),
                "word_count": m.get("word_count", 0),
            }
            for id_, m in combined[:n]
        ]

    def delete(self, doc_id: str) -> None:
        """Remove a document from the collection."""
        self._collection.delete(ids=[doc_id])

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (use with caution)."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("ChromaDB collection '%s' reset.", _COLLECTION_NAME)
