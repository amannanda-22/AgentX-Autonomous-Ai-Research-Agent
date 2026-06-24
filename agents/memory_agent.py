"""
Memory Agent — ChromaDB-backed long-term vector memory.
Stores completed research sessions and retrieves semantically relevant
past context to enrich future research on similar topics.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryAgent:
    """
    Wraps the MemoryManager to provide:
    - store(): persist a completed research session
    - retrieve(): semantic search for relevant past sessions
    - list_recent(): get N most recent sessions
    """

    def __init__(self) -> None:
        # Lazy import to avoid startup errors if ChromaDB is misconfigured
        from memory.memory_manager import MemoryManager

        self._manager = MemoryManager()

    def store(
        self,
        task_id: str,
        topic: str,
        report: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Persist a research report to ChromaDB vector memory."""
        doc_metadata = {
            "task_id": task_id,
            "topic": topic,
            "timestamp": datetime.utcnow().isoformat(),
            "word_count": len(report.split()),
            **(metadata or {}),
        }
        try:
            self._manager.add_document(
                doc_id=task_id,
                text=f"Topic: {topic}\n\n{report[:4000]}",
                metadata=doc_metadata,
            )
            logger.info("Memory stored for task %s (topic: %s)", task_id, topic)
            return True
        except Exception as exc:
            logger.error("MemoryAgent.store failed: %s", exc)
            return False

    def retrieve(
        self, query: str, n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Semantic search: find past research sessions relevant to `query`.
        Returns list of dicts with keys: topic, snippet, task_id, timestamp.
        """
        try:
            raw = self._manager.query(query, n_results=n_results)
            results = []
            for doc, meta in zip(
                raw.get("documents", [[]])[0],
                raw.get("metadatas", [[]])[0],
            ):
                results.append(
                    {
                        "topic": meta.get("topic", "Unknown"),
                        "task_id": meta.get("task_id", ""),
                        "timestamp": meta.get("timestamp", ""),
                        "snippet": doc[:300] + "..." if len(doc) > 300 else doc,
                    }
                )
            return results
        except Exception as exc:
            logger.error("MemoryAgent.retrieve failed: %s", exc)
            return []

    def list_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the N most recently stored research sessions."""
        try:
            return self._manager.list_recent(n=n)
        except Exception as exc:
            logger.error("MemoryAgent.list_recent failed: %s", exc)
            return []

    def get_context_for_topic(self, topic: str) -> str:
        """
        Retrieve formatted past-research context string to inject into
        a new research session about `topic`.
        """
        past = self.retrieve(topic, n_results=3)
        if not past:
            return ""

        lines = ["## Relevant Past Research\n"]
        for p in past:
            lines.append(
                f"- **{p['topic']}** ({p['timestamp'][:10]}): {p['snippet']}\n"
            )
        return "\n".join(lines)
