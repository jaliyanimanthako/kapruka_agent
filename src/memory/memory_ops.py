"""High-level operations for the Kapruka 3-tier memory stack."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

from memory.lt_store import CatalogVectorStore
from memory.schemas import CatalogMatch, ConversationTurn, RecipientProfile
from memory.semantic_store import SemanticProfileStore
from memory.st_store import ShortTermMemoryStore


class CognitiveMemoryStack:
    """Coordinates short-term, long-term, and semantic memory."""

    def __init__(
        self,
        short_term: Optional[ShortTermMemoryStore] = None,
        long_term: Optional[CatalogVectorStore] = None,
        semantic: Optional[SemanticProfileStore] = None,
    ) -> None:
        self.short_term = short_term or ShortTermMemoryStore()
        self.long_term = long_term or CatalogVectorStore()
        self.semantic = semantic or SemanticProfileStore()

    def add_turn(self, user_id: str, session_id: str, role: str, content: str) -> ConversationTurn:
        turn = ConversationTurn(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
            ts=time.time(),
        )
        self.short_term.append(turn)
        return turn

    def recent_context(self, user_id: str, session_id: str, k: int = 6) -> List[ConversationTurn]:
        return self.short_term.recent(user_id=user_id, session_id=session_id, k=k)

    def sync_catalog(self, catalog_path: str | Path = "catalog.json") -> int:
        return self.long_term.ingest_catalog(catalog_path=catalog_path)

    def search_catalog(self, query: str, top_k: int = 5, score_threshold: float = 0.15) -> List[CatalogMatch]:
        return self.long_term.search(query=query, top_k=top_k, score_threshold=score_threshold)

    def save_recipient_profile(
        self,
        recipient_id: str,
        name: str,
        relationship: str = "",
        preferences: Optional[List[str]] = None,
        notes: Optional[List[str]] = None,
    ) -> RecipientProfile:
        profile = RecipientProfile(
            recipient_id=recipient_id,
            name=name,
            relationship=relationship,
            preferences=preferences or [],
            notes=notes or [],
        )
        return self.semantic.upsert_profile(profile)

    def get_recipient_profile(self, recipient_id: str) -> Optional[RecipientProfile]:
        return self.semantic.get_profile(recipient_id)

    def build_context_bundle(
        self,
        user_id: str,
        session_id: str,
        query: str,
        recipient_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, object]:
        return {
            "recent_turns": [turn.to_dict() for turn in self.recent_context(user_id, session_id)],
            "catalog_matches": [
                {
                    "product_id": match.product_id,
                    "score": match.score,
                    "product": match.product.to_dict(),
                }
                for match in self.search_catalog(query=query, top_k=top_k)
            ],
            "recipient_profile": (
                self.get_recipient_profile(recipient_id).to_dict()
                if recipient_id and self.get_recipient_profile(recipient_id)
                else None
            ),
        }
