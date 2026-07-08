"""High-level operations for the Kapruka 3-tier memory stack."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from memory.lt_store import CatalogVectorStore
from memory.schemas import CatalogMatch, ConversationTurn, RecipientProfile
from memory.semantic_store import SemanticProfileStore
from memory.st_store import ShortTermMemoryStore


class CognitiveMemoryStack:
    """Coordinates short-term, long-term, and semantic memory."""

    DIRECT_PRODUCT_KEYWORDS = {
        "bluetooth",
        "speaker",
        "speakers",
        "soundbar",
        "led",
        "tv",
        "tvs",
        "television",
        "laptop",
        "phone",
        "mobile",
        "headphone",
        "headphones",
        "earbud",
        "earbuds",
        "watch",
        "camera",
        "tablet",
        "tab",
        "fan",
        "fridge",
        "microwave",
        "printer",
    }

    GIFT_CONTEXT_KEYWORDS = {
        "gift",
        "wife",
        "husband",
        "girlfriend",
        "boyfriend",
        "mother",
        "father",
        "friend",
        "birthday",
        "anniversary",
        "romantic",
        "surprise",
    }

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

    def search_catalog_with_memory(
        self,
        user_id: str,
        session_id: str,
        query: str,
        recipient_id: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[CatalogMatch]:
        """Search the catalog using recent turns and recipient profile as extra context."""
        enhanced_query = self._build_enhanced_query(
            user_id=user_id,
            session_id=session_id,
            query=query,
            recipient_id=recipient_id,
        )
        return self.search_catalog(
            query=enhanced_query,
            top_k=top_k,
            score_threshold=score_threshold,
        )

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
        score_threshold: float = 0.0,
    ) -> Dict[str, object]:
        catalog_matches = self.search_catalog_with_memory(
            user_id=user_id,
            session_id=session_id,
            query=query,
            recipient_id=recipient_id,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        return {
            "recent_turns": [turn.to_dict() for turn in self.recent_context(user_id, session_id)],
            "catalog_matches": [
                {
                    "product_id": match.product_id,
                    "score": match.score,
                    "product": match.product.to_dict(),
                }
                for match in catalog_matches
            ],
            "recipient_profile": (
                self.get_recipient_profile(recipient_id).to_dict()
                if recipient_id and self.get_recipient_profile(recipient_id)
                else None
            ),
        }

    def _build_enhanced_query(
        self,
        user_id: str,
        session_id: str,
        query: str,
        recipient_id: Optional[str] = None,
    ) -> str:
        """Compose a retrieval query from user intent, ST context, and semantic profile."""
        parts = [query.strip()]
        if self._is_direct_product_query(query):
            return "\n".join(part for part in parts if part)

        recent_turns = self.recent_context(user_id=user_id, session_id=session_id, k=4)
        if recent_turns:
            recent_summary = " ".join(turn.content for turn in recent_turns[-2:])
            if recent_summary:
                parts.append(recent_summary)

        if recipient_id:
            profile = self.get_recipient_profile(recipient_id)
            if profile:
                if profile.relationship:
                    parts.append(f"relationship: {profile.relationship}")
                if profile.preferences:
                    parts.append("preferences: " + ", ".join(profile.preferences))
                if profile.notes:
                    parts.append("notes: " + ", ".join(profile.notes))

        return "\n".join(part for part in parts if part)

    def is_direct_product_query(self, query: str) -> bool:
        """Return True when the user is asking for a specific product/category rather than preference-led gift discovery."""
        raw_tokens = re.findall(r"[a-z0-9]+", query.lower())
        tokens = set(raw_tokens)
        tokens.update(token[:-1] for token in raw_tokens if token.endswith("s") and len(token) > 3)
        if not tokens:
            return False

        has_product_keyword = bool(tokens & self.DIRECT_PRODUCT_KEYWORDS)
        has_gift_context = bool(tokens & self.GIFT_CONTEXT_KEYWORDS)

        return has_product_keyword and not has_gift_context

    def _is_direct_product_query(self, query: str) -> bool:
        return self.is_direct_product_query(query)
