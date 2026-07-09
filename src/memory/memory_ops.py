"""High-level operations for the Kapruka 3-tier memory stack."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from memory.lt_store import CatalogVectorStore
from memory.schemas import CatalogMatch, ConversationTurn, RecipientProfile, UserProfile
from memory.semantic_store import SemanticProfileStore
from memory.st_store import ShortTermMemoryStore
from memory.user_store import UserProfileStore


@dataclass
class MemoryReadDecision:
    """Binary read gate for deciding which memory layers to inject."""

    topic_shifted: bool
    is_direct_product_query: bool
    use_short_term: bool
    use_recipient_profile: bool
    available_recent_turns: int
    applied_recent_turns: int
    available_recipient_profile: bool
    recipient_id: Optional[str]
    reason: str
    short_term_reason: str
    recipient_profile_reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "topic_shifted": self.topic_shifted,
            "is_direct_product_query": self.is_direct_product_query,
            "use_short_term": self.use_short_term,
            "use_recipient_profile": self.use_recipient_profile,
            "available_recent_turns": self.available_recent_turns,
            "applied_recent_turns": self.applied_recent_turns,
            "available_recipient_profile": self.available_recipient_profile,
            "recipient_id": self.recipient_id,
            "reason": self.reason,
            "short_term_reason": self.short_term_reason,
            "recipient_profile_reason": self.recipient_profile_reason,
        }


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

    FOLLOW_UP_KEYWORDS = {
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "them",
        "more",
        "another",
        "similar",
        "same",
        "ones",
        "instead",
    }

    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "can",
        "for",
        "give",
        "hi",
        "i",
        "in",
        "is",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "show",
        "some",
        "the",
        "to",
        "what",
        "your",
    }

    def __init__(
        self,
        short_term: Optional[ShortTermMemoryStore] = None,
        long_term: Optional[CatalogVectorStore] = None,
        semantic: Optional[SemanticProfileStore] = None,
        user_store: Optional[UserProfileStore] = None,
    ) -> None:
        self.short_term = short_term or ShortTermMemoryStore()
        self.long_term = long_term or CatalogVectorStore()
        self.semantic = semantic or SemanticProfileStore()
        self.user_store = user_store or UserProfileStore()

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
        recent_turns = self.recent_context(user_id=user_id, session_id=session_id)
        recipient_profile = self.get_recipient_profile(recipient_id) if recipient_id else None
        decision = self.decide_memory_reads(
            user_id=user_id,
            session_id=session_id,
            query=query,
            recipient_id=recipient_id,
            recent_turns=recent_turns,
            recipient_profile=recipient_profile,
        )
        enhanced_query = self.build_retrieval_query(
            query=query,
            recent_turns=recent_turns,
            recipient_profile=recipient_profile,
            decision=decision,
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
        constraints: Optional[List[str]] = None,
        notes: Optional[List[str]] = None,
    ) -> RecipientProfile:
        profile = RecipientProfile(
            recipient_id=recipient_id,
            name=name,
            relationship=relationship,
            preferences=preferences or [],
            constraints=constraints or [],
            notes=notes or [],
        )
        return self.semantic.upsert_profile(profile)

    def get_recipient_profile(self, recipient_id: str) -> Optional[RecipientProfile]:
        return self.semantic.get_profile(recipient_id)

    def save_user_profile(self, user_id: str, name: str) -> UserProfile:
        profile = UserProfile(user_id=user_id, name=name)
        return self.user_store.upsert_profile(profile)

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        return self.user_store.get_profile(user_id)

    def build_context_bundle(
        self,
        user_id: str,
        session_id: str,
        query: str,
        recipient_id: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> Dict[str, object]:
        recent_turns = self.recent_context(user_id=user_id, session_id=session_id)
        recipient_profile = self.get_recipient_profile(recipient_id) if recipient_id else None
        decision = self.decide_memory_reads(
            user_id=user_id,
            session_id=session_id,
            query=query,
            recipient_id=recipient_id,
            recent_turns=recent_turns,
            recipient_profile=recipient_profile,
        )
        retrieval_query = self.build_retrieval_query(
            query=query,
            recent_turns=recent_turns,
            recipient_profile=recipient_profile,
            decision=decision,
        )
        catalog_matches = self.search_catalog(
            query=retrieval_query,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        return {
            "recent_turns": (
                [turn.to_dict() for turn in recent_turns] if decision.use_short_term else []
            ),
            "catalog_matches": [
                {
                    "product_id": match.product_id,
                    "score": match.score,
                    "product": match.product.to_dict(),
                }
                for match in catalog_matches
            ],
            "recipient_profile": (
                recipient_profile.to_dict() if decision.use_recipient_profile and recipient_profile else None
            ),
            "memory_gate": decision.to_dict(),
        }

    def _build_enhanced_query(
        self,
        user_id: str,
        session_id: str,
        query: str,
        recipient_id: Optional[str] = None,
    ) -> str:
        """Compose a retrieval query from user intent, ST context, and semantic profile."""
        recent_turns = self.recent_context(user_id=user_id, session_id=session_id, k=4)
        recipient_profile = self.get_recipient_profile(recipient_id) if recipient_id else None
        decision = self.decide_memory_reads(
            user_id=user_id,
            session_id=session_id,
            query=query,
            recipient_id=recipient_id,
            recent_turns=recent_turns,
            recipient_profile=recipient_profile,
        )
        return self.build_retrieval_query(
            query=query,
            recent_turns=recent_turns,
            recipient_profile=recipient_profile,
            decision=decision,
        )

    def decide_memory_reads(
        self,
        user_id: str,
        session_id: str,
        query: str,
        recipient_id: Optional[str] = None,
        recent_turns: Optional[List[ConversationTurn]] = None,
        recipient_profile: Optional[RecipientProfile] = None,
    ) -> MemoryReadDecision:
        """Decide whether short-term and recipient-profile memory should affect retrieval."""
        recent_turns = recent_turns if recent_turns is not None else self.recent_context(user_id, session_id)
        if recipient_profile is None and recipient_id:
            recipient_profile = self.get_recipient_profile(recipient_id)

        direct_product_query = self.is_direct_product_query(query)
        follow_up = self._looks_like_follow_up(query)
        query_tokens = self._meaningful_tokens(query)
        recent_tokens = self._meaningful_tokens(" ".join(turn.content for turn in recent_turns[-4:]))
        overlap = query_tokens & recent_tokens

        topic_shifted = bool(recent_turns) and direct_product_query and not follow_up and not overlap

        if not recent_turns:
            use_short_term = False
            short_term_reason = "No short-term turns are available."
        elif follow_up:
            use_short_term = True
            short_term_reason = "The query looks like a follow-up, so recent conversation context is useful."
        elif topic_shifted:
            use_short_term = False
            short_term_reason = "The query appears to be a fresh product topic, so recent turns are gated off."
        elif direct_product_query:
            use_short_term = False
            short_term_reason = "This is a direct product/category lookup, so recent turns are not injected."
        else:
            use_short_term = True
            short_term_reason = "The query is conversational or gift-led, so recent turns remain active."

        if recipient_profile is None:
            use_recipient_profile = False
            recipient_profile_reason = "No recipient profile is available."
        elif self._query_needs_gift_context(query):
            use_recipient_profile = True
            recipient_profile_reason = "The query is gift-led or recipient-led, so the profile is relevant."
        elif self._profile_token_overlap(query, recipient_profile):
            use_recipient_profile = True
            recipient_profile_reason = "The query overlaps with saved recipient preferences or notes."
        else:
            use_recipient_profile = False
            recipient_profile_reason = "The query does not appear relevant to the saved recipient profile."

        if use_short_term and use_recipient_profile:
            reason = "Using both short-term context and recipient profile."
        elif use_short_term:
            reason = "Using short-term context only."
        elif use_recipient_profile:
            reason = "Using recipient profile only."
        else:
            reason = "Using catalog retrieval only."

        return MemoryReadDecision(
            topic_shifted=topic_shifted,
            is_direct_product_query=direct_product_query,
            use_short_term=use_short_term,
            use_recipient_profile=use_recipient_profile,
            available_recent_turns=len(recent_turns),
            applied_recent_turns=len(recent_turns) if use_short_term else 0,
            available_recipient_profile=recipient_profile is not None,
            recipient_id=recipient_id,
            reason=reason,
            short_term_reason=short_term_reason,
            recipient_profile_reason=recipient_profile_reason,
        )

    def build_retrieval_query(
        self,
        query: str,
        recent_turns: Optional[List[ConversationTurn]] = None,
        recipient_profile: Optional[RecipientProfile] = None,
        decision: Optional[MemoryReadDecision] = None,
    ) -> str:
        """Compose the final retrieval query after memory gating."""
        parts = [query.strip()]

        if decision and decision.use_short_term and recent_turns:
            recent_summary = " ".join(turn.content for turn in recent_turns[-2:])
            if recent_summary:
                parts.append(recent_summary)

        if decision and decision.use_recipient_profile and recipient_profile:
            if recipient_profile.relationship:
                parts.append(f"relationship: {recipient_profile.relationship}")
            if recipient_profile.preferences:
                parts.append("preferences: " + ", ".join(recipient_profile.preferences))
            if recipient_profile.constraints:
                parts.append("constraints: " + ", ".join(recipient_profile.constraints))
            if recipient_profile.notes:
                parts.append("notes: " + ", ".join(recipient_profile.notes))

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

    def _query_needs_gift_context(self, query: str) -> bool:
        return bool(self._meaningful_tokens(query) & self.GIFT_CONTEXT_KEYWORDS)

    def _profile_token_overlap(self, query: str, recipient_profile: RecipientProfile) -> bool:
        query_tokens = self._meaningful_tokens(query)
        profile_tokens = self._meaningful_tokens(
            " ".join(
                [
                    recipient_profile.relationship,
                    *recipient_profile.preferences,
                    *recipient_profile.constraints,
                    *recipient_profile.notes,
                ]
            )
        )
        return bool(query_tokens & profile_tokens)

    def _looks_like_follow_up(self, query: str) -> bool:
        normalized = query.lower()
        if any(phrase in normalized for phrase in ("what about", "how about", "more like", "similar to")):
            return True
        return bool(self._meaningful_tokens(query) & self.FOLLOW_UP_KEYWORDS)

    def _meaningful_tokens(self, text: str) -> set[str]:
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in self.STOPWORDS
        }
        tokens.update(token[:-1] for token in list(tokens) if token.endswith("s") and len(token) > 3)
        return tokens
