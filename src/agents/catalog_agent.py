"""Catalog specialist for RAG-based product search."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from infastructure.llm_providers.llm_services import OpenAIChatService
from memory.memory_ops import CognitiveMemoryStack


@dataclass
class CatalogAgentResult:
    """Catalog specialist output."""

    query: str
    bundle: Dict[str, object]
    answer: str
    timings_ms: Dict[str, int]


class CatalogAgent:
    """Use the 3-tier memory stack to retrieve and explain catalog results."""

    def __init__(
        self,
        memory_stack: CognitiveMemoryStack,
        chat_service: Optional[Any] = None,
    ) -> None:
        self.memory_stack = memory_stack
        self.chat_service = chat_service

    def search(
        self,
        query: str,
        user_id: str,
        session_id: str,
        recipient_id: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> CatalogAgentResult:
        retrieval_started_at = time.perf_counter()
        direct_product_query = self.memory_stack.is_direct_product_query(query)

        if progress_callback:
            progress_callback("Loading short-term memory...")
        st_started_at = time.perf_counter()
        recent_turns = self.memory_stack.recent_context(user_id=user_id, session_id=session_id)
        short_term_ms = int((time.perf_counter() - st_started_at) * 1000)

        recipient_profile = None
        semantic_ms = 0
        if recipient_id:
            if progress_callback:
                progress_callback("Loading recipient profile...")
            semantic_started_at = time.perf_counter()
            recipient_profile = self.memory_stack.get_recipient_profile(recipient_id)
            semantic_ms = int((time.perf_counter() - semantic_started_at) * 1000)

        if progress_callback:
            progress_callback("Building retrieval query...")
        query_started_at = time.perf_counter()
        effective_query = query
        if not direct_product_query:
            enhanced_query = self.memory_stack._build_enhanced_query(
                user_id=user_id,
                session_id=session_id,
                query=query,
                recipient_id=recipient_id if self._profile_is_relevant(query, recipient_profile) else None,
            )
            effective_query = enhanced_query
        query_build_ms = int((time.perf_counter() - query_started_at) * 1000)

        if progress_callback:
            progress_callback("Searching catalog vectors...")
        vector_started_at = time.perf_counter()
        catalog_matches, vector_timings = self.memory_stack.long_term.search_detailed(
            query=effective_query,
            top_k=top_k,
            score_threshold=score_threshold,
            progress_callback=progress_callback,
        )
        vector_total_ms = int((time.perf_counter() - vector_started_at) * 1000)

        bundle = {
            "recent_turns": [] if direct_product_query else [turn.to_dict() for turn in recent_turns],
            "catalog_matches": [
                {
                    "product_id": match.product_id,
                    "score": match.score,
                    "product": match.product.to_dict(),
                }
                for match in catalog_matches
            ],
            "recipient_profile": None if direct_product_query else (recipient_profile.to_dict() if recipient_profile else None),
        }
        retrieval_ms = int((time.perf_counter() - retrieval_started_at) * 1000)

        answer_started_at = time.perf_counter()
        answer = self._answer(query=query, bundle=bundle)
        answer_ms = int((time.perf_counter() - answer_started_at) * 1000)

        return CatalogAgentResult(
            query=query,
            bundle=bundle,
            answer=answer,
            timings_ms={
                "short_term_read": short_term_ms,
                "semantic_profile_read": semantic_ms,
                "retrieval_query_build": query_build_ms,
                "vector_search_total": vector_total_ms,
                **vector_timings,
                "catalog_retrieval": retrieval_ms,
                "catalog_answer_generation": answer_ms,
            },
        )

    def _answer(self, query: str, bundle: Dict[str, object]) -> str:
        service = self.chat_service
        if service is None:
            try:
                service = OpenAIChatService()
            except Exception:
                service = None

        if service is not None:
            try:
                return service.answer_query(query=query, bundle=bundle)
            except Exception:
                pass

        catalog_matches = bundle.get("catalog_matches", [])
        if not catalog_matches:
            return "I could not find a strong product match in the current catalog."

        lines = ["Top matching products:"]
        for index, match in enumerate(catalog_matches[:3], 1):
            product = match["product"]
            lines.append(
                f"{index}. {product['name']} - {product['price']} - "
                f"{product['availability']} - {product['url']}"
            )
        return "\n".join(lines)

    def _profile_is_relevant(self, query: str, recipient_profile: Optional[object]) -> bool:
        if recipient_profile is None:
            return False

        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if query_tokens & {"gift", "wife", "husband", "girlfriend", "boyfriend", "anniversary", "birthday"}:
            return True

        profile_tokens = set()
        profile_dict = recipient_profile.to_dict() if hasattr(recipient_profile, "to_dict") else {}
        for value in profile_dict.get("preferences", []) + profile_dict.get("notes", []):
            profile_tokens.update(re.findall(r"[a-z0-9]+", str(value).lower()))

        return bool(query_tokens & profile_tokens)
