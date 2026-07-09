"""Catalog specialist for RAG-based product search."""

from __future__ import annotations

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
    memory_gate: Dict[str, object]
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
            progress_callback("Evaluating memory relevance...")
        gate_started_at = time.perf_counter()
        memory_gate = self.memory_stack.decide_memory_reads(
            user_id=user_id,
            session_id=session_id,
            query=query,
            recipient_id=recipient_id,
            recent_turns=recent_turns,
            recipient_profile=recipient_profile,
        )
        gate_ms = int((time.perf_counter() - gate_started_at) * 1000)

        if progress_callback:
            progress_callback("Building retrieval query...")
        query_started_at = time.perf_counter()
        effective_query = self.memory_stack.build_retrieval_query(
            query=query,
            recent_turns=recent_turns,
            recipient_profile=recipient_profile,
            decision=memory_gate,
        )
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
            "recent_turns": [turn.to_dict() for turn in recent_turns] if memory_gate.use_short_term else [],
            "catalog_matches": [
                {
                    "product_id": match.product_id,
                    "score": match.score,
                    "product": match.product.to_dict(),
                }
                for match in catalog_matches
            ],
            "recipient_profile": (
                recipient_profile.to_dict()
                if memory_gate.use_recipient_profile and recipient_profile
                else None
            ),
        }
        retrieval_ms = int((time.perf_counter() - retrieval_started_at) * 1000)

        answer_started_at = time.perf_counter()
        answer = self._answer(query=query, bundle=bundle)
        answer_ms = int((time.perf_counter() - answer_started_at) * 1000)

        return CatalogAgentResult(
            query=query,
            bundle=bundle,
            memory_gate=memory_gate.to_dict(),
            answer=answer,
            timings_ms={
                "short_term_read": short_term_ms,
                "semantic_profile_read": semantic_ms,
                "memory_relevance_gate": gate_ms,
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
