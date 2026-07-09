"""Specialist orchestration for the Kapruka gift assistant."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

from agents.catalog_agent import CatalogAgent, CatalogAgentResult
from agents.logistics_agent import LogisticsAgent, LogisticsCheckResult
from agents.meta_agent import MetaAgent, MetaAgentResult
from agents.router import KaprukaRouter, RouteDecision
from memory.memory_ops import CognitiveMemoryStack
from memory.st_store import ShortTermMemoryStore


@dataclass
class OrchestratorResponse:
    """Full response from the specialist orchestration layer."""

    route: str
    reasoning: str
    answer: str
    route_decision: Dict[str, object]
    specialist_output: Dict[str, object]
    timings_ms: Dict[str, int]


class KaprukaOrchestrator:
    """Router + specialist agents on top of the 3-tier memory stack."""

    ROUTER_CONFIDENCE_FLOOR = 0.55

    def __init__(
        self,
        memory_stack: Optional[CognitiveMemoryStack] = None,
        router: Optional[KaprukaRouter] = None,
        catalog_agent: Optional[CatalogAgent] = None,
        logistics_agent: Optional[LogisticsAgent] = None,
        meta_agent: Optional[MetaAgent] = None,
    ) -> None:
        self.memory_stack = memory_stack or CognitiveMemoryStack()
        self.router = router or KaprukaRouter()
        self.catalog_agent = catalog_agent or CatalogAgent(self.memory_stack)
        self.logistics_agent = logistics_agent or LogisticsAgent()
        self.meta_agent = meta_agent or MetaAgent()
        self._active_recipient_by_session: Dict[tuple[str, str], Dict[str, str]] = {}

    def handle_message(
        self,
        user_message: str,
        user_id: str = "demo-user",
        session_id: str = "demo-session",
        recipient_id: Optional[str] = None,
        recipient_name: str = "",
        relationship: str = "",
        top_k: int = 5,
        score_threshold: float = 0.0,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> OrchestratorResponse:
        total_started_at = time.perf_counter()
        self._progress(progress_callback, "Routing message...")
        introduced_name = self.router.extract_user_name(user_message)
        if introduced_name:
            self.memory_stack.save_user_profile(user_id=user_id, name=introduced_name)

        route_started_at = time.perf_counter()
        initial_decision = self.router.route(user_message, memory_context="")
        if initial_decision.route in {"smalltalk", "identity"}:
            decision = initial_decision
            routing_memory_context = ""
        else:
            routing_memory_context = self._build_router_memory_context(user_id=user_id, session_id=session_id)
            decision = self.router.route(user_message, memory_context=routing_memory_context)
        route_ms = int((time.perf_counter() - route_started_at) * 1000)
        self._progress(
            progress_callback,
            f"Route selected: {decision.route} ({decision.reasoning})",
        )

        if decision.route in {"smalltalk", "identity"}:
            return self._handle_meta_message(
                user_message=user_message,
                user_id=user_id,
                decision=decision,
                route_ms=route_ms,
                total_started_at=total_started_at,
            )

        if decision.route == "unclear" or decision.confidence < self.ROUTER_CONFIDENCE_FLOOR:
            return self._handle_unclear_message(
                user_message=user_message,
                decision=decision,
                route_ms=route_ms,
                total_started_at=total_started_at,
            )

        if decision.route == "order_status":
            return self._handle_order_status(
                user_message=user_message,
                decision=decision,
                route_ms=route_ms,
                total_started_at=total_started_at,
            )

        resolved_recipient = self._resolve_recipient(
            user_message=user_message,
            user_id=user_id,
            session_id=session_id,
            recipient_id=recipient_id,
            recipient_name=recipient_name,
            relationship=relationship,
        )
        effective_recipient_id = resolved_recipient["recipient_id"] or None
        effective_recipient_name = resolved_recipient["recipient_name"]
        effective_relationship = resolved_recipient["relationship"]
        if effective_recipient_id:
            self._set_active_recipient(
                user_id=user_id,
                session_id=session_id,
                recipient_id=effective_recipient_id,
                recipient_name=effective_recipient_name,
                relationship=effective_relationship,
            )

        if decision.route == "preference_update":
            return self._handle_preference_update(
                user_message=user_message,
                decision=decision,
                user_id=user_id,
                session_id=session_id,
                recipient_id=effective_recipient_id,
                recipient_name=effective_recipient_name,
                relationship=effective_relationship,
                route_ms=route_ms,
                total_started_at=total_started_at,
                progress_callback=progress_callback,
            )

        if decision.route == "logistics_check":
            return self._handle_logistics(
                user_message=user_message,
                decision=decision,
                user_id=user_id,
                session_id=session_id,
                route_ms=route_ms,
                total_started_at=total_started_at,
                progress_callback=progress_callback,
                routing_memory_context=routing_memory_context,
            )

        return self._handle_catalog_search(
            query=user_message,
            decision=decision,
            user_id=user_id,
            session_id=session_id,
            recipient_id=effective_recipient_id,
            top_k=top_k,
            score_threshold=score_threshold,
            route_ms=route_ms,
            total_started_at=total_started_at,
            progress_callback=progress_callback,
        )

    def _handle_preference_update(
        self,
        user_message: str,
        decision: RouteDecision,
        user_id: str,
        session_id: str,
        recipient_id: Optional[str],
        recipient_name: str,
        relationship: str,
        route_ms: int,
        total_started_at: float,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> OrchestratorResponse:
        self._progress(progress_callback, "Updating semantic profile...")
        if not recipient_id:
            answer = "I can save preferences, but I need a recipient id or name first, for example `wife`, `friend`, or `mother`."
            self._progress(progress_callback, "Storing short-term turns...")
            store_started_at = time.perf_counter()
            self._store_turns(user_id=user_id, session_id=session_id, user_message=user_message, answer=answer)
            store_ms = int((time.perf_counter() - store_started_at) * 1000)
            return OrchestratorResponse(
                route=decision.route,
                reasoning=decision.reasoning,
                answer=answer,
                route_decision=asdict(decision),
                specialist_output={"recipient_profile": None, "extracted": {}},
                timings_ms={
                    "router": route_ms,
                    "preference_update": 0,
                    "turn_storage": store_ms,
                    "total": int((time.perf_counter() - total_started_at) * 1000),
                },
            )

        specialist_started_at = time.perf_counter()
        extracted = self.router.extract_preferences(user_message)
        profile = self.memory_stack.save_recipient_profile(
            recipient_id=recipient_id,
            name=recipient_name,
            relationship=relationship,
            preferences=extracted["preferences"],
            constraints=extracted["constraints"],
            notes=extracted["notes"],
        )
        self._set_active_recipient(
            user_id=user_id,
            session_id=session_id,
            recipient_id=profile.recipient_id,
            recipient_name=profile.name,
            relationship=profile.relationship,
        )
        specialist_ms = int((time.perf_counter() - specialist_started_at) * 1000)
        answer = (
            f"I updated {profile.name}'s profile with "
            f"{len(extracted['preferences'])} preference(s), "
            f"{len(extracted['constraints'])} constraint(s), and "
            f"{len(extracted['notes'])} note(s)."
        )
        self._progress(progress_callback, "Storing short-term turns...")
        store_started_at = time.perf_counter()
        self._store_turns(user_id=user_id, session_id=session_id, user_message=user_message, answer=answer)
        store_ms = int((time.perf_counter() - store_started_at) * 1000)
        return OrchestratorResponse(
            route=decision.route,
            reasoning=decision.reasoning,
            answer=answer,
            route_decision=asdict(decision),
            specialist_output={"recipient_profile": profile.to_dict(), "extracted": extracted},
            timings_ms={
                "router": route_ms,
                "preference_update": specialist_ms,
                "turn_storage": store_ms,
                "total": int((time.perf_counter() - total_started_at) * 1000),
            },
        )

    def _handle_meta_message(
        self,
        user_message: str,
        user_id: str,
        decision: RouteDecision,
        route_ms: int,
        total_started_at: float,
    ) -> OrchestratorResponse:
        user_profile = self.memory_stack.get_user_profile(user_id)
        meta_started_at = time.perf_counter()
        result: MetaAgentResult = self.meta_agent.answer(
            user_message=user_message,
            route=decision.route,
            user_profile_name=user_profile.name if user_profile else "",
            memory_context="",
        )
        meta_ms = int((time.perf_counter() - meta_started_at) * 1000)
        return OrchestratorResponse(
            route=decision.route,
            reasoning=decision.reasoning,
            answer=result.answer,
            route_decision=asdict(decision),
            specialist_output={
                "meta": {
                    "handled_directly": True,
                    "source": result.source,
                    "user_profile": user_profile.to_dict() if user_profile else None,
                }
            },
            timings_ms={
                "router": route_ms,
                "meta_response": meta_ms,
                "total": int((time.perf_counter() - total_started_at) * 1000),
            },
        )

    def _handle_unclear_message(
        self,
        user_message: str,
        decision: RouteDecision,
        route_ms: int,
        total_started_at: float,
    ) -> OrchestratorResponse:
        answer = (
            "I can help with Kapruka product search, recipient preferences, or Sri Lankan delivery feasibility. "
            "Tell me what you want to find, remember, or check."
        )
        return OrchestratorResponse(
            route=decision.route,
            reasoning=decision.reasoning,
            answer=answer,
            route_decision=asdict(decision),
            specialist_output={"meta": {"handled_directly": True, "user_message": user_message}},
            timings_ms={
                "router": route_ms,
                "total": int((time.perf_counter() - total_started_at) * 1000),
            },
        )

    def _handle_order_status(
        self,
        user_message: str,
        decision: RouteDecision,
        route_ms: int,
        total_started_at: float,
    ) -> OrchestratorResponse:
        answer = (
            "I do not have live order-tracking connected yet. "
            "If you share an order number, you would still need Kapruka's live order system or support for the exact status."
        )
        return OrchestratorResponse(
            route=decision.route,
            reasoning=decision.reasoning,
            answer=answer,
            route_decision=asdict(decision),
            specialist_output={"meta": {"handled_directly": True, "user_message": user_message}},
            timings_ms={
                "router": route_ms,
                "total": int((time.perf_counter() - total_started_at) * 1000),
            },
        )

    def _handle_logistics(
        self,
        user_message: str,
        decision: RouteDecision,
        user_id: str,
        session_id: str,
        route_ms: int,
        total_started_at: float,
        progress_callback: Optional[Callable[[str], None]] = None,
        routing_memory_context: str = "",
    ) -> OrchestratorResponse:
        self._progress(progress_callback, "Checking delivery feasibility...")
        specialist_started_at = time.perf_counter()
        result = self.logistics_agent.check_delivery(user_message, memory_context=routing_memory_context)
        specialist_ms = int((time.perf_counter() - specialist_started_at) * 1000)
        self._progress(progress_callback, "Storing short-term turns...")
        store_started_at = time.perf_counter()
        self._store_turns(user_id=user_id, session_id=session_id, user_message=user_message, answer=result.summary)
        store_ms = int((time.perf_counter() - store_started_at) * 1000)
        return OrchestratorResponse(
            route=decision.route,
            reasoning=decision.reasoning,
            answer=result.summary,
            route_decision=asdict(decision),
            specialist_output={"logistics": asdict(result)},
            timings_ms={
                "router": route_ms,
                "logistics_check": specialist_ms,
                "turn_storage": store_ms,
                "total": int((time.perf_counter() - total_started_at) * 1000),
            },
        )

    def _handle_catalog_search(
        self,
        query: str,
        decision: RouteDecision,
        user_id: str,
        session_id: str,
        recipient_id: Optional[str],
        top_k: int,
        score_threshold: float,
        route_ms: int,
        total_started_at: float,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> OrchestratorResponse:
        self._progress(progress_callback, "Running catalog retrieval...")
        specialist_started_at = time.perf_counter()
        result: CatalogAgentResult = self.catalog_agent.search(
            query=query,
            user_id=user_id,
            session_id=session_id,
            recipient_id=recipient_id,
            top_k=top_k,
            score_threshold=score_threshold,
            progress_callback=progress_callback,
        )
        specialist_ms = int((time.perf_counter() - specialist_started_at) * 1000)
        self._progress(progress_callback, "Storing short-term turns...")
        store_started_at = time.perf_counter()
        self._store_turns(user_id=user_id, session_id=session_id, user_message=query, answer=result.answer)
        store_ms = int((time.perf_counter() - store_started_at) * 1000)
        return OrchestratorResponse(
            route=decision.route,
            reasoning=decision.reasoning,
            answer=result.answer,
            route_decision=asdict(decision),
            specialist_output={
                "catalog": {
                    "query": result.query,
                    "bundle": result.bundle,
                    "memory_gate": result.memory_gate,
                    "timings_ms": result.timings_ms,
                }
            },
            timings_ms={
                "router": route_ms,
                "catalog_specialist_total": specialist_ms,
                **result.timings_ms,
                "turn_storage": store_ms,
                "total": int((time.perf_counter() - total_started_at) * 1000),
            },
        )

    def _store_turns(self, user_id: str, session_id: str, user_message: str, answer: str) -> None:
        self.memory_stack.add_turn(user_id, session_id, "user", user_message)
        self.memory_stack.add_turn(user_id, session_id, "assistant", answer)

    def _build_router_memory_context(self, user_id: str, session_id: str) -> str:
        recent_turns = self.memory_stack.recent_context(user_id=user_id, session_id=session_id, k=4)
        if not recent_turns:
            return ""
        return "\n".join(f"{turn.role}: {turn.content}" for turn in recent_turns)

    def _resolve_recipient(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        recipient_id: Optional[str],
        recipient_name: str,
        relationship: str,
    ) -> Dict[str, str]:
        if recipient_id:
            return {
                "recipient_id": recipient_id,
                "recipient_name": recipient_name or recipient_id.replace("-", " ").title(),
                "relationship": relationship,
            }

        inferred = self.router.extract_recipient_reference(user_message)
        if inferred["recipient_id"]:
            return inferred

        active = self._active_recipient_by_session.get((user_id, session_id))
        if active and self._mentions_active_recipient_pronoun(user_message):
            return active

        return active or {
            "recipient_id": "",
            "recipient_name": "",
            "relationship": "",
        }

    def _set_active_recipient(
        self,
        user_id: str,
        session_id: str,
        recipient_id: str,
        recipient_name: str,
        relationship: str,
    ) -> None:
        self._active_recipient_by_session[(user_id, session_id)] = {
            "recipient_id": recipient_id,
            "recipient_name": recipient_name,
            "relationship": relationship,
        }

    def _progress(
        self,
        progress_callback: Optional[Callable[[str], None]],
        message: str,
    ) -> None:
        if progress_callback:
            progress_callback(message)

    def _mentions_active_recipient_pronoun(self, user_message: str) -> bool:
        return bool(re.search(r"\b(her|she|hers|him|he|his|them|they|their)\b", user_message, flags=re.IGNORECASE))

def build_orchestrator(
    chat_service: Optional[Any] = None,
    use_database_short_term: bool = False,
) -> KaprukaOrchestrator:
    """Build the default Kapruka specialist orchestrator."""
    memory_stack = CognitiveMemoryStack(
        short_term=ShortTermMemoryStore(use_database=use_database_short_term),
    )
    catalog_agent = CatalogAgent(memory_stack=memory_stack, chat_service=chat_service)
    return KaprukaOrchestrator(memory_stack=memory_stack, catalog_agent=catalog_agent)
