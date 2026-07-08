"""LLM-first intent router for the Kapruka specialist orchestrator."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.prompts.agent_prompts import build_router_policy_prompt, build_router_user_prompt
from infastructure.config import OPENAI_API_KEY, OPENAI_CHAT_MAX_TOKENS, OPENAI_CHAT_MODEL

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenAI = None


PREFERENCE_PATTERNS = (
    r"\bremember\b",
    r"\bnote that\b",
    r"\bkeep in mind\b",
    r"\bsave (this|that)\b",
    r"\bmy (wife|husband|girlfriend|boyfriend|mother|father|friend) (likes|loves|prefers|enjoys)\b",
    r"\b(likes|loves|prefers|enjoys|dislikes|hates|avoids)\b",
    r"\bmy budget is\b",
    r"\bremember .*budget\b",
)

LOGISTICS_PATTERNS = (
    r"\bdeliver(y|able)?\b",
    r"\bship(ping)?\b",
    r"\bsend (it|this|that)\b",
    r"\breach\b",
    r"\barrive\b",
    r"\bdispatch\b",
    r"\beta\b",
    r"\bdistrict\b",
    r"\barea\b",
    r"\blocation\b",
    r"\bcolombo\b",
    r"\bkandy\b",
    r"\bgalle\b",
    r"\bjaffna\b",
    r"\bmatara\b",
    r"\bkurunegala\b",
    r"\bgampaha\b",
    r"\bsame[- ]day\b",
    r"\btoday\b",
    r"\btomorrow\b",
)

VALID_ROUTES = {"catalog_search", "preference_update", "logistics_check"}
RECIPIENT_RELATIONS = {
    "wife": ("Wife", "spouse"),
    "husband": ("Husband", "spouse"),
    "girlfriend": ("Girlfriend", "partner"),
    "boyfriend": ("Boyfriend", "partner"),
    "mother": ("Mother", "parent"),
    "father": ("Father", "parent"),
    "friend": ("Friend", "friend"),
}


@dataclass
class RouteDecision:
    """Routing result for the specialist orchestrator."""

    route: str
    confidence: float
    reasoning: str
    params: Dict[str, str] = field(default_factory=dict)


class KaprukaRouter:
    """Classify a user message into Kapruka specialist routes."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model: str = OPENAI_CHAT_MODEL,
        max_tokens: int = OPENAI_CHAT_MAX_TOKENS,
        use_llm: bool = True,
    ) -> None:
        self.system_prompt = build_router_policy_prompt()
        self.model = model
        self.max_tokens = max_tokens
        self._use_llm = use_llm

        if llm_client is not None:
            self.client = llm_client
        elif use_llm and OpenAI is not None and OPENAI_API_KEY:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.client = None

    def route(self, user_message: str, memory_context: str = "") -> RouteDecision:
        text = user_message.strip()
        if self.client is not None and self._use_llm:
            try:
                return self._route_with_llm(text, memory_context=memory_context)
            except Exception:
                pass
        return self._route_with_rules(text, memory_context=memory_context)

    def extract_preferences(self, user_message: str) -> Dict[str, List[str]]:
        """Extract lightweight preferences and notes from a user message."""
        text = user_message.strip()
        preferences: List[str] = []
        notes: List[str] = []

        for keyword in ("likes", "loves", "prefers", "enjoys"):
            pattern = rf"\b{keyword}\b\s+([^.!?]+)"
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                fragment = match.group(1).strip(" -:")
                for item in self._split_items(fragment):
                    cleaned = item.strip()
                    if cleaned:
                        preferences.append(f"{keyword.title()} {cleaned}")

        if not preferences:
            cleaned = re.sub(r"^(remember|note that)\s+", "", text, flags=re.IGNORECASE).strip()
            if cleaned:
                notes.append(cleaned)

        return {
            "preferences": self._dedupe(preferences),
            "notes": self._dedupe(notes),
        }

    def extract_recipient_reference(self, user_message: str) -> Dict[str, str]:
        """Infer recipient identity from the current message when possible."""
        text = user_message.strip()
        lowered = text.lower()

        relation_match = re.search(
            r"\bmy (?P<relation>wife|husband|girlfriend|boyfriend|mother|father|friend)\b",
            lowered,
        )
        relation = relation_match.group("relation") if relation_match else ""

        name_match = re.search(
            r"\b(?:name is|called)\s+(?P<name>[a-z][a-z\s'-]{1,40})",
            lowered,
        )
        raw_name = name_match.group("name").strip() if name_match else ""

        if raw_name:
            recipient_name = " ".join(part.capitalize() for part in raw_name.split())
            recipient_id = re.sub(r"[^a-z0-9]+", "-", raw_name.lower()).strip("-")
            relationship = RECIPIENT_RELATIONS.get(relation, ("", relation))[1] if relation else ""
            return {
                "recipient_id": recipient_id,
                "recipient_name": recipient_name,
                "relationship": relationship,
            }

        if relation:
            default_name, relationship = RECIPIENT_RELATIONS[relation]
            return {
                "recipient_id": relation,
                "recipient_name": default_name,
                "relationship": relationship,
            }

        return {
            "recipient_id": "",
            "recipient_name": "",
            "relationship": "",
        }

    def _route_with_llm(self, user_message: str, memory_context: str = "") -> RouteDecision:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=min(self.max_tokens, 200),
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": build_router_user_prompt(user_message, memory_context=memory_context)},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return self._parse_llm_response(content=content, user_message=user_message)

    def _parse_llm_response(self, content: str, user_message: str) -> RouteDecision:
        data = json.loads(content)
        route = str(data.get("route", "")).strip()
        if route not in VALID_ROUTES:
            raise ValueError(f"Invalid route from router LLM: {route}")

        params = data.get("params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        normalized_params: Dict[str, str] = {}
        if route == "catalog_search":
            normalized_params["query"] = str(params.get("query") or user_message)
        else:
            normalized_params["message"] = str(params.get("message") or user_message)

        return RouteDecision(
            route=route,
            confidence=float(data.get("confidence", 0.75)),
            reasoning=str(data.get("reasoning", "")).strip() or "LLM-based routing decision.",
            params=normalized_params,
        )

    def _route_with_rules(self, user_message: str, memory_context: str = "") -> RouteDecision:
        lowered = user_message.lower()

        if self._is_preference_update(lowered):
            return RouteDecision(
                route="preference_update",
                confidence=0.94,
                reasoning="The user is providing recipient preferences or memory-worthy profile details.",
                params={"message": user_message},
            )

        if self._is_logistics_question(lowered, memory_context=memory_context):
            return RouteDecision(
                route="logistics_check",
                confidence=0.90,
                reasoning="The user is asking about delivery location, timing, or feasibility in Sri Lanka.",
                params={"message": user_message},
            )

        return RouteDecision(
            route="catalog_search",
            confidence=0.84,
            reasoning="The message is best handled as a Kapruka gift or product search request.",
            params={"query": user_message},
        )

    def _is_preference_update(self, text: str) -> bool:
        if any(
            phrase in text
            for phrase in (
                "suggest",
                "recommend",
                "show me",
                "find me",
                "i need a gift",
                "gift for",
                "looking for",
            )
        ):
            return False
        return self._matches_any(text, PREFERENCE_PATTERNS)

    def _is_logistics_question(self, text: str, memory_context: str = "") -> bool:
        if self._matches_any(text, LOGISTICS_PATTERNS):
            return True
        if "can you" in text and any(word in text for word in ("deliver", "send", "ship")):
            return True
        if self._is_logistics_follow_up(text, memory_context):
            return True
        return False

    def _is_logistics_follow_up(self, text: str, memory_context: str) -> bool:
        if not memory_context:
            return False

        memory_lower = memory_context.lower()
        if not any(word in memory_lower for word in ("deliver", "delivery", "district", "logistics", "arrival")):
            return False

        if re.search(r"\b(i am|i'm|im|near|around|from|at|in)\b", text):
            return True

        if len(text.split()) <= 5:
            return True

        return False

    def _matches_any(self, text: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _split_items(self, text: str) -> List[str]:
        return [
            item.strip()
            for item in re.split(r",| and | but |;", text, flags=re.IGNORECASE)
            if item.strip()
        ]

    def _dedupe(self, values: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for value in values:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(value)
        return ordered
