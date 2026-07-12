"""LLM-first intent router for the Kapruka specialist orchestrator."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.logistics_agent import SRI_LANKAN_DISTRICTS
from agents.prompts.agent_prompts import (
    build_profile_update_guard_policy_prompt,
    build_profile_update_guard_user_prompt,
    build_router_policy_prompt,
    build_router_user_prompt,
)
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
    r"\ballerg(?:y|ic)\b",
    r"\b(?:cannot|can't|can not|should not|must not)\s+(?:eat|have|take)\b",
    r"\b(?:only\s+can|can\s+only)\s+(?:eat|have|take)\b",
    r"\bmy budget is\b",
    r"\bremember .*budget\b",
)

PRODUCT_SEARCH_PATTERNS = (
    r"\bwhat are (my|the|your)?\s*options\b",
    r"\bshow me\b",
    r"\bfind me\b",
    r"\blooking for\b",
    r"\bi need\b",
    r"\bi want\b",
    r"\boptions in\b",
    r"\boptions for\b",
    r"\bgift for\b",
    r"\brecommend\b",
    r"\bsuggest\b",
    r"\bbuy\b",
    r"\bprice\b",
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

SMALLTALK_PATTERNS = (
    r"^\s*(hi|hello|hey|yo|good morning|good afternoon|good evening)\b[\s!.?]*$",
    r"^\s*(thanks|thank you|thx|bye|goodbye|see you)\b[\s!.?]*$",
)

IDENTITY_PATTERNS = (
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\bwhat can you do\b",
    r"\bwhat do you do\b",
    r"\bwho built you\b",
    r"\byour purpose\b",
    r"\bare you (a|an)\b",
    r"\bcall you\b",
    r"\bname you\b",
)

USER_NAME_PATTERNS = (
    r"\bwhat is my name\b",
    r"\bwhat's my name\b",
    r"\bdo you know my name\b",
    r"\bremember my name\b",
)

ORDER_STATUS_PATTERNS = (
    r"\border status\b",
    r"\bstatus of my order\b",
    r"\bwhere is my order\b",
    r"\btrack(ing)? my order\b",
    r"\border number\b",
    r"\bmy order\b",
)

UNCLEAR_PATTERNS = (
    r"^\s*(ok|okay|hmm|huh|hmmm)\s*$",
)

SELF_INTRO_PATTERNS = (
    r"^\s*(hi|hello|hey)\b.*\b(?:i am|i'm|im|my name is|call me)\b",
    r"^\s*(?:i am|i'm|im)\s+(?!near\b|from\b|at\b|in\b|around\b)[a-z][a-z'-]{1,39}\b",
    r"^\s*(?:my name is|call me)\s+[a-z][a-z'-]{1,39}\b",
)

VALID_ROUTES = {
    "smalltalk",
    "identity",
    "catalog_search",
    "preference_update",
    "logistics_check",
    "order_status",
    "unclear",
}
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
        self.profile_update_guard_prompt = build_profile_update_guard_policy_prompt()

        if llm_client is not None:
            self.client = llm_client
        elif use_llm and OpenAI is not None and OPENAI_API_KEY:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.client = None

    def route(self, user_message: str, memory_context: str = "") -> RouteDecision:
        text = user_message.strip()
        if not memory_context:
            stateless_decision = self._route_stateless(text)
            if stateless_decision is not None:
                return self._postprocess_decision(text, stateless_decision)
        if self.client is not None and self._use_llm:
            try:
                return self._postprocess_decision(
                    text,
                    self._route_with_llm(text, memory_context=memory_context),
                )
            except Exception:
                pass
        return self._postprocess_decision(text, self._route_with_rules(text, memory_context=memory_context))

    def extract_preferences(self, user_message: str) -> Dict[str, List[str]]:
        """Extract lightweight preferences and notes from a user message."""
        text = user_message.strip()
        preferences: List[str] = []
        constraints: List[str] = []
        notes: List[str] = []

        normalized_text = self._normalize_preference_text(text)

        for keyword in ("likes", "loves", "prefers", "enjoys"):
            pattern = rf"\b{keyword}\b\s+([^.!?]+)"
            for match in re.finditer(pattern, normalized_text, flags=re.IGNORECASE):
                fragment = match.group(1).strip(" -:")
                for item in self._split_items(fragment):
                    cleaned = item.strip()
                    if cleaned:
                        preferences.append(f"{keyword.title()} {cleaned}")

        for pattern in (
            r"\b(?P<item>[a-z\s]+?)\s+(?:is|are)\s+(?:an?\s+)?allerg(?:y|ic)\b",
            r"\ballerg(?:y|ic)\s+(?:to|for)\s+(?P<item>[a-z\s]+)",
            r"\b(?:cannot|can't|can not|should not|must not)\s+(?:eat|have|take)\s+(?P<item>[a-z\s]+)",
            r"\b(?:avoid|avoids|hates|dislikes)\s+(?P<item>[a-z\s]+)",
        ):
            for match in re.finditer(pattern, normalized_text, flags=re.IGNORECASE):
                for cleaned in self._constraint_items(match.group("item").strip(" -:")):
                    if cleaned:
                        constraints.append(f"Avoids {cleaned}")

        positive_allowed_match = re.search(
            r"\b(?:only\s+can|can\s+only)\s+(?:eat|have|take)\s+(?P<item>[^.!?]+)",
            normalized_text,
            flags=re.IGNORECASE,
        )
        if positive_allowed_match:
            for item in self._split_items(positive_allowed_match.group("item")):
                cleaned = self._clean_preference_fragment(item)
                if cleaned:
                    preferences.append(f"Can eat {cleaned}")

        if not preferences and not constraints:
            cleaned = re.sub(
                r"^(remember(?:\s+that)?|note\s+that|keep\s+in\s+mind(?:\s+that)?|save\s+(?:this|that))\s+",
                "",
                normalized_text,
                flags=re.IGNORECASE,
            ).strip()
            if cleaned:
                notes.append(cleaned)

        return {
            "preferences": self._dedupe(preferences),
            "constraints": self._dedupe(constraints),
            "notes": self._dedupe(notes),
        }

    def should_persist_profile_update(
        self,
        user_message: str,
        extracted: Dict[str, List[str]],
    ) -> bool:
        """Decide whether a routed preference update should actually mutate profile memory."""
        llm_decision = self._judge_profile_update_persistence(user_message=user_message, extracted=extracted)
        if llm_decision is not None:
            return llm_decision

        if extracted.get("preferences") or extracted.get("constraints"):
            return True

        notes = extracted.get("notes", [])
        if not notes:
            return False

        lowered = user_message.lower()
        if self._has_explicit_memory_intent(lowered):
            return True
        if self._looks_like_note_worthy_profile_fact(lowered):
            return True
        return False

    def _judge_profile_update_persistence(
        self,
        user_message: str,
        extracted: Dict[str, List[str]],
    ) -> Optional[bool]:
        if self.client is None or not self._use_llm:
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=min(self.max_tokens, 120),
                messages=[
                    {"role": "system", "content": self.profile_update_guard_prompt},
                    {
                        "role": "user",
                        "content": build_profile_update_guard_user_prompt(
                            user_message=user_message,
                            extracted=extracted,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
        except Exception:
            return None

        if "should_persist" not in data:
            return None
        return bool(data.get("should_persist"))

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
        if not raw_name and relation:
            relation_name_match = re.search(
                rf"\bmy {relation}\s+(?!(?:likes|loves|prefers|enjoys|dislikes|hates|avoids|birthday)\b)(?P<name>[a-z][a-z-]{{1,40}})(?:'s)?\b",
                lowered,
            )
            if relation_name_match:
                raw_name = relation_name_match.group("name").strip()

        if raw_name:
            recipient_name = " ".join(part.capitalize() for part in raw_name.split())
            recipient_id = relation or re.sub(r"[^a-z0-9]+", "-", raw_name.lower()).strip("-")
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

    def extract_user_name(self, user_message: str) -> str:
        """Extract the user's own name from self-introduction phrases when possible."""
        text = user_message.strip()

        patterns = (
            r"\b(?:i am|i'm|im)\s+(?P<name>[A-Za-z][A-Za-z'-]{1,39})\b",
            r"\bmy name is\s+(?P<name>[A-Za-z][A-Za-z'-]{1,39})\b",
            r"\bcall me\s+(?P<name>[A-Za-z][A-Za-z'-]{1,39})\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                raw_name = match.group("name").strip()
                return raw_name[:1].upper() + raw_name[1:]
        return ""

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
        route = str(data.get("intent") or data.get("route") or "").strip()
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
            reasoning=(
                str(data.get("reason") or data.get("reasoning") or "").strip()
                or "LLM-based routing decision."
            ),
            params=normalized_params,
        )

    def _postprocess_decision(self, user_message: str, decision: RouteDecision) -> RouteDecision:
        lowered = user_message.lower()
        if decision.route == "preference_update" and self._looks_like_recommendation_feedback(lowered):
            return RouteDecision(
                route="catalog_search",
                confidence=max(decision.confidence, 0.90),
                reasoning=(
                    "The message is steering product suggestions toward different items, "
                    "so it should be handled as catalog search instead of a memory update."
                ),
                params={"query": user_message},
            )
        return decision

    def _route_with_rules(self, user_message: str, memory_context: str = "") -> RouteDecision:
        lowered = user_message.lower()

        if self._is_user_name_question(lowered):
            return RouteDecision(
                route="identity",
                confidence=0.98,
                reasoning="The user is asking whether their saved name is known.",
                params={"message": user_message},
            )

        if self._is_identity_question(lowered):
            return RouteDecision(
                route="identity",
                confidence=0.98,
                reasoning="The user is asking who the assistant is or what it does.",
                params={"message": user_message},
            )

        if self._is_logistics_question(lowered, memory_context=memory_context):
            return RouteDecision(
                route="logistics_check",
                confidence=0.92,
                reasoning="The user is asking about delivery location, timing, or feasibility in Sri Lanka.",
                params={"message": user_message},
            )

        if self._is_product_search(lowered):
            return RouteDecision(
                route="catalog_search",
                confidence=0.90,
                reasoning="The user is asking for product options or recommendations.",
                params={"query": user_message},
            )

        if self._is_smalltalk(lowered):
            return RouteDecision(
                route="smalltalk",
                confidence=0.98,
                reasoning="The message is a greeting, thanks, goodbye, or casual small talk.",
                params={"message": user_message},
            )

        if self._is_order_status(lowered):
            return RouteDecision(
                route="order_status",
                confidence=0.95,
                reasoning="The user is asking about an existing order or tracking status.",
                params={"message": user_message},
            )

        if self._is_preference_update(lowered):
            return RouteDecision(
                route="preference_update",
                confidence=0.94,
                reasoning="The user is providing recipient preferences or memory-worthy profile details.",
                params={"message": user_message},
            )

        if self._is_unclear(lowered):
            return RouteDecision(
                route="unclear",
                confidence=0.40,
                reasoning="The message is too vague to confidently map to a specialist.",
                params={"message": user_message},
            )

        return RouteDecision(
            route="catalog_search",
            confidence=0.84,
            reasoning="The message is best handled as a Kapruka gift or product search request.",
            params={"query": user_message},
        )

    def _route_stateless(self, user_message: str) -> Optional[RouteDecision]:
        lowered = user_message.lower()

        if self._is_user_name_question(lowered):
            return RouteDecision(
                route="identity",
                confidence=0.99,
                reasoning="The user is asking about their saved name.",
                params={"message": user_message},
            )

        if self._is_identity_question(lowered):
            return RouteDecision(
                route="identity",
                confidence=0.99,
                reasoning="The user is asking an identity or capability question.",
                params={"message": user_message},
            )

        if self._is_logistics_question(lowered):
            return RouteDecision(
                route="logistics_check",
                confidence=0.92,
                reasoning="The message explicitly mentions delivery or location logistics.",
                params={"message": user_message},
            )

        if self._is_product_search(lowered):
            return RouteDecision(
                route="catalog_search",
                confidence=0.90,
                reasoning="The user is asking for product options or recommendations.",
                params={"query": user_message},
            )

        if self._is_smalltalk(lowered):
            return RouteDecision(
                route="smalltalk",
                confidence=0.99,
                reasoning="The message is a greeting or self-introduction.",
                params={"message": user_message},
            )

        return None

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

    def _has_explicit_memory_intent(self, text: str) -> bool:
        return bool(
            re.search(
                r"\b(remember|note that|keep in mind|save this|save that|remember .*budget)\b",
                text,
            )
        )

    def _looks_like_note_worthy_profile_fact(self, text: str) -> bool:
        if "budget" in text:
            return True
        if re.search(r"\b(birthday|anniversary)\b", text) and re.search(r"\b(on|is|coming)\b", text):
            return True
        return False

    def _looks_like_recommendation_feedback(self, text: str) -> bool:
        if self._has_explicit_memory_intent(text):
            return False
        if not re.search(
            (
                r"\b("
                r"suit better|would suit|may suit|might suit|could suit|"
                r"better fit|better option|instead|"
                r"would prefer|prefer more|like more|love more|"
                r"more into|rather have|rather get"
                r")\b"
            ),
            text,
        ):
            return False
        return bool(
            re.search(
                r"\b(item|items|tool|tools|toolbox|tool box|electronic|electronics|gadget|gadgets|gift|gifts)\b",
                text,
            )
        )

    def _is_product_search(self, text: str) -> bool:
        return self._matches_any(text, PRODUCT_SEARCH_PATTERNS)

    def _is_logistics_question(self, text: str, memory_context: str = "") -> bool:
        if self._matches_any(text, LOGISTICS_PATTERNS):
            return True
        if self._mentions_known_delivery_area(text) and self._looks_like_location_statement(text):
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

    def _is_smalltalk(self, text: str) -> bool:
        if self._is_identity_question(text):
            return False
        if self._is_self_introduction(text):
            return True
        return self._matches_any(text, SMALLTALK_PATTERNS)

    def _is_identity_question(self, text: str) -> bool:
        return self._matches_any(text, IDENTITY_PATTERNS)

    def _is_user_name_question(self, text: str) -> bool:
        return self._matches_any(text, USER_NAME_PATTERNS)

    def _is_order_status(self, text: str) -> bool:
        return self._matches_any(text, ORDER_STATUS_PATTERNS)

    def _is_unclear(self, text: str) -> bool:
        return self._matches_any(text, UNCLEAR_PATTERNS)

    def _is_self_introduction(self, text: str) -> bool:
        return self._matches_any(text, SELF_INTRO_PATTERNS)

    def _mentions_known_delivery_area(self, text: str) -> bool:
        lowered = text.lower()
        for aliases in SRI_LANKAN_DISTRICTS.values():
            if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases):
                return True
        return False

    def _looks_like_location_statement(self, text: str) -> bool:
        if re.search(r"\b(i am|i'm|im|near|around|from|at|in|to)\b", text, flags=re.IGNORECASE):
            return True
        return len(text.split()) <= 4

    def _matches_any(self, text: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _split_items(self, text: str) -> List[str]:
        return [
            item.strip()
            for item in re.split(r",| and | but |;", text, flags=re.IGNORECASE)
            if item.strip()
        ]

    def _clean_preference_fragment(self, text: str) -> str:
        cleaned = re.sub(r"\b(for|her|him|them|only|can|eat|have|take)\b", " ", text, flags=re.IGNORECASE)
        cleaned = " ".join(cleaned.split()).strip(" .,:;-")
        if not cleaned:
            return ""
        return cleaned[:1].upper() + cleaned[1:]

    def _constraint_items(self, text: str) -> List[str]:
        food_terms = self._food_terms(text)
        if food_terms:
            return [term[:1].upper() + term[1:] for term in sorted(food_terms)]
        return [
            cleaned
            for item in self._split_items(text)
            if (cleaned := self._clean_preference_fragment(item))
        ]

    def _food_terms(self, text: str) -> set[str]:
        normalized = self._normalize_preference_text(text).lower()
        terms = set()
        for match in re.finditer(r"\b(?:dark|white|milk)?\s*chocolates?\b", normalized):
            term = " ".join(match.group(0).split()).replace("chocolates", "chocolate")
            terms.add(term)
        return terms

    def _normalize_preference_text(self, text: str) -> str:
        return (
            text.replace("chocaltes", "chocolates")
            .replace("chocalates", "chocolates")
            .replace("chocalte", "chocolate")
            .replace("chocalate", "chocolate")
            .replace("alergy", "allergy")
            .replace("alergic", "allergic")
        )

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
