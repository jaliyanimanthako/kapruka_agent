"""Meta specialist for small talk and assistant identity responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from agents.prompts.agent_prompts import build_meta_policy_prompt, build_meta_user_prompt
from infastructure.config import OPENAI_API_KEY, OPENAI_CHAT_MAX_TOKENS, OPENAI_CHAT_MODEL

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenAI = None


@dataclass
class MetaAgentResult:
    """Response from the meta specialist."""

    answer: str
    source: str


class MetaAgent:
    """Answer non-domain conversational messages without catalog or logistics work."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model: str = OPENAI_CHAT_MODEL,
        max_tokens: int = OPENAI_CHAT_MAX_TOKENS,
        use_llm: bool = True,
    ) -> None:
        self.system_prompt = build_meta_policy_prompt()
        self.model = model
        self.max_tokens = max_tokens
        self._use_llm = use_llm

        if llm_client is not None:
            self.client = llm_client
        elif use_llm and OpenAI is not None and OPENAI_API_KEY:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.client = None

    def answer(
        self,
        user_message: str,
        route: str,
        user_profile_name: str = "",
        memory_context: str = "",
    ) -> MetaAgentResult:
        if self.client is not None and self._use_llm:
            try:
                return MetaAgentResult(
                    answer=self._answer_with_llm(
                        user_message=user_message,
                        route=route,
                        user_profile_name=user_profile_name,
                        memory_context=memory_context,
                    ),
                    source="llm",
                )
            except Exception:
                pass

        return MetaAgentResult(
            answer=self._answer_with_rules(
                user_message=user_message,
                route=route,
                user_profile_name=user_profile_name,
            ),
            source="fallback",
        )

    def _answer_with_llm(
        self,
        user_message: str,
        route: str,
        user_profile_name: str = "",
        memory_context: str = "",
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            max_tokens=min(self.max_tokens, 120),
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": build_meta_user_prompt(
                        user_message=user_message,
                        route=route,
                        user_profile_name=user_profile_name,
                        memory_context=memory_context,
                    ),
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def _answer_with_rules(self, user_message: str, route: str, user_profile_name: str = "") -> str:
        lowered = user_message.lower()

        if self._asks_user_name(lowered):
            if user_profile_name:
                return f"Your name is {user_profile_name}."
            return "You have not told me your name yet."

        call_name = self._extract_assistant_call_name(user_message)
        if call_name:
            return f"Yes, you can call me {call_name}."

        if route == "smalltalk":
            if any(token in lowered for token in ("thanks", "thank you", "thx")):
                return "You're welcome."
            if any(token in lowered for token in ("bye", "goodbye", "see you")):
                return "See you."
            if any(token in lowered for token in ("hi", "hello", "hey", "good morning", "good afternoon", "good evening")):
                if user_profile_name:
                    return f"Hi, {user_profile_name}. How can I help?"
                return "Hi. How can I help?"
            return "How can I help?"

        if "what can you do" in lowered or "what do you do" in lowered:
            return "I can help find Kapruka products, remember gift preferences, and check delivery feasibility."

        if "who built you" in lowered:
            return "This project was built as a Kapruka assistant demo."

        return "I am the Kapruka assistant."

    def _extract_assistant_call_name(self, user_message: str) -> str:
        patterns = (
            r"\bcall you\s+(?P<name>[A-Za-z][A-Za-z0-9_-]{1,30})\b",
            r"\bname you\s+(?P<name>[A-Za-z][A-Za-z0-9_-]{1,30})\b",
        )
        for pattern in patterns:
            match = re.search(pattern, user_message, flags=re.IGNORECASE)
            if match:
                raw_name = match.group("name").strip()
                return raw_name[:1].upper() + raw_name[1:]
        return ""

    def _asks_user_name(self, lowered: str) -> bool:
        return bool(
            re.search(
                r"\b(what is my name|what's my name|do you know my name|remember my name)\b",
                lowered,
            )
        )
