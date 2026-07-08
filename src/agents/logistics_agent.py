"""Logistics specialist for Sri Lankan delivery feasibility checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from agents.prompts.agent_prompts import build_logistics_policy_prompt


def _load_districts() -> Dict[str, tuple[str, ...]]:
    data_path = Path(__file__).resolve().parent / "data" / "sri_lanka_districts.json"
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    return {
        item["name"]: tuple(item.get("aliases", []))
        for item in raw.get("districts", [])
    }


SRI_LANKAN_DISTRICTS: Dict[str, tuple[str, ...]] = _load_districts()

LOGISTICS_KEYWORDS = (
    "deliver",
    "delivery",
    "ship",
    "shipping",
    "send this",
    "send it",
    "same-day",
    "same day",
    "today",
    "tomorrow",
    "district",
    "location",
    "area",
    "reach",
    "arrive",
)


@dataclass
class LogisticsCheckResult:
    """Normalized logistics feasibility result."""

    district: Optional[str]
    supported: bool
    needs_manual_confirmation: bool
    urgency: str
    exact_guarantee_requested: bool
    summary: str


class LogisticsAgent:
    """Check delivery feasibility for Sri Lankan districts without pretending to own courier truth."""

    policy_prompt = build_logistics_policy_prompt()

    def check_delivery(self, message: str, memory_context: str = "") -> LogisticsCheckResult:
        district = self._extract_district(message)
        urgency = self._urgency_level(message)
        exact_guarantee_requested = self._exact_guarantee_requested(message)
        logistics_like = self._looks_like_logistics_question(message, memory_context=memory_context)
        live_confirmation_required = urgency != "standard" or exact_guarantee_requested

        if not logistics_like:
            return LogisticsCheckResult(
                district=None,
                supported=False,
                needs_manual_confirmation=False,
                urgency="none",
                exact_guarantee_requested=False,
                summary="The message is not clearly asking about delivery feasibility.",
            )

        if not district:
            return LogisticsCheckResult(
                district=None,
                supported=False,
                needs_manual_confirmation=True,
                urgency=urgency,
                exact_guarantee_requested=exact_guarantee_requested,
                summary=(
                    "I need the target Sri Lankan district before I can assess delivery feasibility. "
                    "Ask the user for the delivery district or city area."
                ),
            )

        if live_confirmation_required:
            return LogisticsCheckResult(
                district=district,
                supported=True,
                needs_manual_confirmation=True,
                urgency=urgency,
                exact_guarantee_requested=exact_guarantee_requested,
                summary=(
                    f"{district} is a recognized Sri Lankan delivery district. "
                    "Basic delivery looks feasible, but live confirmation is required for exact delivery guarantees, "
                    "including same-day, today, tomorrow, exact ETA, or guaranteed arrival."
                ),
            )

        return LogisticsCheckResult(
            district=district,
            supported=True,
            needs_manual_confirmation=False,
            urgency=urgency,
            exact_guarantee_requested=exact_guarantee_requested,
            summary=(
                f"{district} is a recognized Sri Lankan delivery district. "
                "At a basic routing level, delivery looks feasible. Exact timing, item availability, and guaranteed delivery still require live confirmation if the user asks for certainty."
            ),
        )

    def _extract_district(self, message: str) -> Optional[str]:
        lowered = message.lower()
        for district, aliases in SRI_LANKAN_DISTRICTS.items():
            if any(alias in lowered for alias in aliases):
                return district
        return None

    def _looks_like_logistics_question(self, message: str, memory_context: str = "") -> bool:
        lowered = message.lower()
        if any(keyword in lowered for keyword in LOGISTICS_KEYWORDS):
            return True

        if memory_context and any(word in memory_context.lower() for word in ("deliver", "delivery", "district", "arrival")):
            if self._extract_district(message):
                return True
            if re.search(r"\b(i am|i'm|im|near|around|from|at|in)\b", lowered):
                return True

        return False

    def _urgency_level(self, message: str) -> str:
        lowered = message.lower()
        if re.search(r"\bsame[- ]day|today|urgent|asap\b", lowered):
            return "same_day"
        if re.search(r"\btomorrow|next day\b", lowered):
            return "scheduled_soon"
        return "standard"

    def _exact_guarantee_requested(self, message: str) -> bool:
        lowered = message.lower()
        return bool(
            re.search(
                r"\b(exact|guarantee|guaranteed|confirm|confirmed|eta|arrival time|on time|definitely|surely)\b",
                lowered,
            )
        )
