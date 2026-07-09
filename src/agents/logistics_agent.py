"""Logistics specialist for Sri Lankan delivery feasibility checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agents.prompts.agent_prompts import (
    build_logistics_policy_prompt,
    build_logistics_user_prompt,
)
from infastructure.config import OPENAI_API_KEY, OPENAI_CHAT_MAX_TOKENS, OPENAI_CHAT_MODEL

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenAI = None


def _load_districts() -> Dict[str, tuple[str, ...]]:
    data_path = Path(__file__).resolve().parent / "data" / "sri_lanka_districts.json"
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    return {
        item["name"]: tuple(item.get("aliases", []))
        for item in raw.get("districts", [])
    }


SRI_LANKAN_DISTRICTS: Dict[str, tuple[str, ...]] = _load_districts()


def _load_delivery_policy() -> Dict[str, Any]:
    data_path = Path(__file__).resolve().parent / "data" / "kapruka_delivery_policy.json"
    return json.loads(data_path.read_text(encoding="utf-8"))


KAPRUKA_DELIVERY_POLICY: Dict[str, Any] = _load_delivery_policy()

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
    service_tier: str = ""
    typical_timing: str = ""
    cutoff_guidance: str = ""
    item_guidance: str = ""


class LogisticsAgent:
    """LLM-first logistics specialist with deterministic Sri Lankan fallback logic."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model: str = OPENAI_CHAT_MODEL,
        max_tokens: int = OPENAI_CHAT_MAX_TOKENS,
        use_llm: bool = True,
    ) -> None:
        self.policy_prompt = build_logistics_policy_prompt()
        self.model = model
        self.max_tokens = max_tokens
        self._use_llm = use_llm

        if llm_client is not None:
            self.client = llm_client
        elif use_llm and OpenAI is not None and OPENAI_API_KEY:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.client = None

    def check_delivery(self, message: str, memory_context: str = "") -> LogisticsCheckResult:
        if self.client is not None and self._use_llm:
            try:
                return self._check_with_llm(message=message, memory_context=memory_context)
            except Exception:
                pass
        return self._check_with_rules(message=message, memory_context=memory_context)

    def _check_with_llm(self, message: str, memory_context: str = "") -> LogisticsCheckResult:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=min(self.max_tokens, 250),
            messages=[
                {"role": "system", "content": self.policy_prompt},
                {
                    "role": "user",
                    "content": build_logistics_user_prompt(
                        user_message=message,
                        memory_context=memory_context,
                        district_reference=self._district_reference_text(),
                        policy_reference=self._policy_reference_text(),
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return self._parse_llm_response(content)

    def _parse_llm_response(self, content: str) -> LogisticsCheckResult:
        data = json.loads(content)
        district_raw = data.get("district")
        district = str(district_raw).strip() if district_raw not in (None, "", "null") else None

        urgency = str(data.get("urgency", "standard")).strip() or "standard"
        if urgency not in {"none", "standard", "same_day", "scheduled_soon"}:
            urgency = "standard"

        return LogisticsCheckResult(
            district=district,
            supported=bool(data.get("supported", False)),
            needs_manual_confirmation=bool(data.get("needs_manual_confirmation", False)),
            urgency=urgency,
            exact_guarantee_requested=bool(data.get("exact_guarantee_requested", False)),
            summary=str(data.get("summary", "")).strip() or "Delivery feasibility requires clarification.",
            service_tier=str(data.get("service_tier", "")).strip(),
            typical_timing=str(data.get("typical_timing", "")).strip(),
            cutoff_guidance=str(data.get("cutoff_guidance", "")).strip(),
            item_guidance=str(data.get("item_guidance", "")).strip(),
        )

    def _check_with_rules(self, message: str, memory_context: str = "") -> LogisticsCheckResult:
        district = self._extract_district(message)
        urgency = self._urgency_level(message)
        exact_guarantee_requested = self._exact_guarantee_requested(message)
        logistics_like = self._looks_like_logistics_question(message, memory_context=memory_context)
        delivery_mode = self._delivery_mode(message)
        item_type = self._item_type(message)

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
                    "Ask for the delivery district or city area."
                ),
            )

        tier = self._tier_for_district(district)
        needs_manual_confirmation = self._needs_manual_confirmation(
            urgency=urgency,
            delivery_mode=delivery_mode,
            exact_guarantee_requested=exact_guarantee_requested,
            tier=tier,
            item_type=item_type,
        )
        item_guidance = self._item_guidance(tier=tier, item_type=item_type)
        summary = self._summary(
            district=district,
            tier=tier,
            urgency=urgency,
            delivery_mode=delivery_mode,
            item_type=item_type,
            exact_guarantee_requested=exact_guarantee_requested,
            needs_manual_confirmation=needs_manual_confirmation,
            item_guidance=item_guidance,
        )

        return LogisticsCheckResult(
            district=district,
            supported=True,
            needs_manual_confirmation=needs_manual_confirmation,
            urgency=urgency,
            exact_guarantee_requested=exact_guarantee_requested,
            summary=summary,
            service_tier=tier.get("name", ""),
            typical_timing=tier.get("timing", ""),
            cutoff_guidance=tier.get("cutoff_guidance", ""),
            item_guidance=item_guidance,
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

    def _delivery_mode(self, message: str) -> str:
        lowered = message.lower()
        if re.search(r"\bmidnight|specific[- ]time|exact time|time slot\b", lowered):
            return "specific_time"
        if re.search(r"\bschedule|scheduled|birthday|anniversary|on \d{1,2}(st|nd|rd|th)?\b", lowered):
            return "scheduled"
        if re.search(r"\bsame[- ]day|today|urgent|asap\b", lowered):
            return "same_day"
        return "standard"

    def _item_type(self, message: str) -> str:
        lowered = message.lower()
        sensitive_keywords = KAPRUKA_DELIVERY_POLICY.get("sensitive_item_keywords", [])
        retail_keywords = KAPRUKA_DELIVERY_POLICY.get("retail_item_keywords", [])
        if any(keyword in lowered for keyword in sensitive_keywords):
            return "sensitive"
        if any(keyword in lowered for keyword in retail_keywords):
            return "retail"
        return "unspecified"

    def _exact_guarantee_requested(self, message: str) -> bool:
        lowered = message.lower()
        return bool(
            re.search(
                r"\b(exact|guarantee|guaranteed|confirm|confirmed|eta|arrival time|on time|definitely|surely)\b",
                lowered,
            )
        )

    def _tier_for_district(self, district: str) -> Dict[str, Any]:
        for tier in KAPRUKA_DELIVERY_POLICY.get("tiers", []):
            if district in tier.get("districts", []):
                return tier
        return {
            "name": "Unknown",
            "label": "Recognized district",
            "timing": "Delivery timing requires checkout confirmation.",
            "same_day_guidance": "Same-day should not be assumed without live confirmation.",
            "cutoff_guidance": "Cutoff depends on item and checkout routing.",
            "perishable_guidance": "Perishable availability requires item-level confirmation.",
        }

    def _needs_manual_confirmation(
        self,
        urgency: str,
        delivery_mode: str,
        exact_guarantee_requested: bool,
        tier: Dict[str, Any],
        item_type: str,
    ) -> bool:
        if exact_guarantee_requested:
            return True
        if delivery_mode in {"specific_time", "scheduled"}:
            return True
        if urgency in {"same_day", "scheduled_soon"}:
            return True
        if tier.get("name") != "Tier 1" and item_type == "sensitive":
            return True
        return False

    def _item_guidance(self, tier: Dict[str, Any], item_type: str) -> str:
        if item_type == "sensitive":
            return tier.get("perishable_guidance", "Sensitive items require item-level confirmation.")
        if item_type == "retail":
            return "Packaged retail items are usually better suited to standard courier delivery."
        return "Item-specific availability is checked at checkout."

    def _summary(
        self,
        district: str,
        tier: Dict[str, Any],
        urgency: str,
        delivery_mode: str,
        item_type: str,
        exact_guarantee_requested: bool,
        needs_manual_confirmation: bool,
        item_guidance: str,
    ) -> str:
        tier_name = tier.get("name", "")
        tier_label = tier.get("label", "").lower()
        timing = tier.get("timing", "Delivery timing requires checkout confirmation.")
        cutoff = tier.get("cutoff_guidance", "")

        parts = [
            f"{district} is covered under {tier_name} ({tier_label}).",
            timing,
        ]

        if delivery_mode == "same_day" or urgency == "same_day":
            parts.append(tier.get("same_day_guidance", "Same-day delivery requires live confirmation."))
            if cutoff:
                parts.append(cutoff)
        elif delivery_mode == "scheduled":
            parts.append("Scheduled delivery is usually handled by selecting the delivery date at checkout, subject to item and address support.")
        elif delivery_mode == "specific_time":
            parts.append("Midnight or exact-time delivery needs live coordination and may carry a surcharge.")

        if item_type != "unspecified":
            parts.append(item_guidance)

        if exact_guarantee_requested or needs_manual_confirmation:
            parts.append("Use live checkout or hotline confirmation before promising the exact slot, charge, or availability.")
        else:
            parts.append("Delivery charges and final availability are still calculated at checkout.")

        return " ".join(part for part in parts if part)

    def _district_reference_text(self) -> str:
        lines = []
        for district, aliases in SRI_LANKAN_DISTRICTS.items():
            lines.append(f"- {district}: {', '.join(aliases)}")
        return "\n".join(lines)

    def _policy_reference_text(self) -> str:
        lines = ["Delivery options:"]
        for option in KAPRUKA_DELIVERY_POLICY.get("delivery_options", []):
            lines.append(
                f"- {option['name']}: {option['target_areas']} | {option['timing']} | best for {option['best_used_for']}"
            )

        lines.append("District tiers:")
        for tier in KAPRUKA_DELIVERY_POLICY.get("tiers", []):
            lines.append(
                f"- {tier['name']} ({tier['label']}): {', '.join(tier['districts'])}. "
                f"{tier['timing']} {tier['same_day_guidance']} {tier['cutoff_guidance']} {tier['perishable_guidance']}"
            )

        lines.append("General policies:")
        for policy in KAPRUKA_DELIVERY_POLICY.get("general_policies", []):
            lines.append(f"- {policy}")
        return "\n".join(lines)
