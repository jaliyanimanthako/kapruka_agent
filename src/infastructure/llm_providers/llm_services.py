"""OpenAI chat service for the Kapruka memory-assisted gift advisor."""

from __future__ import annotations

import json
from typing import Dict, List

from infastructure.config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MAX_TOKENS,
    OPENAI_CHAT_MODEL,
    OPENAI_CHAT_TEMPERATURE,
)

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenAI = None


SYSTEM_PROMPT = """You are a gift recommendation assistant for Kapruka.
Use the provided memory sections carefully:
- SHORT_TERM_MEMORY: recent conversation context
- SEMANTIC_PROFILE: recipient preferences and notes
- LONG_TERM_CATALOG: retrieved products from the catalog

Rules:
- USER_QUERY has the highest priority. If it asks for a specific product type or category, follow that first.
- Never infer that the user wants to avoid a product category just because USER_QUERY names that category.
- Only mention avoidance, allergies, or exclusions when SEMANTIC_PROFILE explicitly contains that constraint.
- Base recommendations only on the provided catalog matches.
- If profile preferences exist, use them only when they do not conflict with the current USER_QUERY.
- Treat profile constraints, allergies, and avoids as hard exclusions when explaining recommendations.
- Explain briefly why each suggestion matches the user and recipient context.
- If the catalog matches are weak or unrelated, say that clearly.
- Do not claim the catalog lacks a product type if relevant catalog matches are present.
"""

ANSWER_REVIEW_SYSTEM_PROMPT = """You validate whether a draft answer is relevant to a user's shopping request.

Rules:
- Judge only against USER_QUERY, DRAFT_ANSWER, and LONG_TERM_CATALOG.
- Every product recommendation in DRAFT_ANSWER must refer to a product that appears in LONG_TERM_CATALOG.
- For broad gift-discovery queries like gifts or presents for a person, semantic suitability is acceptable even without literal keyword overlap.
- For specific product/category queries, require a direct category fit. Do not approve weak lifestyle justifications for unrelated products.
- If the answer recommends unrelated products, mark relevant=false.
- Return JSON only.
"""


class OpenAIChatService:
    """Thin wrapper around the OpenAI chat completions API."""

    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_CHAT_MODEL,
        temperature: float = OPENAI_CHAT_TEMPERATURE,
        max_tokens: int = OPENAI_CHAT_MAX_TOKENS,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for chat completions.")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer_query(self, query: str, bundle: Dict[str, object]) -> str:
        """Generate an answer from the query and memory bundle."""
        prompt = build_memory_prompt(query=query, bundle=bundle)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def judge_answer_relevance(self, query: str, bundle: Dict[str, object], answer: str) -> Dict[str, object]:
        """Judge whether a draft answer is relevant to the user's request."""
        prompt = build_answer_review_prompt(query=query, bundle=bundle, answer=answer)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=min(self.max_tokens, 300),
            messages=[
                {"role": "system", "content": ANSWER_REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return _normalize_answer_review_result(_parse_json_response(content))


def build_memory_prompt(query: str, bundle: Dict[str, object]) -> str:
    """Render the three memory layers into one user prompt."""
    recent_turns = bundle.get("recent_turns", [])
    recipient_profile = bundle.get("recipient_profile")
    catalog_matches = bundle.get("catalog_matches", [])

    lines: List[str] = [f"USER_QUERY:\n{query}", ""]

    lines.append("SHORT_TERM_MEMORY:")
    if recent_turns:
        for turn in recent_turns:
            lines.append(f"- {turn['role']}: {turn['content']}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("SEMANTIC_PROFILE:")
    if recipient_profile:
        lines.append(f"- name: {recipient_profile.get('name', '')}")
        lines.append(f"- relationship: {recipient_profile.get('relationship', '')}")
        preferences = recipient_profile.get("preferences", [])
        constraints = recipient_profile.get("constraints", [])
        notes = recipient_profile.get("notes", [])
        lines.append(f"- preferences: {', '.join(preferences) if preferences else 'none'}")
        lines.append(f"- constraints: {', '.join(constraints) if constraints else 'none'}")
        lines.append(f"- notes: {', '.join(notes) if notes else 'none'}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("LONG_TERM_CATALOG:")
    if catalog_matches:
        for index, match in enumerate(catalog_matches, 1):
            product = match["product"]
            lines.append(
                f"{index}. {product['name']} | score={match['score']:.4f} | "
                f"price={product['price']} | availability={product['availability']}"
            )
            lines.append(f"   description: {product['description']}")
            lines.append(f"   url: {product['url']}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append(
        "TASK:\n"
        "Recommend the best 3 products from LONG_TERM_CATALOG for the USER_QUERY. "
        "Use SHORT_TERM_MEMORY and SEMANTIC_PROFILE to personalize the answer."
    )
    return "\n".join(lines)


def build_answer_review_prompt(query: str, bundle: Dict[str, object], answer: str) -> str:
    """Render the answer-review prompt for the relevance judge."""
    catalog_matches = bundle.get("catalog_matches", [])

    lines: List[str] = [f"USER_QUERY:\n{query}", ""]
    lines.append("LONG_TERM_CATALOG:")
    if catalog_matches:
        for index, match in enumerate(catalog_matches, 1):
            product = match["product"]
            lines.append(
                f"{index}. {product['name']} | score={match['score']:.4f} | "
                f"price={product['price']} | availability={product['availability']}"
            )
            lines.append(f"   description: {product['description']}")
            lines.append(f"   url: {product['url']}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append(f"DRAFT_ANSWER:\n{answer}")
    lines.append("")
    lines.append(
        "Return strict JSON with this schema:\n"
        '{\n'
        '  "relevant": true,\n'
        '  "confidence": 0.0,\n'
        '  "reason": "short explanation",\n'
        '  "supported_products": ["exact product names from LONG_TERM_CATALOG"],\n'
        '  "unsupported_products": ["exact product names from DRAFT_ANSWER that are not relevant"],\n'
        '  "mentioned_products": ["exact product names from DRAFT_ANSWER that appear in LONG_TERM_CATALOG"]\n'
        '}'
    )
    return "\n".join(lines)


def _parse_json_response(content: str) -> Dict[str, object]:
    """Parse a JSON object from a model response, including fenced JSON."""
    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from the answer review model.")
    return parsed


def _normalize_answer_review_result(result: Dict[str, object]) -> Dict[str, object]:
    """Normalize the answer-review result into the expected shape."""
    def _string_list(value: object) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()]

    confidence = result.get("confidence", 0.0)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0

    return {
        "relevant": bool(result.get("relevant", False)),
        "confidence": max(0.0, min(1.0, confidence_value)),
        "reason": str(result.get("reason", "")).strip(),
        "supported_products": _string_list(result.get("supported_products")),
        "unsupported_products": _string_list(result.get("unsupported_products")),
        "mentioned_products": _string_list(result.get("mentioned_products")),
    }
