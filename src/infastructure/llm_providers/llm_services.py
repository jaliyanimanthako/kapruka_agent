"""OpenAI chat service for the Kapruka memory-assisted gift advisor."""

from __future__ import annotations

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
- Base recommendations only on the provided catalog matches.
- If profile preferences exist, use them only when they do not conflict with the current USER_QUERY.
- Treat profile constraints, allergies, and avoids as hard exclusions when explaining recommendations.
- Explain briefly why each suggestion matches the user and recipient context.
- If the catalog matches are weak or unrelated, say that clearly.
- Do not claim the catalog lacks a product type if relevant catalog matches are present.
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
