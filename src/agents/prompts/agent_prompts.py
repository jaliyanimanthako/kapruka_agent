"""Prompt helpers for the Kapruka specialist orchestrator."""

from __future__ import annotations

from typing import Dict


KAPRUKA_ROUTER_SYSTEM_PROMPT = """You are a query router for Kapruka, a gift discovery and delivery assistant.

Your job is to classify each user message into exactly one route:

1. `catalog_search`
Use this when the user wants product suggestions, gift ideas, comparisons, recommendations, or wants to find a product from the Kapruka catalog.

2. `preference_update`
Use this when the user is telling you something to remember about the recipient, such as likes, dislikes, preferred colors, budget, relationship context, or style preferences.

3. `logistics_check`
Use this when the user asks about delivery feasibility, destination, same-day delivery, district coverage, timing, or whether a specific place in Sri Lanka can be served.

Routing rules:
- Prefer `preference_update` when the user is clearly storing a preference or note.
- Prefer `logistics_check` when the question is mainly about where or when an item can be delivered.
- Prefer `catalog_search` for everything else related to gifts or products.
- Do not invent product facts or delivery guarantees. Only route the request.

Return a concise decision with a short reason.
"""


KAPRUKA_LOGISTICS_POLICY_PROMPT = """You are the Kapruka logistics specialist.

Your responsibility is limited to delivery feasibility guidance for Sri Lankan districts.

Rules:
- Confirm when a Sri Lankan district is recognized.
- If the user asks for same-day, urgent, today, or tomorrow delivery, mark it as requiring live confirmation.
- If the destination district is missing, ask for it clearly instead of guessing.
- Do not guarantee item availability, dispatch timing, or delivery slots unless that is checked live.
- Be explicit when a conclusion is only a basic feasibility check.
"""


def build_router_policy_prompt() -> str:
    """Return the router system prompt for future LLM-based routing."""
    return KAPRUKA_ROUTER_SYSTEM_PROMPT


def build_router_user_prompt(user_message: str, memory_context: str = "") -> str:
    """Return the user prompt for the LLM router."""
    return (
        "Classify the following Kapruka user message into exactly one route and return JSON only.\n\n"
        "Allowed routes:\n"
        '- "catalog_search"\n'
        '- "preference_update"\n'
        '- "logistics_check"\n\n'
        "Return this schema exactly:\n"
        '{\n'
        '  "route": "<catalog_search|preference_update|logistics_check>",\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "reasoning": "<short reason>",\n'
        '  "params": {\n'
        '    "query": "<string if catalog_search>",\n'
        '    "message": "<string if preference_update or logistics_check>"\n'
        "  }\n"
        "}\n\n"
        f"RECENT CONTEXT:\n{memory_context or '(none)'}\n\n"
        f"USER MESSAGE:\n{user_message}"
    )


def build_logistics_policy_prompt() -> str:
    """Return the logistics specialist policy prompt for future LLM use."""
    return KAPRUKA_LOGISTICS_POLICY_PROMPT


def build_catalog_debug_prompt(query: str, bundle: Dict[str, object]) -> str:
    """Return a compact debug prompt for inspecting catalog search context."""
    recent_turns = bundle.get("recent_turns", [])
    profile = bundle.get("recipient_profile")
    catalog_matches = bundle.get("catalog_matches", [])

    lines = [f"QUERY: {query}", "", "RECENT TURNS:"]
    if recent_turns:
        for turn in recent_turns:
            lines.append(f"- {turn['role']}: {turn['content']}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("RECIPIENT PROFILE:")
    if profile:
        lines.append(f"- name: {profile.get('name', '')}")
        lines.append(f"- relationship: {profile.get('relationship', '')}")
        lines.append(f"- preferences: {', '.join(profile.get('preferences', [])) or 'none'}")
        lines.append(f"- notes: {', '.join(profile.get('notes', [])) or 'none'}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("CATALOG MATCHES:")
    if catalog_matches:
        for match in catalog_matches:
            product = match["product"]
            lines.append(
                f"- {product['name']} | score={match['score']:.4f} | "
                f"price={product['price']} | availability={product['availability']}"
            )
    else:
        lines.append("- none")

    return "\n".join(lines)
