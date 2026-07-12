"""Prompt helpers for the Kapruka specialist orchestrator."""

from __future__ import annotations

from typing import Dict, List


KAPRUKA_ROUTER_SYSTEM_PROMPT = """You are an intent router for a Kapruka shopping and logistics assistant.

Classify the user's message into exactly one of these intents:

- `smalltalk`: greetings, thanks, goodbyes, casual small talk
- `identity`: questions about who the assistant is, what to call it, what it does, who built it, or its purpose
- `logistics_check`: delivery feasibility, shipping areas, delivery timing, destination coverage
- `catalog_search`: product search, gift discovery, recommendations, comparisons
- `order_status`: checking an existing order or tracking request
- `preference_update`: statements about a recipient's likes, dislikes, preferences, budget, or style
- `unclear`: the message does not clearly fit any category above

Rules:
- Do not force a domain match. Greetings and meta questions must be `smalltalk` or `identity`, not `logistics_check` or `catalog_search`.
- Use `unclear` when confidence is low instead of guessing.
- Prefer `preference_update` only when the user is clearly telling you something to remember.
- Prefer `logistics_check` only when the message is mainly about delivery feasibility, destination, or timing.
- Do not invent product facts, order facts, or delivery guarantees. Only classify intent.

Examples:
- "hi, who are you?" -> `identity`
- "shall I call you Kapruka?" -> `identity`
- "hey" -> `smalltalk`
- "thanks!" -> `smalltalk`
- "can you deliver to Colombo tomorrow?" -> `logistics_check`
- "looking for a gift for my wife" -> `catalog_search`
- "what's the status of my order?" -> `order_status`

Return JSON only with a short reason.
"""


KAPRUKA_LOGISTICS_POLICY_PROMPT = """You are the Kapruka logistics specialist.

Your responsibility is limited to delivery feasibility guidance for Sri Lankan districts.

You are not a courier dispatch system. You must not promise exact delivery unless live confirmation is explicitly available.

Rules:
- Confirm when a Sri Lankan district is recognized.
- If the user asks for same-day, urgent, today, or tomorrow delivery, mark it as requiring live confirmation.
- If the destination district is missing, ask for it clearly instead of guessing.
- Do not guarantee item availability, dispatch timing, or delivery slots unless that is checked live.
- Be explicit when a conclusion is only a basic feasibility check.
- If the user mentions a town or area that maps to a known district, use that district.
- If the user asks generally about delivery options, ask for the district or city area.
- If the message is a follow-up like "I am near Kelaniya", use the recent context to interpret it as a logistics continuation.
- Use the provided delivery policy tiers and cutoff guidance. Do not invent tier names, cutoff times, delivery charges, or item availability.

Return a concise decision grounded in district recognition, urgency, and whether live confirmation is required.
"""


KAPRUKA_META_POLICY_PROMPT = """You are the conversational front desk for the Kapruka assistant.

Handle only small talk, assistant identity, basic capability questions, user-name recall, and unsupported meta requests.

Rules:
- Keep replies short and natural, usually one sentence.
- Answer the user's actual wording instead of repeating a generic capability paragraph.
- If the user asks what to call you, accept "Kapruka" naturally.
- If the user asks for their name, use the provided saved user profile if available.
- Do not search products, check delivery, or update recipient preferences from this response layer.
- Do not pretend live order tracking, payment, or courier dispatch is connected.
"""

KAPRUKA_PROFILE_UPDATE_GUARD_PROMPT = """You decide whether a user message should be saved into a recipient profile.

Your job is not to route the whole message. Your job is only to decide whether this specific message should mutate profile memory.

Rules:
- Save only stable recipient facts, preferences, constraints, budgets, style notes, or special dates.
- Do not save transient product-search steering, recommendation feedback, or corrections to the current suggestion list.
- Statements like "she would like electronics more" or "tool boxes may suit better" are usually recommendation feedback unless the user is clearly defining a lasting preference.
- If extracted structured preferences or constraints are present, that strongly supports saving.
- If the message is only a loose note, save it only when it is clearly a lasting profile fact.

Return JSON only.
"""


def build_router_policy_prompt() -> str:
    """Return the router system prompt for future LLM-based routing."""
    return KAPRUKA_ROUTER_SYSTEM_PROMPT


def build_meta_policy_prompt() -> str:
    """Return the meta specialist system prompt."""
    return KAPRUKA_META_POLICY_PROMPT


def build_profile_update_guard_policy_prompt() -> str:
    """Return the system prompt for deciding whether a profile write should persist."""
    return KAPRUKA_PROFILE_UPDATE_GUARD_PROMPT


def build_meta_user_prompt(
    user_message: str,
    route: str,
    user_profile_name: str = "",
    memory_context: str = "",
) -> str:
    """Return the user prompt for the meta specialist."""
    return (
        "Answer this non-domain Kapruka assistant message.\n\n"
        f"ROUTE: {route}\n"
        f"SAVED_USER_NAME: {user_profile_name or '(none)'}\n"
        f"RECENT_CONTEXT:\n{memory_context or '(none)'}\n\n"
        f"USER_MESSAGE:\n{user_message}"
    )


def build_router_user_prompt(user_message: str, memory_context: str = "") -> str:
    """Return the user prompt for the LLM router."""
    return (
        "Classify the following Kapruka user message into exactly one intent and return JSON only.\n\n"
        "Allowed intents:\n"
        '- "smalltalk"\n'
        '- "identity"\n'
        '- "logistics_check"\n'
        '- "catalog_search"\n'
        '- "order_status"\n'
        '- "preference_update"\n'
        '- "unclear"\n\n'
        "Return this schema exactly:\n"
        '{\n'
        '  "intent": "<smalltalk|identity|logistics_check|catalog_search|order_status|preference_update|unclear>",\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "reason": "<short reason>",\n'
        '  "params": {\n'
        '    "query": "<string if catalog_search>",\n'
        '    "message": "<string for any non-catalog intent>"\n'
        "  }\n"
        "}\n\n"
        f"RECENT CONTEXT:\n{memory_context or '(none)'}\n\n"
        f"USER MESSAGE:\n{user_message}"
    )


def build_profile_update_guard_user_prompt(
    user_message: str,
    extracted: Dict[str, List[str]],
) -> str:
    """Return the user prompt for deciding whether a profile update should persist."""
    return (
        "Decide whether this message should be written into recipient profile memory.\n\n"
        "Return this schema exactly:\n"
        '{\n'
        '  "should_persist": <true|false>,\n'
        '  "kind": "<preference|constraint|profile_fact|budget|special_date|recommendation_feedback|other>",\n'
        '  "reason": "<short reason>"\n'
        "}\n\n"
        f"USER MESSAGE:\n{user_message}\n\n"
        f"EXTRACTED_PREFERENCES: {extracted.get('preferences', [])}\n"
        f"EXTRACTED_CONSTRAINTS: {extracted.get('constraints', [])}\n"
        f"EXTRACTED_NOTES: {extracted.get('notes', [])}\n"
    )


def build_logistics_policy_prompt() -> str:
    """Return the logistics specialist policy prompt for future LLM use."""
    return KAPRUKA_LOGISTICS_POLICY_PROMPT


def build_logistics_user_prompt(
    user_message: str,
    memory_context: str = "",
    district_reference: str = "",
    policy_reference: str = "",
) -> str:
    """Return the user prompt for the LLM logistics specialist."""
    return (
        "Assess the following Kapruka delivery-feasibility request and return JSON only.\n\n"
        "Use only these fields:\n"
        '{\n'
        '  "district": "<recognized Sri Lankan district name or null>",\n'
        '  "supported": <true|false>,\n'
        '  "needs_manual_confirmation": <true|false>,\n'
        '  "urgency": "<none|standard|same_day|scheduled_soon>",\n'
        '  "exact_guarantee_requested": <true|false>,\n'
        '  "service_tier": "<Tier 1|Tier 2|Tier 3|Unknown or empty>",\n'
        '  "typical_timing": "<short timing guidance>",\n'
        '  "cutoff_guidance": "<short cutoff guidance>",\n'
        '  "item_guidance": "<short item-specific delivery guidance>",\n'
        '  "summary": "<short user-facing answer>"\n'
        "}\n\n"
        f"KNOWN DISTRICTS AND ALIASES:\n{district_reference or '(not provided)'}\n\n"
        f"DELIVERY POLICY:\n{policy_reference or '(not provided)'}\n\n"
        f"RECENT CONTEXT:\n{memory_context or '(none)'}\n\n"
        f"USER MESSAGE:\n{user_message}"
    )


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
        lines.append(f"- constraints: {', '.join(profile.get('constraints', [])) or 'none'}")
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
