"""Catalog specialist for RAG-based product search."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from infastructure.llm_providers.llm_services import OpenAIChatService
from memory.memory_ops import CognitiveMemoryStack


@dataclass
class CatalogAgentResult:
    """Catalog specialist output."""

    query: str
    bundle: Dict[str, object]
    memory_gate: Dict[str, object]
    reflection: Dict[str, object]
    answer: str
    timings_ms: Dict[str, int]


class CatalogAgent:
    """Use the 3-tier memory stack to retrieve and explain catalog results."""

    def __init__(
        self,
        memory_stack: CognitiveMemoryStack,
        chat_service: Optional[Any] = None,
    ) -> None:
        self.memory_stack = memory_stack
        self.chat_service = chat_service

    def search(
        self,
        query: str,
        user_id: str,
        session_id: str,
        recipient_id: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> CatalogAgentResult:
        retrieval_started_at = time.perf_counter()

        if progress_callback:
            progress_callback("Loading short-term memory...")
        st_started_at = time.perf_counter()
        recent_turns = self.memory_stack.recent_context(user_id=user_id, session_id=session_id)
        short_term_ms = int((time.perf_counter() - st_started_at) * 1000)

        recipient_profile = None
        semantic_ms = 0
        if recipient_id:
            if progress_callback:
                progress_callback("Loading recipient profile...")
            semantic_started_at = time.perf_counter()
            recipient_profile = self.memory_stack.get_recipient_profile(recipient_id)
            semantic_ms = int((time.perf_counter() - semantic_started_at) * 1000)

        if progress_callback:
            progress_callback("Evaluating memory relevance...")
        gate_started_at = time.perf_counter()
        memory_gate = self.memory_stack.decide_memory_reads(
            user_id=user_id,
            session_id=session_id,
            query=query,
            recipient_id=recipient_id,
            recent_turns=recent_turns,
            recipient_profile=recipient_profile,
        )
        gate_ms = int((time.perf_counter() - gate_started_at) * 1000)

        if progress_callback:
            progress_callback("Building retrieval query...")
        query_started_at = time.perf_counter()
        effective_query = self.memory_stack.build_retrieval_query(
            query=query,
            recent_turns=recent_turns,
            recipient_profile=recipient_profile,
            decision=memory_gate,
        )
        query_build_ms = int((time.perf_counter() - query_started_at) * 1000)

        if progress_callback:
            progress_callback("Searching catalog vectors...")
        vector_started_at = time.perf_counter()
        catalog_matches, vector_timings = self.memory_stack.long_term.search_detailed(
            query=effective_query,
            top_k=top_k,
            score_threshold=score_threshold,
            progress_callback=progress_callback,
        )
        vector_total_ms = int((time.perf_counter() - vector_started_at) * 1000)

        bundle = {
            "recent_turns": [turn.to_dict() for turn in recent_turns] if memory_gate.use_short_term else [],
            "catalog_matches": [
                {
                    "product_id": match.product_id,
                    "score": match.score,
                    "product": match.product.to_dict(),
                }
                for match in catalog_matches
            ],
            "recipient_profile": (
                recipient_profile.to_dict()
                if memory_gate.use_recipient_profile and recipient_profile
                else None
            ),
        }
        relevance_started_at = time.perf_counter()
        relevance_filter = self._enforce_direct_product_relevance(query=query, bundle=bundle)
        relevance_ms = int((time.perf_counter() - relevance_started_at) * 1000)
        bundle["product_relevance_filter"] = relevance_filter

        if progress_callback:
            progress_callback("Reflecting on gift safety and preference alignment...")
        reflection_started_at = time.perf_counter()
        reflection = self._reflect_and_revise(bundle)
        reflection_ms = int((time.perf_counter() - reflection_started_at) * 1000)
        retrieval_ms = int((time.perf_counter() - retrieval_started_at) * 1000)

        answer_started_at = time.perf_counter()
        answer = self._answer(query=query, bundle=bundle)
        answer_ms = int((time.perf_counter() - answer_started_at) * 1000)

        return CatalogAgentResult(
            query=query,
            bundle=bundle,
            memory_gate=memory_gate.to_dict(),
            reflection=reflection,
            answer=answer,
            timings_ms={
                "short_term_read": short_term_ms,
                "semantic_profile_read": semantic_ms,
                "memory_relevance_gate": gate_ms,
                "retrieval_query_build": query_build_ms,
                "vector_search_total": vector_total_ms,
                **vector_timings,
                "product_relevance_filter": relevance_ms,
                "reflection_loop": reflection_ms,
                "catalog_retrieval": retrieval_ms,
                "catalog_answer_generation": answer_ms,
            },
        )

    def _answer(self, query: str, bundle: Dict[str, object]) -> str:
        relevance_filter = bundle.get("product_relevance_filter")
        catalog_matches = bundle.get("catalog_matches", [])
        if (
            isinstance(relevance_filter, dict)
            and relevance_filter.get("applied")
            and catalog_matches
        ):
            return self._format_direct_product_answer(bundle=bundle)

        service = self.chat_service
        if service is None:
            try:
                service = OpenAIChatService()
            except Exception:
                service = None

        if service is not None:
            try:
                return service.answer_query(query=query, bundle=bundle)
            except Exception:
                pass

        if not catalog_matches:
            return "I could not find a strong product match in the current catalog."

        return self._format_direct_product_answer(bundle=bundle, title="Top matching products:")

    def _format_direct_product_answer(
        self,
        bundle: Dict[str, object],
        title: str = "Here are the best matching options I found:",
    ) -> str:
        catalog_matches = list(bundle.get("catalog_matches", []))
        if not catalog_matches:
            return "I could not find a strong product match in the current catalog."

        lines = ["Top matching products:"]
        for index, match in enumerate(catalog_matches[:3], 1):
            product = match["product"]
            description = self._short_description(str(product.get("description", "")))
            lines.append(f"{index}. {product['name']}")
            lines.append(f"Price: {product['price']}")
            lines.append(f"Availability: {product['availability']}")
            if description:
                lines.append(f"Description: {description}")
            lines.append(f"URL: {product['url']}")
            lines.append("")
        lines[0] = title
        return "\n".join(lines)

    def _short_description(self, description: str, max_chars: int = 220) -> str:
        cleaned = " ".join(description.split())
        cleaned = re.sub(r"\b(Get|Send|Online|Kapruka):?\s+", "", cleaned, flags=re.IGNORECASE)
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rsplit(" ", 1)[0].rstrip(" .,") + "."

    def _reflect_and_revise(self, bundle: Dict[str, object]) -> Dict[str, object]:
        """Remove catalog matches that violate recipient constraints before answer generation."""
        profile = bundle.get("recipient_profile") or {}
        constraints = profile.get("constraints", []) if isinstance(profile, dict) else []
        catalog_matches = list(bundle.get("catalog_matches", []))

        violations = []
        revised_matches = []
        for match in catalog_matches:
            product = match.get("product", {})
            violated_constraints = self._violated_constraints(product=product, constraints=constraints)
            if violated_constraints:
                violations.append(
                    {
                        "product_id": match.get("product_id", ""),
                        "product_name": product.get("name", ""),
                        "violated_constraints": violated_constraints,
                    }
                )
                continue
            revised_matches.append(match)

        if violations:
            bundle["catalog_matches"] = revised_matches

        return {
            "draft_count": len(catalog_matches),
            "revised_count": len(revised_matches),
            "violations": violations,
            "revised": bool(violations),
        }

    def _enforce_direct_product_relevance(self, query: str, bundle: Dict[str, object]) -> Dict[str, object]:
        """For explicit product-category queries, remove unrelated catalog matches."""
        category = self._requested_product_category(query)
        catalog_matches = list(bundle.get("catalog_matches", []))
        if not category:
            return {
                "applied": False,
                "category": "",
                "removed_count": 0,
                "removed_products": [],
            }

        kept = []
        removed = []
        for match in catalog_matches:
            product = match.get("product", {})
            product_text = " ".join(
                str(product.get(key, ""))
                for key in ("name", "description", "url")
            )
            if self._product_matches_category(product_text, category):
                kept.append(match)
            else:
                removed.append(product.get("name", ""))

        if kept:
            bundle["catalog_matches"] = kept

        return {
            "applied": True,
            "category": category,
            "removed_count": len(removed) if kept else 0,
            "removed_products": removed if kept else [],
        }

    def _requested_product_category(self, query: str) -> str:
        normalized = query.lower()
        if re.search(r"\b(led\s+tv|tv|tvs|television|televisions|smart\s+tv|qled|uhd)\b", normalized):
            return "tv"
        if re.search(r"\b(bluetooth\s+speaker|speaker|speakers|soundbar|sound\s+bar)\b", normalized):
            return "speaker"
        if re.search(r"\b(cake|cakes|bento\s+cake|gateau)\b", normalized):
            return "cake"
        if re.search(r"\b(flower|flowers|bouquet|rose|roses)\b", normalized):
            return "flowers"
        if re.search(r"\b(teddy|bear|soft\s*toy|softtoy)\b", normalized):
            return "teddy"
        if re.search(r"\b(chocolate|chocolates|ferrero)\b", normalized):
            return "chocolate"
        return ""

    def _product_matches_category(self, product_text: str, category: str) -> bool:
        normalized = product_text.lower()
        patterns = {
            "tv": r"\b(tv|television)\b",
            "speaker": r"\b(speaker|speakers|soundbar|sound\s+bar)\b",
            "cake": r"\b(cake|cakes|gateau)\b",
            "flowers": r"\b(flower|flowers|bouquet|rose|roses)\b",
            "teddy": r"\b(teddy|bear|soft\s*toy|softtoy)\b",
            "chocolate": r"\b(chocolate|chocolates|ferrero)\b",
        }
        pattern = patterns.get(category)
        return bool(pattern and re.search(pattern, normalized))

    def _violated_constraints(self, product: Dict[str, object], constraints: List[str]) -> List[str]:
        product_text = " ".join(
            str(product.get(key, ""))
            for key in ("name", "description")
        )
        product_terms = self._constraint_terms(product_text)
        if not product_terms:
            return []

        violated = []
        for constraint in constraints:
            constraint_terms = self._constraint_terms(str(constraint))
            if product_terms & constraint_terms:
                violated.append(str(constraint))
        return violated

    def _constraint_terms(self, text: str) -> set[str]:
        normalized = (
            text.lower()
            .replace("chocaltes", "chocolates")
            .replace("chocalates", "chocolates")
            .replace("chocalte", "chocolate")
            .replace("chocalate", "chocolate")
        )
        terms = set()
        for match in re.finditer(r"\b(?:dark|white|milk)?\s*chocolates?\b", normalized):
            term = " ".join(match.group(0).split()).replace("chocolates", "chocolate")
            terms.add(term)
        for match in re.finditer(r"\b(?:peanuts?|nuts?|gluten|dairy|egg|eggs|seafood|fish)\b", normalized):
            terms.add(match.group(0))
        return terms
