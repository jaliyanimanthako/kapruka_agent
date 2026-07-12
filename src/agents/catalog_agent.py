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

    ANSWER_REVIEW_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "available",
        "birthday",
        "buy",
        "can",
        "catalog",
        "do",
        "father",
        "find",
        "for",
        "gift",
        "gifts",
        "have",
        "hello",
        "help",
        "hi",
        "i",
        "in",
        "interest",
        "interested",
        "is",
        "it",
        "looking",
        "me",
        "my",
        "of",
        "on",
        "option",
        "options",
        "please",
        "price",
        "prices",
        "present",
        "presents",
        "recommend",
        "romantic",
        "searching",
        "show",
        "some",
        "suggest",
        "surprise",
        "that",
        "the",
        "these",
        "this",
        "to",
        "want",
        "what",
        "wife",
        "with",
        "you",
        "your",
        "okay",
        "ok",
        "also",
        "need",
        "suggest",
    }

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
        answer_reflection_started_at = time.perf_counter()
        reflection = self._reflect_on_answer(query=query, bundle=bundle, reflection=reflection, answer=answer)
        answer = str(reflection.get("final_answer", answer))
        answer_reflection_ms = int((time.perf_counter() - answer_reflection_started_at) * 1000)

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
                "answer_reflection_loop": answer_reflection_ms,
            },
        )

    def _answer(self, query: str, bundle: Dict[str, object]) -> str:
        relevance_filter = bundle.get("product_relevance_filter")
        catalog_matches = bundle.get("catalog_matches", [])
        if not catalog_matches:
            return "I could not find a strong product match in the current catalog."
        if (
            isinstance(relevance_filter, dict)
            and relevance_filter.get("applied")
            and catalog_matches
        ):
            return self._format_direct_product_answer(bundle=bundle)

        service = self._get_chat_service()
        if service is not None:
            try:
                return service.answer_query(query=query, bundle=bundle)
            except Exception:
                pass

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

    def _reflect_on_answer(
        self,
        query: str,
        bundle: Dict[str, object],
        reflection: Dict[str, object],
        answer: str,
    ) -> Dict[str, object]:
        """Validate that the final answer recommends products directly supported by the request."""
        reviewed_answer = answer or ""
        validation = self._judge_answer_relevance(query=query, bundle=bundle, answer=reviewed_answer)
        if validation is None:
            validation = self._heuristic_answer_validation(query=query, bundle=bundle, answer=reviewed_answer)

        reviewed_answer = self._apply_answer_validation(bundle=bundle, answer=reviewed_answer, validation=validation)
        reflection["answer_validation"] = validation
        reflection["final_answer"] = reviewed_answer
        return reflection

    def _judge_answer_relevance(
        self,
        query: str,
        bundle: Dict[str, object],
        answer: str,
    ) -> Optional[Dict[str, object]]:
        service = self._get_chat_service()
        if service is None or not hasattr(service, "judge_answer_relevance"):
            return None

        try:
            result = service.judge_answer_relevance(query=query, bundle=bundle, answer=answer)
        except Exception:
            return None

        if not isinstance(result, dict):
            return None

        supported_products = [str(name) for name in result.get("supported_products", [])]
        unsupported_products = [str(name) for name in result.get("unsupported_products", [])]
        mentioned_products = [str(name) for name in result.get("mentioned_products", [])]
        if not mentioned_products:
            mentioned_products = supported_products + [
                name for name in unsupported_products if name not in supported_products
            ]

        relevant = bool(result.get("relevant", False))
        return {
            "checked": True,
            "source": "llm_judge",
            "reason": str(result.get("reason", "")).strip() or "The LLM judge reviewed the draft answer.",
            "confidence": float(result.get("confidence", 0.0) or 0.0),
            "requested_terms": self._requested_answer_terms(query),
            "mentioned_products": mentioned_products,
            "supported_products": supported_products,
            "unsupported_products": unsupported_products,
            "answer_revised": not relevant,
            "answer_supported": relevant,
        }

    def _heuristic_answer_validation(
        self,
        query: str,
        bundle: Dict[str, object],
        answer: str,
    ) -> Dict[str, object]:
        requested_terms = self._requested_answer_terms(query)
        if not self._should_validate_answer_relevance(query):
            return {
                "checked": False,
                "source": "heuristic_fallback",
                "reason": "The query is broad gift discovery, so strict answer-term validation is skipped.",
                "confidence": 0.0,
                "requested_terms": requested_terms,
                "mentioned_products": [],
                "supported_products": [],
                "unsupported_products": [],
                "answer_revised": False,
                "answer_supported": True,
            }
        if not requested_terms:
            return {
                "checked": False,
                "source": "heuristic_fallback",
                "reason": "No strict product terms were detected in the query.",
                "confidence": 0.0,
                "requested_terms": [],
                "mentioned_products": [],
                "supported_products": [],
                "unsupported_products": [],
                "answer_revised": False,
                "answer_supported": True,
            }

        catalog_matches = list(bundle.get("catalog_matches", []))
        mentioned_products = self._mentioned_catalog_products(answer=answer, catalog_matches=catalog_matches)
        if not mentioned_products:
            return {
                "checked": True,
                "source": "heuristic_fallback",
                "reason": "The answer did not explicitly mention catalog product names.",
                "confidence": 0.0,
                "requested_terms": requested_terms,
                "mentioned_products": [],
                "supported_products": [],
                "unsupported_products": [],
                "answer_revised": False,
                "answer_supported": True,
            }

        category = ""
        relevance_filter = bundle.get("product_relevance_filter")
        if isinstance(relevance_filter, dict):
            category = str(relevance_filter.get("category", ""))

        supported_products = []
        unsupported_products = []
        for match in catalog_matches:
            product = match.get("product", {})
            product_name = str(product.get("name", ""))
            if product_name not in mentioned_products:
                continue
            product_text = " ".join(
                str(product.get(key, ""))
                for key in ("name", "description", "url")
            )
            if self._product_supports_answer_terms(product_text, requested_terms, category):
                supported_products.append(product_name)
            else:
                unsupported_products.append(product_name)

        revised = bool(unsupported_products)
        return {
            "checked": True,
            "source": "heuristic_fallback",
            "reason": (
                "The answer mentioned products that do not match the requested terms."
                if revised
                else "The answer stayed aligned with the requested terms."
            ),
            "confidence": 0.0,
            "requested_terms": requested_terms,
            "mentioned_products": mentioned_products,
            "supported_products": supported_products,
            "unsupported_products": unsupported_products,
            "answer_revised": revised,
            "answer_supported": not revised,
        }

    def _apply_answer_validation(
        self,
        bundle: Dict[str, object],
        answer: str,
        validation: Dict[str, object],
    ) -> str:
        if not validation.get("checked") or validation.get("answer_supported"):
            return answer

        supported_products = {
            str(name)
            for name in validation.get("supported_products", [])
            if str(name).strip()
        }
        if supported_products:
            bundle["catalog_matches"] = [
                match
                for match in list(bundle.get("catalog_matches", []))
                if str(match.get("product", {}).get("name", "")) in supported_products
            ]
        else:
            bundle["catalog_matches"] = []

        if bundle.get("catalog_matches"):
            return self._format_direct_product_answer(bundle=bundle)
        return "I could not find a strong product match in the current catalog."

    def _get_chat_service(self) -> Optional[Any]:
        service = self.chat_service
        if service is not None:
            return service
        try:
            return OpenAIChatService()
        except Exception:
            return None

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

        bundle["catalog_matches"] = kept

        return {
            "applied": True,
            "category": category,
            "removed_count": len(removed),
            "removed_products": removed,
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

    def _requested_answer_terms(self, query: str) -> List[str]:
        requested_terms = []
        seen = set()
        for token in re.findall(r"[a-z0-9]+", query.lower()):
            normalized = token[:-1] if token.endswith("s") and len(token) > 3 else token
            if normalized in self.ANSWER_REVIEW_STOPWORDS or len(normalized) <= 1 or normalized in seen:
                continue
            seen.add(normalized)
            requested_terms.append(normalized)
        return requested_terms

    def _mentioned_catalog_products(self, answer: str, catalog_matches: List[Dict[str, object]]) -> List[str]:
        mentioned = []
        answer_lower = answer.lower()
        for match in catalog_matches:
            product = match.get("product", {})
            product_name = str(product.get("name", ""))
            if product_name and product_name.lower() in answer_lower:
                mentioned.append(product_name)
        return mentioned

    def _product_supports_answer_terms(
        self,
        product_text: str,
        requested_terms: List[str],
        category: str,
    ) -> bool:
        if category and self._product_matches_category(product_text, category):
            return True

        normalized = product_text.lower()
        return any(term in normalized for term in requested_terms)

    def _should_validate_answer_relevance(self, query: str) -> bool:
        if self.memory_stack.is_direct_product_query(query):
            return True

        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        gift_context = getattr(self.memory_stack, "GIFT_CONTEXT_KEYWORDS", set())
        if query_tokens & gift_context:
            return False

        return True
