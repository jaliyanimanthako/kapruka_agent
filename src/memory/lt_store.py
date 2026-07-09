"""Long-term catalog memory backed by Qdrant."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from infastructure.llm_providers.embeddings import OpenAIEmbeddingProvider, SimpleHashEmbedder, get_default_catalog_embedder
from infastructure.config import EMBEDDING_DIM, QDRANT_COLLECTION_NAME
from infastructure.db.qdrant_client import ensure_collection, qdrant_available, query_points, upsert_points
from memory.schemas import CatalogMatch, CatalogProduct


class CatalogVectorStore:
    """Persistent Qdrant vector store for the crawled `catalog.json`."""

    QUERY_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "can",
        "for",
        "give",
        "have",
        "i",
        "in",
        "is",
        "me",
        "my",
        "of",
        "on",
        "options",
        "or",
        "please",
        "show",
        "some",
        "the",
        "to",
        "what",
        "with",
        "your",
    }

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION_NAME,
        embedder: Optional[OpenAIEmbeddingProvider | SimpleHashEmbedder] = None,
        catalog_path: str | Path = "catalog.json",
    ) -> None:
        self.collection_name = collection_name
        self.embedder = embedder or get_default_catalog_embedder()
        self.catalog_path = Path(catalog_path)

    def available(self) -> bool:
        return qdrant_available()

    def ingest_catalog(self, catalog_path: str | Path = "catalog.json") -> int:
        """Load products from `catalog.json` and upsert them into Qdrant."""
        if not self.available():
            return 0

        try:
            from qdrant_client.http.models import PointStruct
        except ModuleNotFoundError:
            return 0

        try:
            ensure_collection(collection_name=self.collection_name, vector_size=self.embedder.dim)
        except Exception:
            return 0

        catalog_data = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        products = [CatalogProduct(**product) for product in catalog_data.get("products", [])]

        deduped_products: Dict[str, CatalogProduct] = {}
        for product in products:
            deduped_products[self._product_id(product.url)] = product

        points: List[PointStruct] = []
        documents = [self._product_document(product) for product in deduped_products.values()]
        embeddings = self.embedder.embed_texts(documents)
        for product, embedding in zip(deduped_products.values(), embeddings):
            point_id = self._product_id(product.url)
            payload = product.to_dict()
            payload["product_id"] = point_id
            payload["document"] = self._product_document(product)
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, product.url)),
                    vector=embedding,
                    payload=payload,
                )
            )

        try:
            upsert_points(points, collection_name=self.collection_name)
            return len(points)
        except Exception:
            return 0

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.15) -> List[CatalogMatch]:
        """Search the Qdrant catalog memory."""
        matches, _ = self.search_detailed(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        return matches

    def search_detailed(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.15,
        progress_callback: Optional[Callable[[str], None]] = None,
        ) -> tuple[List[CatalogMatch], Dict[str, int]]:
        """Search the catalog and return step timings for embedding and Qdrant lookup."""
        lexical_started_at = time.perf_counter()
        lexical_matches = self._lexical_matches(query=query, top_k=top_k)
        lexical_ms = int((time.perf_counter() - lexical_started_at) * 1000)

        if not self.available():
            return lexical_matches, {
                "lexical_search": lexical_ms,
                "query_embedding": 0,
                "qdrant_search": 0,
                "result_rerank": 0,
            }

        if progress_callback:
            progress_callback("Generating query embedding...")

        try:
            embedding_started_at = time.perf_counter()
            query_vector = self.embedder.embed_text(query)
            embedding_ms = int((time.perf_counter() - embedding_started_at) * 1000)

            if progress_callback:
                progress_callback("Querying Qdrant...")

            qdrant_started_at = time.perf_counter()
            response = query_points(
                query_vector=query_vector,
                limit=max(top_k * 3, top_k),
                score_threshold=score_threshold,
                collection_name=self.collection_name,
            )
            qdrant_ms = int((time.perf_counter() - qdrant_started_at) * 1000)
        except Exception:
            return lexical_matches, {
                "lexical_search": lexical_ms,
                "query_embedding": 0,
                "qdrant_search": 0,
                "result_rerank": 0,
            }

        points = getattr(response, "points", response)
        semantic_matches: List[CatalogMatch] = []
        for point in points:
            payload = point.payload or {}
            product = CatalogProduct(
                name=payload.get("name", ""),
                price=payload.get("price", ""),
                description=payload.get("description", ""),
                availability=payload.get("availability", ""),
                url=payload.get("url", ""),
            )
            semantic_matches.append(
                CatalogMatch(
                    product=product,
                    score=float(point.score or 0.0),
                    product_id=str(payload.get("product_id", "")),
                )
            )
        rerank_started_at = time.perf_counter()
        matches = self._merge_and_rerank_results(
            query=query,
            semantic_matches=semantic_matches,
            lexical_matches=lexical_matches,
            top_k=top_k,
        )
        rerank_ms = int((time.perf_counter() - rerank_started_at) * 1000)
        return matches, {
            "lexical_search": lexical_ms,
            "query_embedding": embedding_ms,
            "qdrant_search": qdrant_ms,
            "result_rerank": rerank_ms,
        }

    def _product_id(self, url: str) -> str:
        match = re.search(r"/kid/([^/?#]+)", url, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return re.sub(r"[^a-z0-9]+", "-", url.lower()).strip("-")

    def _product_document(self, product: CatalogProduct) -> str:
        return (
            f"Product: {product.name}\n"
            f"Price: {product.price}\n"
            f"Availability: {product.availability}\n"
            f"Description: {product.description}\n"
            f"URL: {product.url}"
        )

    def _lexical_matches(self, query: str, top_k: int) -> List[CatalogMatch]:
        if not self.catalog_path.exists():
            return []

        tokens = self._query_tokens(query)
        if not tokens:
            return []

        catalog_data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        results: List[CatalogMatch] = []
        for item in catalog_data.get("products", []):
            product = CatalogProduct(**item)
            haystack = f"{product.name} {product.description}".lower()
            overlap = sum(1 for token in tokens if token in haystack)
            if overlap == 0:
                continue
            results.append(
                CatalogMatch(
                    product=product,
                    score=float(overlap),
                    product_id=self._product_id(product.url),
                )
            )

        results.sort(key=lambda match: (-match.score, match.product.name))
        return results[: max(top_k * 3, top_k)]

    def _merge_and_rerank_results(
        self,
        query: str,
        semantic_matches: List[CatalogMatch],
        lexical_matches: List[CatalogMatch],
        top_k: int,
    ) -> List[CatalogMatch]:
        tokens = self._query_tokens(query)
        merged: Dict[str, CatalogMatch] = {}

        for match in semantic_matches:
            merged[match.product_id] = match

        for match in lexical_matches:
            existing = merged.get(match.product_id)
            if existing is None or match.score > existing.score:
                merged[match.product_id] = match

        def ranking_tuple(match: CatalogMatch) -> tuple[float, float, str]:
            name_desc = f"{match.product.name} {match.product.description}".lower()
            lexical_overlap = sum(1 for token in tokens if token in name_desc)
            semantic_score = match.score if match.score <= 1.0 else 0.0
            lexical_boost = float(lexical_overlap) * 10.0
            availability_boost = 2.0 if "in stock" in match.product.availability.lower() else 0.0
            combined = lexical_boost + semantic_score + availability_boost
            return (combined, semantic_score, match.product.name)

        ranked = sorted(merged.values(), key=ranking_tuple, reverse=True)
        return ranked[:top_k]

    def _query_tokens(self, query: str) -> List[str]:
        tokens: List[str] = []
        seen = set()
        for token in re.findall(r"[a-z0-9]+", query.lower()):
            normalized = token[:-1] if token.endswith("s") and len(token) > 3 else token
            for candidate in (token, normalized):
                if len(candidate) <= 1 or candidate in self.QUERY_STOPWORDS or candidate in seen:
                    continue
                seen.add(candidate)
                tokens.append(candidate)
        return tokens
