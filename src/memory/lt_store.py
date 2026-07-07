"""Long-term catalog memory backed by Qdrant."""

from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from infastructure.config import EMBEDDING_DIM, QDRANT_COLLECTION_NAME
from infastructure.db.qdrant_client import ensure_collection, qdrant_available, query_points, upsert_points
from memory.schemas import CatalogMatch, CatalogProduct


class SimpleHashEmbedder:
    """Deterministic lightweight embedder for catalog text."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def embed_text(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            bucket = hash(token) % self.dim
            vector[bucket] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]


class CatalogVectorStore:
    """Persistent Qdrant vector store for the crawled `catalog.json`."""

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION_NAME,
        embedder: Optional[SimpleHashEmbedder] = None,
    ) -> None:
        self.collection_name = collection_name
        self.embedder = embedder or SimpleHashEmbedder()

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
        if not self.available():
            return []

        try:
            response = query_points(
                query_vector=self.embedder.embed_text(query),
                limit=top_k,
                score_threshold=score_threshold,
                collection_name=self.collection_name,
            )
        except Exception:
            return []

        points = getattr(response, "points", response)
        matches: List[CatalogMatch] = []
        for point in points:
            payload = point.payload or {}
            product = CatalogProduct(
                name=payload.get("name", ""),
                price=payload.get("price", ""),
                description=payload.get("description", ""),
                availability=payload.get("availability", ""),
                url=payload.get("url", ""),
            )
            matches.append(
                CatalogMatch(
                    product=product,
                    score=float(point.score or 0.0),
                    product_id=str(payload.get("product_id", "")),
                )
            )
        return matches

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
