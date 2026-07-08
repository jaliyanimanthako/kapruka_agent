"""Embedding providers for catalog vectorization."""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional

from infastructure.config import EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDING_PROVIDER, OPENAI_API_KEY

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenAI = None


class SimpleHashEmbedder:
    """Deterministic fallback embedder used when OpenAI is unavailable."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.model = "hash-fallback"

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


class OpenAIEmbeddingProvider:
    """OpenAI embeddings provider with the interface expected by Qdrant storage."""

    def __init__(
        self,
        api_key: str,
        model: str = EMBEDDING_MODEL,
        dimensions: int = EMBEDDING_DIM,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for OpenAI embeddings.")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dim = dimensions

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        items = [text for text in texts if text]
        if not items:
            return []

        create_kwargs = {
            "model": self.model,
            "input": items,
        }
        if self.model.startswith("text-embedding-3"):
            create_kwargs["dimensions"] = self.dim

        response = self.client.embeddings.create(**create_kwargs)
        return [item.embedding for item in response.data]


def get_default_catalog_embedder() -> OpenAIEmbeddingProvider | SimpleHashEmbedder:
    """Return the preferred embedder for catalog search."""
    if EMBEDDING_PROVIDER == "openai" and OPENAI_API_KEY:
        try:
            return OpenAIEmbeddingProvider(api_key=OPENAI_API_KEY)
        except Exception:
            return SimpleHashEmbedder()
    return SimpleHashEmbedder()
