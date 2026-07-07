"""Qdrant client helpers for catalog vector storage."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    QdrantClient = None
    Distance = FieldCondition = Filter = MatchValue = PointStruct = VectorParams = None

from infastructure.config import EMBEDDING_DIM, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL


_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Return a singleton Qdrant client."""
    global _client
    if _client is not None:
        return _client

    if QdrantClient is None:
        raise RuntimeError("qdrant-client package is required for Qdrant support.")

    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is not configured.")

    _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=30)
    return _client


def qdrant_available() -> bool:
    """Check whether a Qdrant endpoint is configured."""
    return bool(QDRANT_URL)


def ensure_collection(
    collection_name: str = QDRANT_COLLECTION_NAME,
    vector_size: int = EMBEDDING_DIM,
) -> None:
    """Create the configured collection if it does not exist."""
    client = get_qdrant_client()
    existing = {collection.name for collection in client.get_collections().collections}
    if collection_name in existing:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def upsert_points(
    points: List[PointStruct],
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> None:
    """Upsert points into Qdrant."""
    if not points:
        return
    get_qdrant_client().upsert(collection_name=collection_name, points=points)


def query_points(
    query_vector: List[float],
    limit: int = 5,
    score_threshold: float = 0.0,
    collection_name: str = QDRANT_COLLECTION_NAME,
    filters: Optional[Dict[str, Any]] = None,
):
    """Run a semantic query against Qdrant."""
    query_filter = None
    if filters:
        query_filter = Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ]
        )

    return get_qdrant_client().query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )
