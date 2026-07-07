"""Schemas for the Kapruka 3-tier cognitive memory stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class ConversationTurn:
    """One conversation message stored in short-term memory."""

    user_id: str
    session_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    ts: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "ts": self.ts,
        }


@dataclass
class CatalogProduct:
    """Product record loaded from the crawled catalog."""

    name: str
    price: str
    description: str
    availability: str
    url: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "price": self.price,
            "description": self.description,
            "availability": self.availability,
            "url": self.url,
        }


@dataclass
class CatalogMatch:
    """Semantic result returned from Qdrant catalog retrieval."""

    product: CatalogProduct
    score: float
    product_id: str


@dataclass
class RecipientProfile:
    """JSON-backed semantic profile for a recipient."""

    recipient_id: str
    name: str
    relationship: str = ""
    preferences: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recipient_id": self.recipient_id,
            "name": self.name,
            "relationship": self.relationship,
            "preferences": self.preferences,
            "notes": self.notes,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecipientProfile":
        return cls(
            recipient_id=data["recipient_id"],
            name=data["name"],
            relationship=data.get("relationship", ""),
            preferences=list(data.get("preferences", [])),
            notes=list(data.get("notes", [])),
            updated_at=float(data.get("updated_at", 0.0)),
        )
