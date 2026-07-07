"""JSON-backed semantic memory for recipient profiles."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from memory.schemas import RecipientProfile


class SemanticProfileStore:
    """Stores recipient preferences and notes in a JSON file."""

    def __init__(self, storage_path: str | Path = "recipient_profiles.json") -> None:
        self.storage_path = Path(storage_path)

    def upsert_profile(self, profile: RecipientProfile) -> RecipientProfile:
        data = self._read()
        existing = data.get(profile.recipient_id)
        merged = RecipientProfile.from_dict(existing) if existing else RecipientProfile(
            recipient_id=profile.recipient_id,
            name=profile.name,
            relationship=profile.relationship,
        )

        if profile.name:
            merged.name = profile.name
        if profile.relationship:
            merged.relationship = profile.relationship

        merged.preferences = self._merge_unique(merged.preferences, profile.preferences)
        merged.notes = self._merge_unique(merged.notes, profile.notes)
        merged.updated_at = time.time()

        data[profile.recipient_id] = merged.to_dict()
        self._write(data)
        return merged

    def get_profile(self, recipient_id: str) -> Optional[RecipientProfile]:
        data = self._read()
        profile = data.get(recipient_id)
        if not profile:
            return None
        return RecipientProfile.from_dict(profile)

    def list_profiles(self) -> List[RecipientProfile]:
        return [RecipientProfile.from_dict(item) for item in self._read().values()]

    def remember_preference(
        self,
        recipient_id: str,
        name: str,
        relationship: str = "",
        preference: str = "",
        note: str = "",
    ) -> RecipientProfile:
        return self.upsert_profile(
            RecipientProfile(
                recipient_id=recipient_id,
                name=name,
                relationship=relationship,
                preferences=[preference] if preference else [],
                notes=[note] if note else [],
            )
        )

    def _read(self) -> Dict[str, Dict]:
        if not self.storage_path.exists():
            return {}
        return json.loads(self.storage_path.read_text(encoding="utf-8"))

    def _write(self, data: Dict[str, Dict]) -> None:
        self.storage_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _merge_unique(self, existing: List[str], incoming: List[str]) -> List[str]:
        merged = list(existing)
        existing_lower = {item.lower() for item in existing}
        for item in incoming:
            if item and item.lower() not in existing_lower:
                merged.append(item)
                existing_lower.add(item.lower())
        return merged
