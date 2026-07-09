"""JSON-backed semantic memory for user profiles."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

from memory.schemas import UserProfile


class UserProfileStore:
    """Stores lightweight user identity details in a JSON file."""

    def __init__(self, storage_path: str | Path = "user_profiles.json") -> None:
        self.storage_path = Path(storage_path)

    def upsert_profile(self, profile: UserProfile) -> UserProfile:
        data = self._read()
        existing = data.get(profile.user_id)
        merged = UserProfile.from_dict(existing) if existing else UserProfile(user_id=profile.user_id)

        if profile.name:
            merged.name = profile.name

        merged.updated_at = time.time()
        data[profile.user_id] = merged.to_dict()
        self._write(data)
        return merged

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        data = self._read()
        profile = data.get(user_id)
        if not profile:
            return None
        return UserProfile.from_dict(profile)

    def _read(self) -> Dict[str, Dict]:
        if not self.storage_path.exists():
            return {}
        return json.loads(self.storage_path.read_text(encoding="utf-8"))

    def _write(self, data: Dict[str, Dict]) -> None:
        self.storage_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
