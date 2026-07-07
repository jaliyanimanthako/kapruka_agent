from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from memory.lt_store import CatalogVectorStore
from memory.memory_ops import CognitiveMemoryStack
from memory.schemas import CatalogProduct
from memory.semantic_store import SemanticProfileStore
from memory.st_store import ShortTermMemoryStore


class MemoryStackTests(unittest.TestCase):
    def test_short_term_local_ring_buffer(self) -> None:
        store = ShortTermMemoryStore(max_turns=2, ttl_seconds=3600, use_database=False)
        stack = CognitiveMemoryStack(short_term=store)

        stack.add_turn("u1", "s1", "user", "first")
        stack.add_turn("u1", "s1", "assistant", "second")
        stack.add_turn("u1", "s1", "user", "third")

        recent = store.recent("u1", "s1", 5)
        self.assertEqual([turn.content for turn in recent], ["second", "third"])

    def test_semantic_profile_store_merges_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SemanticProfileStore(Path(tmp_dir) / "profiles.json")
            store.remember_preference("wife", "Wife", preference="Loves dark chocolate")
            profile = store.remember_preference("wife", "Wife", note="Prefers elegant packaging")

            self.assertEqual(profile.preferences, ["Loves dark chocolate"])
            self.assertEqual(profile.notes, ["Prefers elegant packaging"])

    def test_catalog_vector_store_uses_kid_suffix_as_product_id(self) -> None:
        store = CatalogVectorStore()
        product = CatalogProduct(
            name="Dark Chocolate Box",
            price="US$10.00",
            description="Premium dark chocolate gift box",
            availability="In Stock",
            url="https://www.kapruka.com/buyonline/dark-chocolate-box/kid/choco123",
        )

        self.assertEqual(store._product_id(product.url), "choco123")
        self.assertIn("Dark Chocolate Box", store._product_document(product))

    def test_context_bundle_combines_three_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_store = SemanticProfileStore(Path(tmp_dir) / "profiles.json")
            short_term = ShortTermMemoryStore(max_turns=5, ttl_seconds=3600, use_database=False)
            stack = CognitiveMemoryStack(short_term=short_term, semantic=profile_store)

            stack.add_turn("u1", "s1", "user", "Need a gift for my wife")
            stack.save_recipient_profile(
                recipient_id="wife",
                name="Wife",
                preferences=["Loves dark chocolate"],
                notes=["Anniversary next week"],
            )

            bundle = stack.build_context_bundle("u1", "s1", "dark chocolate gift", recipient_id="wife")

            self.assertEqual(len(bundle["recent_turns"]), 1)
            self.assertEqual(bundle["recipient_profile"]["preferences"], ["Loves dark chocolate"])


if __name__ == "__main__":
    unittest.main()
