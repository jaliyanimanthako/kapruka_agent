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
from infastructure.llm_providers.embeddings import SimpleHashEmbedder, get_default_catalog_embedder
from infastructure.llm_providers.llm_services import build_memory_prompt


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

    def test_default_catalog_embedder_falls_back_without_openai_key(self) -> None:
        embedder = get_default_catalog_embedder()
        self.assertTrue(hasattr(embedder, "embed_text"))
        self.assertTrue(hasattr(embedder, "embed_texts"))

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

    def test_enhanced_query_includes_recent_turns_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_store = SemanticProfileStore(Path(tmp_dir) / "profiles.json")
            short_term = ShortTermMemoryStore(max_turns=5, ttl_seconds=3600, use_database=False)
            stack = CognitiveMemoryStack(short_term=short_term, semantic=profile_store)

            stack.add_turn("u1", "s1", "user", "Need a romantic gift")
            stack.add_turn("u1", "s1", "assistant", "Do you prefer chocolates or flowers?")
            stack.save_recipient_profile(
                recipient_id="wife",
                name="Wife",
                relationship="spouse",
                preferences=["Loves dark chocolate"],
                notes=["Anniversary next week"],
            )

            query = stack._build_enhanced_query(
                user_id="u1",
                session_id="s1",
                query="gift for wife",
                recipient_id="wife",
            )

            self.assertIn("gift for wife", query)
            self.assertIn("Loves dark chocolate", query)
            self.assertIn("Anniversary next week", query)

    def test_build_memory_prompt_shows_all_three_sections(self) -> None:
        bundle = {
            "recent_turns": [{"role": "user", "content": "Need a gift for my wife"}],
            "recipient_profile": {
                "name": "Wife",
                "relationship": "spouse",
                "preferences": ["Loves dark chocolate"],
                "notes": ["Anniversary next week"],
            },
            "catalog_matches": [
                {
                    "product_id": "combo1",
                    "score": 0.42,
                    "product": {
                        "name": "Ferrero Rocher Heart Bouquet For Her",
                        "price": "US$20.00",
                        "availability": "In Stock",
                        "description": "Chocolate bouquet gift",
                        "url": "https://example.com/product",
                    },
                }
            ],
        }

        prompt = build_memory_prompt("gift for wife", bundle)
        self.assertIn("SHORT_TERM_MEMORY", prompt)
        self.assertIn("SEMANTIC_PROFILE", prompt)
        self.assertIn("LONG_TERM_CATALOG", prompt)
        self.assertIn("Ferrero Rocher Heart Bouquet For Her", prompt)

    def test_direct_product_query_does_not_absorb_profile_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_store = SemanticProfileStore(Path(tmp_dir) / "profiles.json")
            short_term = ShortTermMemoryStore(max_turns=5, ttl_seconds=3600, use_database=False)
            stack = CognitiveMemoryStack(short_term=short_term, semantic=profile_store)

            stack.add_turn("u1", "s1", "user", "I need a gift for my wife")
            stack.save_recipient_profile(
                recipient_id="wife",
                name="Wife",
                relationship="spouse",
                preferences=["Loves dark chocolate"],
                notes=["Prefers elegant packaging"],
            )

            query = stack._build_enhanced_query(
                user_id="u1",
                session_id="s1",
                query="bluetooth speakers",
                recipient_id="wife",
            )

            self.assertEqual(query, "bluetooth speakers")

    def test_lexical_matches_find_led_tvs(self) -> None:
        store = CatalogVectorStore()
        matches = store._lexical_matches("led tvs", top_k=5)
        names = [match.product.name.lower() for match in matches]
        self.assertTrue(any("tv" in name for name in names))


if __name__ == "__main__":
    unittest.main()
