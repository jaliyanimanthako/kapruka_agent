from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.catalog_agent import CatalogAgent
from agents.logistics_agent import LogisticsAgent
from agents.orchestrator import KaprukaOrchestrator
from agents.router import KaprukaRouter
from memory.memory_ops import CognitiveMemoryStack
from memory.schemas import CatalogMatch, CatalogProduct
from memory.semantic_store import SemanticProfileStore
from memory.st_store import ShortTermMemoryStore


class FakeRouterClient:
    def __init__(self, content: str) -> None:
        self._content = content
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        class _Message:
            def __init__(self, content: str) -> None:
                self.content = content

        class _Choice:
            def __init__(self, content: str) -> None:
                self.message = _Message(content)

        class _Response:
            def __init__(self, content: str) -> None:
                self.choices = [_Choice(content)]

        return _Response(self._content)


class FakeLongTermStore:
    def ingest_catalog(self, catalog_path: str | Path = "catalog.json") -> int:
        return 0

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        return self.search_detailed(query, top_k=top_k, score_threshold=score_threshold)[0]

    def search_detailed(self, query: str, top_k: int = 5, score_threshold: float = 0.0, progress_callback=None):
        product = CatalogProduct(
            name="Ferrero Rocher Heart Bouquet For Her",
            price="US$20.00",
            description="Chocolate bouquet gift for romantic occasions",
            availability="In Stock",
            url="https://www.kapruka.com/buyonline/ferrero-rocher-heart-bouquet-f/kid/combochg164",
        )
        return [CatalogMatch(product=product, score=0.42, product_id="combochg164")], {
            "lexical_search": 1,
            "query_embedding": 1,
            "qdrant_search": 1,
            "result_rerank": 1,
        }


class FakeChatService:
    def answer_query(self, query: str, bundle: dict) -> str:
        return f"stubbed answer for: {query} ({len(bundle.get('catalog_matches', []))} match)"


class AgentTests(unittest.TestCase):
    def test_router_uses_llm_json_response(self) -> None:
        client = FakeRouterClient(
            '{"route":"logistics_check","confidence":0.97,"reasoning":"Delivery question.","params":{"message":"Can you deliver this to Colombo today?"}}'
        )
        decision = KaprukaRouter(llm_client=client).route("Can you deliver this to Colombo today?")
        self.assertEqual(decision.route, "logistics_check")
        self.assertEqual(decision.params["message"], "Can you deliver this to Colombo today?")

    def test_router_classifies_preference_update(self) -> None:
        decision = KaprukaRouter(use_llm=False).route("Remember that my wife loves dark chocolate")
        self.assertEqual(decision.route, "preference_update")

    def test_router_classifies_logistics(self) -> None:
        decision = KaprukaRouter(use_llm=False).route("Can you deliver this to Colombo today?")
        self.assertEqual(decision.route, "logistics_check")

    def test_logistics_agent_detects_known_district(self) -> None:
        result = LogisticsAgent().check_delivery("Need same-day delivery to Kandy")
        self.assertEqual(result.district, "Kandy")
        self.assertTrue(result.supported)
        self.assertTrue(result.needs_manual_confirmation)
        self.assertEqual(result.urgency, "same_day")

    def test_logistics_agent_requires_district_when_missing(self) -> None:
        result = LogisticsAgent().check_delivery("Can you deliver this tomorrow?")
        self.assertIsNone(result.district)
        self.assertFalse(result.supported)
        self.assertTrue(result.needs_manual_confirmation)

    def test_logistics_agent_requires_live_confirmation_for_exact_guarantee(self) -> None:
        result = LogisticsAgent().check_delivery("Can you guarantee delivery to Colombo by tomorrow?")
        self.assertEqual(result.district, "Colombo")
        self.assertTrue(result.supported)
        self.assertTrue(result.needs_manual_confirmation)
        self.assertTrue(result.exact_guarantee_requested)

    def test_logistics_agent_uses_memory_context_for_location_follow_up(self) -> None:
        result = LogisticsAgent().check_delivery(
            "i am near kelaniya",
            memory_context="user: what are your delivery options\nassistant: I need the target Sri Lankan district before I can assess delivery feasibility.",
        )
        self.assertEqual(result.district, "Gampaha")
        self.assertTrue(result.supported)

    def test_router_uses_memory_context_for_logistics_follow_up(self) -> None:
        decision = KaprukaRouter(use_llm=False).route(
            "i am near kelaniya",
            memory_context="user: what are your delivery options\nassistant: I need the target Sri Lankan district before I can assess delivery feasibility.",
        )
        self.assertEqual(decision.route, "logistics_check")

    def test_orchestrator_updates_semantic_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
            )

            response = orchestrator.handle_message(
                "Remember that my wife loves dark chocolate and elegant packaging",
                recipient_id="wife",
                recipient_name="Wife",
            )

            self.assertEqual(response.route, "preference_update")
            profile = stack.get_recipient_profile("wife")
            assert profile is not None
            self.assertIn("Loves dark chocolate", profile.preferences)

    def test_orchestrator_reuses_active_recipient_for_follow_up_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
            )

            first = orchestrator.handle_message("my wife loves dark chocolate")
            second = orchestrator.handle_message("she prefers elegant packaging")

            self.assertEqual(first.route, "preference_update")
            self.assertEqual(second.route, "preference_update")
            profile = stack.get_recipient_profile("wife")
            assert profile is not None
            self.assertIn("Loves dark chocolate", profile.preferences)
            self.assertIn("Prefers elegant packaging", profile.preferences)

    def test_orchestrator_runs_catalog_specialist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            stack.save_recipient_profile(
                recipient_id="wife",
                name="Wife",
                relationship="spouse",
                preferences=["Loves dark chocolate"],
                notes=["Prefers elegant packaging"],
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
            )

            response = orchestrator.handle_message("gift for wife", recipient_id="wife")

            self.assertEqual(response.route, "catalog_search")
            self.assertIn("stubbed answer", response.answer)
            self.assertEqual(
                response.specialist_output["catalog"]["bundle"]["catalog_matches"][0]["product"]["name"],
                "Ferrero Rocher Heart Bouquet For Her",
            )
            self.assertIn("router", response.timings_ms)
            self.assertIn("short_term_read", response.timings_ms)
            self.assertIn("semantic_profile_read", response.timings_ms)
            self.assertIn("retrieval_query_build", response.timings_ms)
            self.assertIn("lexical_search", response.timings_ms)
            self.assertIn("query_embedding", response.timings_ms)
            self.assertIn("qdrant_search", response.timings_ms)
            self.assertIn("result_rerank", response.timings_ms)
            self.assertIn("catalog_retrieval", response.timings_ms)
            self.assertIn("catalog_answer_generation", response.timings_ms)
            self.assertIn("total", response.timings_ms)

    def test_orchestrator_does_not_inject_default_wife_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
            )

            response = orchestrator.handle_message("bluetooth speakers")

            self.assertEqual(response.route, "catalog_search")
            self.assertIsNone(response.specialist_output["catalog"]["bundle"]["recipient_profile"])

    def test_direct_product_query_ignores_recent_turn_context_in_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            stack.add_turn("demo-user", "demo-session", "user", "bluetooth speakers")
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
            )

            response = orchestrator.handle_message("hi what are my options in led tvs")

            self.assertEqual(response.route, "catalog_search")
            self.assertEqual(response.specialist_output["catalog"]["bundle"]["recent_turns"], [])


if __name__ == "__main__":
    unittest.main()
