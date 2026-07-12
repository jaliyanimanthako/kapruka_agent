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
from agents.meta_agent import MetaAgent
from agents.orchestrator import KaprukaOrchestrator
from agents.router import KaprukaRouter
from memory.memory_ops import CognitiveMemoryStack
from memory.schemas import CatalogMatch, CatalogProduct
from memory.semantic_store import SemanticProfileStore
from memory.st_store import ShortTermMemoryStore
from memory.user_store import UserProfileStore


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


class FakeLogisticsClient(FakeRouterClient):
    pass


class FakeRouterGuardClient:
    def __init__(self, route_content: str, guard_content: str) -> None:
        self._route_content = route_content
        self._guard_content = guard_content
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        messages = kwargs.get("messages", [])
        system_prompt = ""
        if messages:
            system_prompt = str(messages[0].get("content", ""))
        content = self._guard_content if "should be saved into a recipient profile" in system_prompt else self._route_content

        class _Message:
            def __init__(self, content: str) -> None:
                self.content = content

        class _Choice:
            def __init__(self, content: str) -> None:
                self.message = _Message(content)

        class _Response:
            def __init__(self, content: str) -> None:
                self.choices = [_Choice(content)]

        return _Response(content)


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


class FakeUnsafeChocolateLongTermStore:
    def ingest_catalog(self, catalog_path: str | Path = "catalog.json") -> int:
        return 0

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        return self.search_detailed(query, top_k=top_k, score_threshold=score_threshold)[0]

    def search_detailed(self, query: str, top_k: int = 5, score_threshold: float = 0.0, progress_callback=None):
        unsafe = CatalogProduct(
            name="Dark Chocolate Birthday Hamper",
            price="US$20.00",
            description="A premium dark chocolate hamper",
            availability="In Stock",
            url="https://example.com/dark",
        )
        safe = CatalogProduct(
            name="White Chocolate Birthday Hamper",
            price="US$22.00",
            description="A white chocolate gift hamper",
            availability="In Stock",
            url="https://example.com/white",
        )
        return [
            CatalogMatch(product=unsafe, score=0.9, product_id="dark1"),
            CatalogMatch(product=safe, score=0.8, product_id="white1"),
        ], {
            "lexical_search": 1,
            "query_embedding": 1,
            "qdrant_search": 1,
            "result_rerank": 1,
        }


class FakeMixedTvLongTermStore:
    def ingest_catalog(self, catalog_path: str | Path = "catalog.json") -> int:
        return 0

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        return self.search_detailed(query, top_k=top_k, score_threshold=score_threshold)[0]

    def search_detailed(self, query: str, top_k: int = 5, score_threshold: float = 0.0, progress_callback=None):
        teddy = CatalogProduct(
            name="Adarei Teddy In Love",
            price="US$19.44",
            description="Soft teddy bear gift",
            availability="In Stock",
            url="https://example.com/teddy",
        )
        tv = CatalogProduct(
            name="Konka 32 Inch Full Hd Led Tv Kg32ee682",
            price="US$120.00",
            description="Full HD LED TV for home entertainment",
            availability="In Stock",
            url="https://example.com/tv",
        )
        return [
            CatalogMatch(product=teddy, score=0.99, product_id="teddy1"),
            CatalogMatch(product=tv, score=0.42, product_id="tv1"),
        ], {
            "lexical_search": 1,
            "query_embedding": 1,
            "qdrant_search": 1,
            "result_rerank": 1,
        }


class FakeChatService:
    def answer_query(self, query: str, bundle: dict) -> str:
        return f"stubbed answer for: {query} ({len(bundle.get('catalog_matches', []))} match)"

    def judge_answer_relevance(self, query: str, bundle: dict, answer: str) -> dict:
        return {
            "relevant": True,
            "confidence": 0.95,
            "reason": "The draft answer stays within the retrieved catalog context.",
            "supported_products": [],
            "unsupported_products": [],
            "mentioned_products": [],
        }


class FakeFurnitureHallucinatingChatService:
    def answer_query(self, query: str, bundle: dict) -> str:
        return (
            "Here are some furniture options available in the catalog that might interest you:\n\n"
            "1. Queen Of My World Gift Set\n"
            "Price: US$41.22\n"
            "Description: This gift set celebrates the most special woman in your life.\n"
            "Availability: In Stock\n"
            "- Why it matches: While not traditional furniture, this gift set can enhance the ambiance of a living space.\n\n"
            "2. Ferrero Rocher Heart Bouquet For Her\n"
            "Price: US$20.00\n"
            "Description: Chocolate bouquet gift for romantic occasions.\n"
            "Availability: In Stock\n"
            "- Why it matches: While not furniture, it can add charm to the home environment."
        )

    def judge_answer_relevance(self, query: str, bundle: dict, answer: str) -> dict:
        return {
            "relevant": False,
            "confidence": 0.98,
            "reason": "The mentioned products are not furniture and the answer relies on weak decor justifications.",
            "supported_products": [],
            "unsupported_products": ["Ferrero Rocher Heart Bouquet For Her"],
            "mentioned_products": ["Ferrero Rocher Heart Bouquet For Her"],
        }


class FakeFatherGiftChatService:
    def answer_query(self, query: str, bundle: dict) -> str:
        return (
            "A good option for your father is Ferrero Rocher Heart Bouquet For Her.\n"
            "It is in stock and works as a thoughtful present."
        )

    def judge_answer_relevance(self, query: str, bundle: dict, answer: str) -> dict:
        return {
            "relevant": True,
            "confidence": 0.82,
            "reason": "This is a broad gift-discovery request and the recommendation is acceptable from the retrieved catalog.",
            "supported_products": ["Ferrero Rocher Heart Bouquet For Her"],
            "unsupported_products": [],
            "mentioned_products": ["Ferrero Rocher Heart Bouquet For Her"],
        }


class AgentTests(unittest.TestCase):
    def test_router_uses_llm_json_response(self) -> None:
        client = FakeRouterClient(
            '{"intent":"logistics_check","confidence":0.97,"reason":"Delivery question.","params":{"message":"Can you deliver this to Colombo today?"}}'
        )
        decision = KaprukaRouter(llm_client=client).route("Can you deliver this to Colombo today?")
        self.assertEqual(decision.route, "logistics_check")
        self.assertEqual(decision.params["message"], "Can you deliver this to Colombo today?")

    def test_router_classifies_identity(self) -> None:
        decision = KaprukaRouter(use_llm=False).route("hi, who are you?")
        self.assertEqual(decision.route, "identity")

    def test_router_classifies_smalltalk(self) -> None:
        decision = KaprukaRouter(use_llm=False).route("thanks!")
        self.assertEqual(decision.route, "smalltalk")

    def test_router_classifies_preference_update(self) -> None:
        decision = KaprukaRouter(use_llm=False).route("Remember that my wife loves dark chocolate")
        self.assertEqual(decision.route, "preference_update")

    def test_router_corrects_llm_preference_misroute_for_recommendation_feedback(self) -> None:
        client = FakeRouterClient(
            '{"intent":"preference_update","confidence":0.96,"reason":"The user is sharing a recipient note.","params":{"message":"But I think some electronic items or any tool boxes may suit better"}}'
        )
        decision = KaprukaRouter(llm_client=client).route(
            "But I think some electronic items or any tool boxes may suit better"
        )

        self.assertEqual(decision.route, "catalog_search")

    def test_router_corrects_llm_preference_misroute_for_love_more_feedback(self) -> None:
        client = FakeRouterClient(
            '{"intent":"preference_update","confidence":0.96,"reason":"The user is sharing a recipient note.","params":{"message":"I think she will love more to get some electronic items"}}'
        )
        decision = KaprukaRouter(llm_client=client).route(
            "I think she will love more to get some electronic items"
        )

        self.assertEqual(decision.route, "catalog_search")

    def test_router_classifies_logistics(self) -> None:
        decision = KaprukaRouter(use_llm=False).route("Can you deliver this to Colombo today?")
        self.assertEqual(decision.route, "logistics_check")

    def test_router_does_not_treat_intro_plus_product_request_as_smalltalk(self) -> None:
        decision = KaprukaRouter(use_llm=False).route("I am gayantha, what are your options in cakes")

        self.assertEqual(decision.route, "catalog_search")

    def test_logistics_agent_detects_known_district(self) -> None:
        result = LogisticsAgent().check_delivery("Need same-day delivery to Kandy")
        self.assertEqual(result.district, "Kandy")
        self.assertTrue(result.supported)
        self.assertTrue(result.needs_manual_confirmation)
        self.assertEqual(result.urgency, "same_day")
        self.assertEqual(result.service_tier, "Tier 2")
        self.assertIn("8:30 AM", result.cutoff_guidance)

    def test_logistics_agent_uses_llm_json_response(self) -> None:
        client = FakeLogisticsClient(
            '{"district":"Colombo","supported":true,"needs_manual_confirmation":true,"urgency":"scheduled_soon","exact_guarantee_requested":true,"service_tier":"Tier 1","typical_timing":"Same-day for supported items","cutoff_guidance":"10:00 AM to 12:00 PM","item_guidance":"Fresh items require checkout confirmation.","summary":"Colombo is supported, but live confirmation is required."}'
        )
        result = LogisticsAgent(llm_client=client).check_delivery("Can you guarantee delivery to Colombo by tomorrow?")
        self.assertEqual(result.district, "Colombo")
        self.assertTrue(result.supported)
        self.assertTrue(result.needs_manual_confirmation)
        self.assertEqual(result.urgency, "scheduled_soon")
        self.assertEqual(result.service_tier, "Tier 1")

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

    def test_logistics_agent_uses_tier_one_policy_for_colombo_same_day_cake(self) -> None:
        result = LogisticsAgent().check_delivery("Can you deliver a birthday cake to Colombo today?")

        self.assertEqual(result.district, "Colombo")
        self.assertEqual(result.service_tier, "Tier 1")
        self.assertTrue(result.needs_manual_confirmation)
        self.assertIn("10:00 AM to 12:00 PM", result.summary)
        self.assertIn("Perishables are most feasible", result.item_guidance)

    def test_logistics_agent_warns_for_regional_perishable_items(self) -> None:
        result = LogisticsAgent().check_delivery("Can you deliver a fresh cream cake to Jaffna?")

        self.assertEqual(result.district, "Jaffna")
        self.assertEqual(result.service_tier, "Tier 3")
        self.assertTrue(result.needs_manual_confirmation)
        self.assertIn("2 to 3 business days", result.typical_timing)
        self.assertIn("generally restricted", result.item_guidance)

    def test_logistics_agent_standard_retail_shipping_can_avoid_manual_confirmation(self) -> None:
        result = LogisticsAgent().check_delivery("Can you deliver a speaker to Kurunegala?")

        self.assertEqual(result.district, "Kurunegala")
        self.assertEqual(result.service_tier, "Tier 3")
        self.assertFalse(result.needs_manual_confirmation)
        self.assertIn("standard courier", result.item_guidance)

    def test_router_uses_memory_context_for_logistics_follow_up(self) -> None:
        decision = KaprukaRouter(use_llm=False).route(
            "i am near kelaniya",
            memory_context="user: what are your delivery options\nassistant: I need the target Sri Lankan district before I can assess delivery feasibility.",
        )
        self.assertEqual(decision.route, "logistics_check")

    def test_router_uses_memory_context_for_delivery_follow_up_with_i_need_phrase(self) -> None:
        decision = KaprukaRouter(use_llm=False).route(
            "I need to deliver it to Gampaha",
            memory_context="user: what are the delivery options you got\nassistant: Please provide your district or city area for specific delivery options.",
        )
        self.assertEqual(decision.route, "logistics_check")

    def test_router_routes_known_location_without_memory_context(self) -> None:
        decision = KaprukaRouter(use_llm=False).route("i am near kelaniya")

        self.assertEqual(decision.route, "logistics_check")

    def test_router_routes_delivery_request_with_i_need_phrase_without_memory_context(self) -> None:
        decision = KaprukaRouter(use_llm=False).route("I need to deliver it to Gampaha")

        self.assertEqual(decision.route, "logistics_check")

    def test_logistics_agent_handles_known_location_without_memory_context(self) -> None:
        result = LogisticsAgent().check_delivery("i am near kelaniya")

        self.assertEqual(result.district, "Gampaha")
        self.assertTrue(result.supported)

    def test_orchestrator_routes_location_after_user_id_change(self) -> None:
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

            orchestrator.handle_message(
                "what are your delivery options",
                user_id="old-user",
                session_id="demo-session",
            )
            response = orchestrator.handle_message(
                "i am near kelaniya",
                user_id="new-user",
                session_id="demo-session",
            )

            self.assertEqual(response.route, "logistics_check")
            self.assertEqual(response.specialist_output["logistics"]["district"], "Gampaha")

    def test_orchestrator_keeps_delivery_follow_up_in_logistics_path(self) -> None:
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

            first = orchestrator.handle_message("what are the delivery options you got")
            second = orchestrator.handle_message("I need to deliver it to Gampaha")

            self.assertEqual(first.route, "logistics_check")
            self.assertEqual(second.route, "logistics_check")
            self.assertEqual(second.specialist_output["logistics"]["district"], "Gampaha")

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

    def test_orchestrator_handles_identity_without_short_term_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                user_store=UserProfileStore(Path(tmp_dir) / "user_profiles.json"),
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
                meta_agent=MetaAgent(use_llm=False),
            )

            response = orchestrator.handle_message("hi, who are you?")

            self.assertEqual(response.route, "identity")
            self.assertEqual(response.answer, "I am the Kapruka assistant.")
            self.assertEqual(stack.recent_context("demo-user", "demo-session"), [])
            self.assertNotIn("turn_storage", response.timings_ms)
            self.assertEqual(response.specialist_output["meta"]["source"], "fallback")

    def test_orchestrator_handles_greeting_with_short_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                user_store=UserProfileStore(Path(tmp_dir) / "user_profiles.json"),
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
                meta_agent=MetaAgent(use_llm=False),
            )

            response = orchestrator.handle_message("hi")

            self.assertEqual(response.route, "smalltalk")
            self.assertEqual(response.answer, "Hi. How can I help?")
            self.assertEqual(stack.recent_context("demo-user", "demo-session"), [])

    def test_orchestrator_remembers_user_name_from_intro(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                user_store=UserProfileStore(Path(tmp_dir) / "user_profiles.json"),
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
                meta_agent=MetaAgent(use_llm=False),
            )

            response = orchestrator.handle_message("hi I'm Jaiya")

            self.assertEqual(response.route, "smalltalk")
            self.assertEqual(response.answer, "Hi, Jaiya. How can I help?")
            profile = stack.get_user_profile("demo-user")
            assert profile is not None
            self.assertEqual(profile.name, "Jaiya")

    def test_orchestrator_saves_name_but_routes_intro_product_request_to_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                user_store=UserProfileStore(Path(tmp_dir) / "user_profiles.json"),
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
                meta_agent=MetaAgent(use_llm=False),
            )

            response = orchestrator.handle_message(
                "I am gayantha, what are your options in cakes",
                user_id="110-011",
                session_id="demo-session",
            )

            self.assertEqual(response.route, "catalog_search")
            profile = stack.get_user_profile("110-011")
            assert profile is not None
            self.assertEqual(profile.name, "Gayantha")

    def test_orchestrator_recalls_user_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                user_store=UserProfileStore(Path(tmp_dir) / "user_profiles.json"),
            )
            stack.save_user_profile("demo-user", "Jaiya")
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
                meta_agent=MetaAgent(use_llm=False),
            )

            response = orchestrator.handle_message("what is my name?")

            self.assertEqual(response.route, "identity")
            self.assertEqual(response.answer, "Your name is Jaiya.")

    def test_orchestrator_answers_assistant_call_name_contextually(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                user_store=UserProfileStore(Path(tmp_dir) / "user_profiles.json"),
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
                meta_agent=MetaAgent(use_llm=False),
            )

            response = orchestrator.handle_message("Shall I call you kapruka then?")

            self.assertEqual(response.route, "identity")
            self.assertEqual(response.answer, "Yes, you can call me Kapruka.")

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

    def test_orchestrator_updates_active_recipient_with_pronoun_allergy_correction(self) -> None:
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
            second = orchestrator.handle_message(
                "Actualy we found that dark chocaltes are alergy for her, only can eat white chocalates"
            )

            self.assertEqual(first.route, "preference_update")
            self.assertEqual(second.route, "preference_update")
            self.assertNotIn("need a recipient", second.answer.lower())
            profile = stack.get_recipient_profile("wife")
            assert profile is not None
            self.assertNotIn("Loves dark chocolate", profile.preferences)
            self.assertIn("Can eat White chocolates", profile.preferences)
            self.assertIn("Avoids Dark chocolate", profile.constraints)

    def test_catalog_search_sets_active_recipient_for_later_pronoun_update(self) -> None:
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

            first = orchestrator.handle_message("I need a gift for my wife")
            second = orchestrator.handle_message("dark chocolates are allergy for her")

            self.assertEqual(first.route, "catalog_search")
            self.assertEqual(second.route, "preference_update")
            self.assertNotIn("need a recipient", second.answer.lower())
            profile = stack.get_recipient_profile("wife")
            assert profile is not None
            self.assertIn("Avoids Dark chocolate", profile.constraints)

    def test_orchestrator_does_not_save_recommendation_feedback_as_profile_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            router = KaprukaRouter(
                llm_client=FakeRouterClient(
                    '{"intent":"preference_update","confidence":0.96,"reason":"The user is sharing a recipient note.","params":{"message":"But I think some electronic items or any tool boxes may suit better"}}'
                )
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                router=router,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
            )

            response = orchestrator.handle_message(
                "But I think some electronic items or any tool boxes may suit better",
                recipient_id="father",
                recipient_name="Father",
                relationship="parent",
            )

            self.assertEqual(response.route, "catalog_search")
            self.assertIsNone(stack.get_recipient_profile("father"))
            self.assertNotIn("updated father's profile", response.answer.lower())

    def test_orchestrator_does_not_save_love_more_feedback_as_profile_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            router = KaprukaRouter(
                llm_client=FakeRouterClient(
                    '{"intent":"preference_update","confidence":0.96,"reason":"The user is sharing a recipient note.","params":{"message":"I think she will love more to get some electronic items"}}'
                )
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                router=router,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
            )

            response = orchestrator.handle_message(
                "I think she will love more to get some electronic items",
                recipient_id="girlfriend",
                recipient_name="Girlfriend",
                relationship="partner",
            )

            self.assertEqual(response.route, "catalog_search")
            self.assertIsNone(stack.get_recipient_profile("girlfriend"))
            self.assertNotIn("updated girlfriend's profile", response.answer.lower())

    def test_orchestrator_blocks_note_only_misroute_at_profile_write_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            router = KaprukaRouter(
                llm_client=FakeRouterGuardClient(
                    route_content='{"intent":"preference_update","confidence":0.96,"reason":"The user is sharing a recipient note.","params":{"message":"I think she would really enjoy some electronic items"}}',
                    guard_content='{"should_persist":false,"kind":"recommendation_feedback","reason":"This is steering the current recommendation, not stating a lasting profile fact."}',
                )
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                router=router,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
            )

            response = orchestrator.handle_message(
                "I think she would really enjoy some electronic items",
                recipient_id="girlfriend",
                recipient_name="Girlfriend",
                relationship="partner",
            )

            self.assertEqual(response.route, "catalog_search")
            self.assertIsNone(stack.get_recipient_profile("girlfriend"))
            self.assertNotIn("updated girlfriend's profile", response.answer.lower())

    def test_orchestrator_allows_natural_note_only_profile_update_via_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            router = KaprukaRouter(
                llm_client=FakeRouterGuardClient(
                    route_content='{"intent":"preference_update","confidence":0.96,"reason":"The user is sharing a recipient note.","params":{"message":"Her style leans minimalist with clean office decor"}}',
                    guard_content='{"should_persist":true,"kind":"profile_fact","reason":"This is a stable recipient style note that should be remembered."}',
                )
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                router=router,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
            )

            response = orchestrator.handle_message(
                "Her style leans minimalist with clean office decor",
                recipient_id="girlfriend",
                recipient_name="Girlfriend",
                relationship="partner",
            )

            profile = stack.get_recipient_profile("girlfriend")
            assert profile is not None
            self.assertEqual(response.route, "preference_update")
            self.assertIn("updated girlfriend's profile", response.answer.lower())
            self.assertIn("her style leans minimalist with clean office decor", [note.lower() for note in profile.notes])

    def test_orchestrator_allows_explicit_note_only_profile_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            router = KaprukaRouter(
                llm_client=FakeRouterClient(
                    '{"intent":"preference_update","confidence":0.96,"reason":"The user is explicitly asking to remember a profile fact.","params":{"message":"Remember that her birthday is on 20th July"}}'
                )
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                router=router,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
            )

            response = orchestrator.handle_message(
                "Remember that her birthday is on 20th July",
                recipient_id="girlfriend",
                recipient_name="Girlfriend",
                relationship="partner",
            )

            profile = stack.get_recipient_profile("girlfriend")
            assert profile is not None
            self.assertEqual(response.route, "preference_update")
            self.assertIn("updated girlfriend's profile", response.answer.lower())
            self.assertIn("her birthday is on 20th july", [note.lower() for note in profile.notes])

    def test_router_keeps_wife_id_when_name_is_given(self) -> None:
        recipient = KaprukaRouter(use_llm=False).extract_recipient_reference(
            "My wife Neth's birthday is coming on 20th July"
        )

        self.assertEqual(recipient["recipient_id"], "wife")
        self.assertEqual(recipient["recipient_name"], "Neth")
        self.assertEqual(recipient["relationship"], "spouse")

    def test_catalog_reflection_removes_allergy_violations_before_answering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeUnsafeChocolateLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            stack.save_recipient_profile(
                recipient_id="wife",
                name="Wife",
                relationship="spouse",
                preferences=["Can eat White chocolates"],
                constraints=["Avoids Dark chocolate"],
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
            )

            response = orchestrator.handle_message("birthday chocolate gift for wife", recipient_id="wife")

            catalog_output = response.specialist_output["catalog"]
            product_names = [
                match["product"]["name"]
                for match in catalog_output["bundle"]["catalog_matches"]
            ]
            self.assertNotIn("Dark Chocolate Birthday Hamper", product_names)
            self.assertIn("White Chocolate Birthday Hamper", product_names)
            self.assertTrue(catalog_output["reflection"]["revised"])
            self.assertEqual(catalog_output["reflection"]["violations"][0]["product_id"], "dark1")
            self.assertIn("reflection_loop", response.timings_ms)

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
            self.assertIn("memory_relevance_gate", response.timings_ms)
            self.assertIn("retrieval_query_build", response.timings_ms)
            self.assertIn("lexical_search", response.timings_ms)
            self.assertIn("query_embedding", response.timings_ms)
            self.assertIn("qdrant_search", response.timings_ms)
            self.assertIn("result_rerank", response.timings_ms)
            self.assertIn("catalog_retrieval", response.timings_ms)
            self.assertIn("catalog_answer_generation", response.timings_ms)
            self.assertIn("total", response.timings_ms)
            self.assertTrue(response.specialist_output["catalog"]["memory_gate"]["use_recipient_profile"])

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
            self.assertFalse(response.specialist_output["catalog"]["memory_gate"]["use_recipient_profile"])

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
            self.assertTrue(response.specialist_output["catalog"]["memory_gate"]["topic_shifted"])
            self.assertFalse(response.specialist_output["catalog"]["memory_gate"]["use_short_term"])

    def test_direct_product_query_filters_unrelated_catalog_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeMixedTvLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(memory_stack=stack, chat_service=FakeChatService()),
            )

            response = orchestrator.handle_message("what are your options in led tvs")

            catalog_output = response.specialist_output["catalog"]
            product_names = [
                match["product"]["name"]
                for match in catalog_output["bundle"]["catalog_matches"]
            ]
            self.assertEqual(product_names, ["Konka 32 Inch Full Hd Led Tv Kg32ee682"])
            self.assertEqual(catalog_output["bundle"]["product_relevance_filter"]["category"], "tv")
            self.assertEqual(catalog_output["bundle"]["product_relevance_filter"]["removed_count"], 1)
            self.assertIn("Konka 32 Inch Full Hd Led Tv Kg32ee682", response.answer)
            self.assertNotIn("avoid tv", response.answer.lower())
            self.assertNotIn("non-tv", response.answer.lower())
            self.assertNotIn("Adarei Teddy", response.answer)

    def test_direct_product_query_clears_catalog_when_no_relevant_matches_exist(self) -> None:
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

            response = orchestrator.handle_message("what are your options in led tvs")

            catalog_output = response.specialist_output["catalog"]
            self.assertEqual(catalog_output["bundle"]["catalog_matches"], [])
            self.assertEqual(catalog_output["bundle"]["product_relevance_filter"]["category"], "tv")
            self.assertEqual(catalog_output["bundle"]["product_relevance_filter"]["removed_count"], 1)
            self.assertIn("could not find a strong product match", response.answer.lower())
            self.assertNotIn("Ferrero Rocher Heart Bouquet For Her", response.answer)

    def test_answer_reflection_rejects_irrelevant_furniture_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(
                    memory_stack=stack,
                    chat_service=FakeFurnitureHallucinatingChatService(),
                ),
            )

            response = orchestrator.handle_message("what furniture options do you have")

            catalog_output = response.specialist_output["catalog"]
            answer_validation = catalog_output["reflection"]["answer_validation"]
            self.assertTrue(answer_validation["checked"])
            self.assertEqual(answer_validation["source"], "llm_judge")
            self.assertTrue(answer_validation["answer_revised"])
            self.assertEqual(answer_validation["requested_terms"], ["furniture"])
            self.assertIn("Ferrero Rocher Heart Bouquet For Her", answer_validation["mentioned_products"])
            self.assertEqual(catalog_output["bundle"]["catalog_matches"], [])
            self.assertIn("could not find a strong product match", response.answer.lower())
            self.assertIn("answer_reflection_loop", response.timings_ms)

    def test_answer_reflection_does_not_block_broad_gift_discovery_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stack = CognitiveMemoryStack(
                short_term=ShortTermMemoryStore(use_database=False),
                long_term=FakeLongTermStore(),
                semantic=SemanticProfileStore(Path(tmp_dir) / "profiles.json"),
            )
            orchestrator = KaprukaOrchestrator(
                memory_stack=stack,
                catalog_agent=CatalogAgent(
                    memory_stack=stack,
                    chat_service=FakeFatherGiftChatService(),
                ),
            )

            response = orchestrator.handle_message("okay also i need a present to give to my father what can you suggest?")

            catalog_output = response.specialist_output["catalog"]
            answer_validation = catalog_output["reflection"]["answer_validation"]
            self.assertTrue(answer_validation["checked"])
            self.assertEqual(answer_validation["source"], "llm_judge")
            self.assertFalse(answer_validation["answer_revised"])
            self.assertTrue(answer_validation["answer_supported"])
            self.assertIn("broad gift-discovery", answer_validation["reason"].lower())
            self.assertIn("Ferrero Rocher Heart Bouquet For Her", response.answer)
            self.assertNotIn("could not find a strong product match", response.answer.lower())


if __name__ == "__main__":
    unittest.main()
