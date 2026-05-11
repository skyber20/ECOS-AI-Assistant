import os
import unittest
from unittest.mock import patch

import hybrid_candidate_retriever as retriever
from intent_parser import Complexity, IntentType, KeywordSynonyms, NextAction, ResearchIntent


class FakeChromaDir:
    def exists(self) -> bool:
        return True

    def __str__(self) -> str:
        return "/tmp/chroma"


class HybridCandidateRetrieverTest(unittest.TestCase):
    def tearDown(self) -> None:
        retriever._VECTOR_LAST_ERROR = None

    def test_research_intent_prioritizes_focused_english_queries(self) -> None:
        intent = ResearchIntent(
            original_query="Покажи ВВП США",
            english_query="Show GDP in the United States",
            keyword_synonyms=[
                KeywordSynonyms(
                    keyword="ВВП",
                    english_keyword="GDP",
                    synonyms=["gross domestic product", "current US dollars"],
                )
            ],
            intent_type=IntentType.SIMPLE_DATA,
            complexity=Complexity.EASY,
            indicators=["ВВП"],
            confidence=0.8,
            next_action=NextAction.PROCEED_WITH_ASSUMPTIONS,
        )

        vector_queries, bm25_queries = retriever._retrieval_queries(intent)

        self.assertEqual(vector_queries[0], "Show GDP in the United States")
        self.assertEqual(bm25_queries[1], "GDP gross domestic product current US dollars")
        self.assertEqual(bm25_queries[-1], "Покажи ВВП США")

    def test_vector_search_records_unavailability_in_fallback_mode(self) -> None:
        with (
            patch.object(retriever, "CHROMA_DIR", FakeChromaDir()),
            patch.dict(os.environ, {retriever.VECTOR_STRICT_ENV: "false"}),
            patch.object(retriever, "_embedding_model", side_effect=OSError("missing model")),
        ):
            results = retriever._search_vector("inflation United States", top_k=3)

        self.assertEqual(results, [])
        self.assertIn("missing model", retriever._VECTOR_LAST_ERROR)

    def test_vector_search_can_be_strict(self) -> None:
        with (
            patch.object(retriever, "CHROMA_DIR", FakeChromaDir()),
            patch.dict(os.environ, {retriever.VECTOR_STRICT_ENV: "true"}),
            patch.object(retriever, "_embedding_model", side_effect=OSError("missing model")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Vector search is unavailable"):
                retriever._search_vector("inflation United States", top_k=3)


if __name__ == "__main__":
    unittest.main()
