import json
import unittest

from dataset_reranker import _parse_rerank_json
from parser.intent_parser import _parse_intent_json
from designer.research_designer import _parse_design_json
from tests.test_dataset_pipeline import sample_design, sample_intent


class SchemaSanitizerTest(unittest.TestCase):
    def test_intent_parser_ignores_extra_llm_lists(self) -> None:
        payload = {
            "schema_version": "1.1",
            "original_query": "Покажи ВРП России",
            "intent_type": "simple_data",
            "complexity": "easy",
            "topic": "ВРП России",
            "geography": ["Россия"],
            "time_range": {
                "raw": "2020-2024",
                "start_year": 2020,
                "end_year": 2024,
                "is_explicit": True,
                "unexpected_nested": [],
            },
            "frequency": "годовая",
            "indicators": ["ВРП"],
            "dataset_spec": {
                "row_grain": "страна-год",
                "columns": ["страна", "год", "ВРП"],
                "frequency": "годовая",
                "extra_columns": [],
            },
            "confidence": 0.9,
            "next_action": "proceed_with_assumptions",
            "debug": [],
        }

        intent = _parse_intent_json(json.dumps(payload, ensure_ascii=False), "fallback")

        self.assertEqual(intent.topic, "ВРП России")
        self.assertEqual(intent.dataset_spec.row_grain, "страна-год")

    def test_design_parser_ignores_extra_nested_fields(self) -> None:
        intent = sample_intent()
        payload = sample_design(intent).model_dump(mode="json")
        payload["unexpected"] = []
        payload["hypotheses"] = [
            {
                "id": "H1",
                "statement": "ВРП меняется по годам.",
                "null_hypothesis": "ВРП не меняется.",
                "direction": "none",
                "variables": ["ВРП"],
                "extra_nested": [],
            }
        ]

        design = _parse_design_json(json.dumps(payload, ensure_ascii=False), intent)

        self.assertEqual(design.hypotheses[0].id, "H1")

    def test_reranker_parser_ignores_extra_fields(self) -> None:
        payload = {
            "results": [],
            "no_results_reason": "Нет релевантных кандидатов.",
            "debug": [],
        }

        response = _parse_rerank_json(json.dumps(payload, ensure_ascii=False))

        self.assertEqual(response.results, [])
        self.assertEqual(response.no_results_reason, "Нет релевантных кандидатов.")


if __name__ == "__main__":
    unittest.main()
