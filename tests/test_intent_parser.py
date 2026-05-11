import json
import unittest
from unittest.mock import patch

from intent_parser import (
    Complexity,
    DataAvailabilityStatus,
    IntentType,
    LLMMode,
    LLMSettings,
    NextAction,
    SYSTEM_PROMPT,
    _parse_intent_json_with_repair,
    _create_json_completion,
    _parse_intent_json,
    build_intent_response_repair_messages,
    create_llm_settings,
)


class IntentParserSchemaTest(unittest.TestCase):
    def test_parses_simple_data_contract(self) -> None:
        payload = {
            "schema_version": "1.1",
            "original_query": "Покажи динамику ИПЦ России за 2014-2024",
            "intent_type": "simple_data",
            "complexity": "easy",
            "topic": "Динамика индекса потребительских цен в России",
            "objects": ["Российская Федерация"],
            "geography": ["Российская Федерация"],
            "time_range": {
                "raw": "2014-2024",
                "start_year": 2014,
                "end_year": 2024,
                "is_explicit": True,
            },
            "frequency": "годовая",
            "disciplinary_perspective": "макроэкономика",
            "indicators": ["ИПЦ"],
            "indicator_specs": [
                {
                    "name": "ИПЦ",
                    "definition": "индекс потребительских цен, декабрь к декабрю предыдущего года",
                    "unit": "%",
                    "role": "primary",
                }
            ],
            "entities": ["Российская Федерация"],
            "granularity": "год",
            "research_questions": [
                "Как менялся ИПЦ в России по годам за 2014-2024 годы?"
            ],
            "research_design": None,
            "derived_metrics": [],
            "dataset_spec": {
                "row_grain": "год",
                "columns": ["год", "ИПЦ_дек_к_дек", "%", "источник", "дата_выгрузки"],
                "rows_approx": "11",
                "frequency": "годовая",
            },
            "source_candidates": [
                {
                    "name": "Росстат",
                    "dataset_or_indicator": "Индексы потребительских цен",
                    "role": "official source",
                    "notes": None,
                }
            ],
            "data_availability": {
                "status": "likely_available",
                "verdict": None,
                "reasons": [],
                "alternatives": [],
            },
            "ambiguities": [],
            "clarifying_questions": [],
            "assumptions_if_no_answer": [],
            "confidence": 0.9,
            "next_action": "proceed_with_assumptions",
        }

        intent = _parse_intent_json(json.dumps(payload, ensure_ascii=False), "fallback")

        self.assertEqual(intent.intent_type, IntentType.SIMPLE_DATA)
        self.assertEqual(intent.complexity, Complexity.EASY)
        self.assertEqual(intent.next_action, NextAction.PROCEED_WITH_ASSUMPTIONS)
        self.assertEqual(intent.dataset_spec.row_grain, "год")
        self.assertEqual(intent.indicator_specs[0].unit, "%")

    def test_parses_no_data_contract_without_numbers(self) -> None:
        payload = {
            "schema_version": "1.1",
            "original_query": "Дай данные о зарплатах в IT-секторе КНДР за 2020-2024",
            "intent_type": "no_data",
            "complexity": "medium",
            "topic": "Зарплаты в IT-секторе КНДР",
            "objects": ["IT-сектор КНДР"],
            "geography": ["КНДР"],
            "time_range": {
                "raw": "2020-2024",
                "start_year": 2020,
                "end_year": 2024,
                "is_explicit": True,
            },
            "frequency": "годовая",
            "disciplinary_perspective": "рынок труда",
            "indicators": ["зарплаты в IT-секторе"],
            "indicator_specs": [
                {
                    "name": "зарплаты в IT-секторе",
                    "definition": "официальная статистика заработной платы по отрасли",
                    "unit": None,
                    "role": "primary",
                }
            ],
            "entities": ["КНДР", "IT-сектор"],
            "granularity": "страна-год",
            "research_questions": [],
            "research_design": None,
            "derived_metrics": [],
            "dataset_spec": {
                "row_grain": "страна-год",
                "columns": ["страна", "год", "зарплата_IT", "источник"],
                "rows_approx": "5",
                "frequency": "годовая",
            },
            "source_candidates": [],
            "data_availability": {
                "status": "likely_unavailable",
                "verdict": "данные недоступны в верифицированных структурированных источниках",
                "reasons": [
                    "КНДР не публикует регулярную официальную статистику по зарплатам по отраслям"
                ],
                "alternatives": [
                    "использовать исследовательские публикации с явной пометкой, что это оценки"
                ],
            },
            "ambiguities": [],
            "clarifying_questions": [],
            "assumptions_if_no_answer": [],
            "confidence": 0.7,
            "next_action": "report_no_data",
        }

        intent = _parse_intent_json(json.dumps(payload, ensure_ascii=False), "fallback")

        self.assertEqual(intent.intent_type, IntentType.NO_DATA)
        self.assertEqual(intent.next_action, NextAction.REPORT_NO_DATA)
        self.assertEqual(
            intent.data_availability.status,
            DataAvailabilityStatus.LIKELY_UNAVAILABLE,
        )

    def test_normalizes_frequency_aliases(self) -> None:
        payload = {
            "schema_version": "1.1",
            "original_query": "Покажи показатель по годам",
            "intent_type": "simple_data",
            "complexity": "easy",
            "topic": None,
            "objects": [],
            "geography": [],
            "time_range": None,
            "frequency": "annual",
            "disciplinary_perspective": None,
            "indicators": [],
            "indicator_specs": [],
            "entities": [],
            "granularity": None,
            "research_questions": [],
            "research_design": None,
            "derived_metrics": [],
            "dataset_spec": {
                "row_grain": "год",
                "columns": [],
                "rows_approx": None,
                "frequency": "yearly",
            },
            "source_candidates": [],
            "data_availability": {
                "status": "unknown",
                "verdict": None,
                "reasons": [],
                "alternatives": [],
            },
            "ambiguities": [],
            "clarifying_questions": [],
            "assumptions_if_no_answer": [],
            "confidence": 0.5,
            "next_action": "proceed_with_assumptions",
        }

        intent = _parse_intent_json(json.dumps(payload, ensure_ascii=False), "fallback")

        self.assertEqual(intent.frequency, "годовая")
        self.assertEqual(intent.dataset_spec.frequency, "годовая")

    def test_normalizes_string_time_range_from_llm_response(self) -> None:
        payload = {
            "schema_version": "1.1",
            "original_query": "Покажи показатель за 2010-2020",
            "intent_type": "simple_data",
            "complexity": "easy",
            "topic": None,
            "objects": [],
            "geography": ["Россия"],
            "time_range": "2010-2020",
            "frequency": "annual",
            "disciplinary_perspective": None,
            "indicators": ["показатель"],
            "indicator_specs": [],
            "entities": [],
            "granularity": None,
            "research_questions": [],
            "research_design": None,
            "derived_metrics": [],
            "dataset_spec": None,
            "source_candidates": [],
            "data_availability": {
                "status": "unknown",
                "verdict": None,
                "reasons": [],
                "alternatives": [],
            },
            "ambiguities": [],
            "clarifying_questions": [],
            "assumptions_if_no_answer": [],
            "confidence": 0.5,
            "next_action": "proceed_with_assumptions",
        }

        intent = _parse_intent_json(json.dumps(payload, ensure_ascii=False), "fallback")

        self.assertEqual(intent.time_range.raw, "2010-2020")
        self.assertEqual(intent.time_range.start_year, 2010)
        self.assertEqual(intent.time_range.end_year, 2020)
        self.assertTrue(intent.time_range.is_explicit)

    def test_repairs_invalid_intent_response_without_changing_tool(self) -> None:
        invalid_payload = {
            "schema_version": "1.1",
            "original_query": "Покажи инфляцию России за 2020-2024",
            "intent_type": "simple_data",
            "complexity": "easy",
            "topic": "инфляция России",
            "objects": [],
            "geography": ["Россия"],
            "time_range": None,
            "frequency": "годовая",
            "disciplinary_perspective": None,
            "indicators": ["инфляция"],
            "indicator_specs": [],
            "entities": [],
            "granularity": None,
            "research_questions": [],
            "research_design": None,
            "derived_metrics": [],
            "dataset_spec": None,
            "source_candidates": [],
            "data_availability": {
                "status": "unknown",
                "verdict": None,
                "reasons": [],
                "alternatives": [],
            },
            "ambiguities": [],
            "clarifying_questions": [],
            "assumptions_if_no_answer": [],
            "confidence": "high",
            "next_action": "proceed_with_assumptions",
        }
        fixed_payload = dict(invalid_payload, confidence=0.8)

        class FakeCompletions:
            def __init__(self) -> None:
                self.request = None

            def create(self, **kwargs):
                self.request = kwargs

                class Message:
                    content = json.dumps(fixed_payload, ensure_ascii=False)

                class Choice:
                    message = Message()

                class Response:
                    choices = [Choice()]

                return Response()

        class FakeChat:
            def __init__(self) -> None:
                self.completions = FakeCompletions()

        class FakeClient:
            def __init__(self) -> None:
                self.chat = FakeChat()

        client = FakeClient()
        settings = LLMSettings(
            provider="yandex",
            client=client,
            mode=LLMMode.CHAT_COMPLETIONS,
            model="gpt://test-project/yandexgpt/latest",
        )

        intent = _parse_intent_json_with_repair(
            json.dumps(invalid_payload, ensure_ascii=False),
            original_query="fallback",
            settings=settings,
            max_repairs=1,
        )

        request = client.chat.completions.request
        self.assertEqual(intent.confidence, 0.8)
        self.assertIn("исправляешь только JSON-ответ", request["messages"][0]["content"])
        repair_payload = json.loads(request["messages"][1]["content"])
        self.assertIn("confidence", repair_payload["parser_error"])
        self.assertIn('"confidence": "high"', repair_payload["invalid_response"])

    def test_repair_prompt_contains_invalid_response_and_error(self) -> None:
        messages = build_intent_response_repair_messages(
            original_query="Покажи инфляцию",
            invalid_response='{"confidence": "high"}',
            parser_error="confidence should be a number",
        )

        payload = json.loads(messages[1]["content"])
        self.assertIn("ResearchIntent", messages[0]["content"])
        self.assertEqual(payload["original_query"], "Покажи инфляцию")
        self.assertEqual(payload["parser_error"], "confidence should be a number")

    def test_loads_qwen_settings_from_env(self) -> None:
        env = {
            "LLM_PROVIDER": "qwen",
            "QWEN_API_KEY": "test-key",
            "QWEN_BASE_URL": "https://example.test/v1",
            "QWEN_MODEL": "test-qwen",
        }

        with patch.dict("os.environ", env, clear=True):
            settings = create_llm_settings()

        self.assertEqual(settings.provider, "qwen")
        self.assertEqual(settings.mode, LLMMode.CHAT_COMPLETIONS)
        self.assertEqual(settings.model, "test-qwen")

    def test_loads_yandex_chat_settings_from_env(self) -> None:
        env = {
            "LLM_PROVIDER": "yandex",
            "YANDEX_API_KEY": "test-key",
            "YANDEX_BASE_URL": "https://ai.api.cloud.yandex.net/v1",
            "YANDEX_PROJECT": "test-project",
            "YANDEX_MODEL": "gpt://test-project/yandexgpt/latest",
        }

        with patch.dict("os.environ", env, clear=True):
            settings = create_llm_settings()

        self.assertEqual(settings.provider, "yandex")
        self.assertEqual(settings.mode, LLMMode.CHAT_COMPLETIONS)
        self.assertEqual(settings.model, "gpt://test-project/yandexgpt/latest")

    def test_yandex_defaults_model_from_project(self) -> None:
        env = {
            "LLM_PROVIDER": "yandex",
            "YANDEX_API_KEY": "test-key",
            "YANDEX_BASE_URL": "https://ai.api.cloud.yandex.net/v1",
            "YANDEX_PROJECT": "test-project",
        }

        with patch.dict("os.environ", env, clear=True):
            settings = create_llm_settings()

        self.assertEqual(settings.provider, "yandex")
        self.assertEqual(settings.mode, LLMMode.CHAT_COMPLETIONS)
        self.assertEqual(settings.model, "gpt://test-project/qwen3.6-35b-a3b/latest")

    def test_chat_completion_sends_system_prompt_and_json_format(self) -> None:
        class FakeCompletions:
            def __init__(self) -> None:
                self.request = None

            def create(self, **kwargs):
                self.request = kwargs

                class Message:
                    content = "{}"

                class Choice:
                    message = Message()

                class Response:
                    choices = [Choice()]

                return Response()

        class FakeChat:
            def __init__(self) -> None:
                self.completions = FakeCompletions()

        class FakeClient:
            def __init__(self) -> None:
                self.chat = FakeChat()

        client = FakeClient()
        settings = LLMSettings(
            provider="yandex",
            client=client,
            mode=LLMMode.CHAT_COMPLETIONS,
            model="gpt://test-project/yandexgpt/latest",
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Покажи инфляцию России за 2020-2024."},
        ]

        _create_json_completion(settings, messages)

        request = client.chat.completions.request
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertIn("ResearchIntent", request["messages"][0]["content"])
        self.assertEqual(request["response_format"], {"type": "json_object"})

    def test_rejects_unknown_provider(self) -> None:
        with patch.dict("os.environ", {"LLM_PROVIDER": "openai"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "qwen, yandex"):
                create_llm_settings()

    def test_parses_markdown_fenced_json_and_normalizes_null_lists(self) -> None:
        payload = """
```json
{
  "schema_version": "1.1",
  "original_query": "Дай данные по инфляции.",
  "intent_type": "simple_data",
  "complexity": "easy",
  "topic": "инфляция",
  "objects": null,
  "geography": null,
  "time_range": {
    "raw": null,
    "start_year": null,
    "end_year": null,
    "is_explicit": false
  },
  "frequency": null,
  "disciplinary_perspective": null,
  "indicators": ["инфляция"],
  "indicator_specs": null,
  "entities": null,
  "granularity": null,
  "research_questions": null,
  "research_design": null,
  "derived_metrics": null,
  "dataset_spec": {
    "row_grain": "год",
    "columns": null,
    "rows_approx": null,
    "frequency": "annual"
  },
  "source_candidates": null,
  "data_availability": {
    "status": "needs_source_check",
    "verdict": null,
    "reasons": null,
    "alternatives": null
  },
  "ambiguities": ["не указана страна"],
  "clarifying_questions": ["Какая страна нужна?"],
  "assumptions_if_no_answer": null,
  "confidence": 0.4,
  "next_action": "proceed_with_assumptions"
}
```
"""

        intent = _parse_intent_json(payload, "fallback")

        self.assertEqual(intent.intent_type, IntentType.AMBIGUOUS)
        self.assertEqual(intent.complexity, Complexity.MEDIUM)
        self.assertEqual(intent.next_action, NextAction.ASK_CLARIFICATION)
        self.assertEqual(intent.geography, [])
        self.assertEqual(intent.dataset_spec.columns, [])
        self.assertEqual(intent.dataset_spec.frequency, "годовая")


if __name__ == "__main__":
    unittest.main()
