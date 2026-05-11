import json
import unittest

from parser.intent_parser import (
    Complexity,
    DatasetSpec,
    IndicatorSpec,
    IntentType,
    LLMMode,
    LLMSettings,
    NextAction,
    ResearchIntent,
    TimeRange,
)
from orchestrator import (
    ClarificationAnswer,
    OrchestrationStatus,
    ResearchPipelineAgent,
    build_refine_intent_messages,
    prepare_intent_for_design,
    validate_intent_for_design,
)


class OrchestratorTest(unittest.TestCase):
    def test_pipeline_agent_reaches_full_flow(self) -> None:
        query = "Покажи динамику ИПЦ России за 2020-2024."
        intent = ResearchIntent(
            original_query=query,
            intent_type=IntentType.SIMPLE_DATA,
            complexity=Complexity.EASY,
            topic="динамика ИПЦ России",
            geography=["Россия"],
            time_range=TimeRange(
                raw="2020-2024",
                start_year=2020,
                end_year=2024,
                is_explicit=True,
            ),
            frequency="годовая",
            indicators=["ИПЦ"],
            indicator_specs=[
                IndicatorSpec(
                    name="ИПЦ",
                    definition="индекс потребительских цен, декабрь к декабрю предыдущего года",
                    unit="%",
                    role="primary",
                )
            ],
            dataset_spec=DatasetSpec(
                row_grain="год",
                columns=["год", "ИПЦ", "источник"],
                rows_approx="5",
                frequency="годовая",
            ),
            confidence=0.9,
            next_action=NextAction.PROCEED_WITH_ASSUMPTIONS,
        )
        design_payload = {
            "original_query": query,
            "design_summary": "Дизайн динамического ряда ИПЦ России.",
            "methodology_notes": "Использовать годовую частоту.",
            "can_continue": True,
        }
        client = _SequencedLLMClient(
            [
                intent.model_dump_json(ensure_ascii=False),
                json.dumps(design_payload, ensure_ascii=False),
            ]
        )
        settings = LLMSettings(
            provider="test",
            client=client,
            mode=LLMMode.CHAT_COMPLETIONS,
            model="test-model",
        )

        agent = ResearchPipelineAgent(settings=settings)

        result = agent.run(query, use_registry=False)

        self.assertEqual(result.status, OrchestrationStatus.DESIGN_READY)
        self.assertIsNotNone(result.research_design)
        self.assertIsNotNone(result.dataset_structure)
        self.assertIsNotNone(result.build_plan)
        self.assertIsNotNone(result.build_script)
        self.assertEqual(result.dataset_structure.row_grain, "год")
        self.assertEqual(len(client.requests), 2)
        self.assertIn("парсер исследовательского намерения", client.requests[0]["messages"][0]["content"])
        self.assertIn("модуль дизайна количественного исследования", client.requests[1]["messages"][0]["content"])

    def test_ambiguous_intent_requires_clarification_before_design(self) -> None:
        intent = ResearchIntent(
            original_query="Дай данные по инфляции.",
            intent_type=IntentType.AMBIGUOUS,
            complexity=Complexity.MEDIUM,
            topic="инфляция",
            indicators=["инфляция"],
            clarifying_questions=[
                "Какая страна или регион вас интересует?",
                "За какой период нужны данные?",
            ],
            ambiguities=[
                "Не указана география.",
                "Не указан период.",
            ],
            assumptions_if_no_answer=[
                "география — Российская Федерация",
                "период — последние 10 лет",
            ],
            confidence=0.5,
            next_action=NextAction.ASK_CLARIFICATION,
        )

        result = prepare_intent_for_design(intent)

        self.assertEqual(result.status, OrchestrationStatus.NEEDS_CLARIFICATION)
        self.assertIsNone(result.research_design)
        self.assertTrue(result.clarification_requests)

    def test_complete_intent_is_ready_for_design(self) -> None:
        intent = ResearchIntent(
            original_query="Покажи динамику ИПЦ России за 2014-2024 годы.",
            intent_type=IntentType.SIMPLE_DATA,
            complexity=Complexity.EASY,
            topic="динамика ИПЦ России",
            geography=["Россия"],
            time_range=TimeRange(
                raw="2014-2024",
                start_year=2014,
                end_year=2024,
                is_explicit=True,
            ),
            frequency="годовая",
            indicators=["ИПЦ"],
            indicator_specs=[
                IndicatorSpec(
                    name="ИПЦ",
                    definition="индекс потребительских цен, декабрь к декабрю предыдущего года",
                    unit="%",
                    role="primary",
                )
            ],
            dataset_spec=DatasetSpec(
                row_grain="год",
                columns=["год", "ИПЦ", "источник"],
                rows_approx="11",
                frequency="годовая",
            ),
            confidence=0.95,
            next_action=NextAction.PROCEED_WITH_ASSUMPTIONS,
        )

        result = prepare_intent_for_design(intent)

        self.assertEqual(result.status, OrchestrationStatus.READY_FOR_DESIGN)
        self.assertFalse(result.clarification_requests)

    def test_incomplete_indicator_spec_is_reasked(self) -> None:
        intent = ResearchIntent(
            original_query="Покажи инфляцию России за 2020-2024.",
            intent_type=IntentType.SIMPLE_DATA,
            complexity=Complexity.EASY,
            topic="инфляция России",
            geography=["Россия"],
            time_range=TimeRange(
                raw="2020-2024",
                start_year=2020,
                end_year=2024,
                is_explicit=True,
            ),
            frequency="годовая",
            indicators=["инфляция"],
            indicator_specs=[
                IndicatorSpec(
                    name="инфляция",
                    definition=None,
                    unit=None,
                    role="primary",
                )
            ],
            dataset_spec=DatasetSpec(
                row_grain="год",
                columns=["год", "инфляция", "источник"],
                rows_approx="5",
                frequency="годовая",
            ),
            confidence=0.8,
            next_action=NextAction.PROCEED_WITH_ASSUMPTIONS,
        )

        requests = validate_intent_for_design(intent)

        self.assertTrue(any(item.field == "indicator_specs" for item in requests))

    def test_incomplete_dataset_spec_is_reasked(self) -> None:
        intent = ResearchIntent(
            original_query="Покажи ВВП России за 2020-2024.",
            intent_type=IntentType.SIMPLE_DATA,
            complexity=Complexity.EASY,
            topic="ВВП России",
            geography=["Россия"],
            time_range=TimeRange(
                raw="2020-2024",
                start_year=2020,
                end_year=2024,
                is_explicit=True,
            ),
            frequency="годовая",
            indicators=["ВВП"],
            indicator_specs=[
                IndicatorSpec(
                    name="ВВП",
                    definition="валовой внутренний продукт в текущих ценах",
                    unit="руб.",
                    role="primary",
                )
            ],
            dataset_spec=DatasetSpec(
                row_grain=None,
                columns=[],
                rows_approx="5",
                frequency="годовая",
            ),
            confidence=0.85,
            next_action=NextAction.PROCEED_WITH_ASSUMPTIONS,
        )

        requests = validate_intent_for_design(intent)

        self.assertTrue(any(item.field == "dataset_spec" for item in requests))

    def test_use_defaults_allows_ready_status(self) -> None:
        intent = ResearchIntent(
            original_query="Дай данные по инфляции.",
            intent_type=IntentType.AMBIGUOUS,
            complexity=Complexity.MEDIUM,
            topic="инфляция",
            indicators=["инфляция"],
            clarifying_questions=["Какая страна нужна?"],
            ambiguities=["Не указана география."],
            assumptions_if_no_answer=["география — Российская Федерация"],
            confidence=0.5,
            next_action=NextAction.ASK_CLARIFICATION,
        )

        result = prepare_intent_for_design(intent, use_defaults=True)

        self.assertEqual(result.status, OrchestrationStatus.READY_FOR_DESIGN)

    def test_use_defaults_still_reasks_when_default_is_missing(self) -> None:
        intent = ResearchIntent(
            original_query="Дай данные по инфляции.",
            intent_type=IntentType.AMBIGUOUS,
            complexity=Complexity.MEDIUM,
            topic="инфляция",
            indicators=["инфляция"],
            clarifying_questions=["Какая страна нужна?"],
            ambiguities=["Не указана география."],
            confidence=0.5,
            next_action=NextAction.ASK_CLARIFICATION,
        )

        result = prepare_intent_for_design(intent, use_defaults=True)

        self.assertEqual(result.status, OrchestrationStatus.NEEDS_CLARIFICATION)
        self.assertTrue(result.clarification_requests)

    def test_clarification_defaults_match_fields_and_deduplicate(self) -> None:
        intent = ResearchIntent(
            original_query="Дай данные по инфляции.",
            intent_type=IntentType.AMBIGUOUS,
            complexity=Complexity.MEDIUM,
            topic="инфляция",
            indicators=["инфляция"],
            clarifying_questions=[
                "Для какой страны или региона нужны данные?",
                "Какой период времени вас интересует?",
                "Какая частота данных требуется?",
            ],
            ambiguities=[
                "Отсутствует указание страны или региона",
                "Не указан временной период",
                "Не указана частота",
            ],
            assumptions_if_no_answer=[
                "Страна: Россия",
                "Период: последние 10 лет",
                "Частота: годовая",
            ],
            confidence=0.5,
            next_action=NextAction.ASK_CLARIFICATION,
        )

        result = prepare_intent_for_design(intent)
        by_field = {request.field: request for request in result.clarification_requests}

        self.assertEqual(set(by_field), {"geography", "time_range", "frequency"})
        self.assertEqual(by_field["geography"].default_assumption, "Страна: Россия")
        self.assertEqual(by_field["time_range"].default_assumption, "Период: последние 10 лет")
        self.assertEqual(by_field["frequency"].default_assumption, "Частота: годовая")

    def test_refine_prompt_contains_previous_intent_and_answers(self) -> None:
        intent = ResearchIntent(
            original_query="Сравни ВВП в США и России",
            intent_type=IntentType.COMPARATIVE,
            complexity=Complexity.MEDIUM,
            topic="сравнение ВВП",
            geography=["США", "Россия"],
            indicators=["ВВП"],
            clarifying_questions=["За какой период сравнить ВВП?"],
            ambiguities=["Не указан период."],
            confidence=0.8,
            next_action=NextAction.ASK_CLARIFICATION,
        )
        answers = [
            ClarificationAnswer(
                field="time_range",
                question="За какой период сравнить ВВП?",
                answer="2010-2024, годовая частота",
            )
        ]

        messages = build_refine_intent_messages(intent, answers)
        system_message = messages[0]["content"]
        user_message = messages[1]["content"]

        self.assertIn("Ты обновляешь JSON первого этапа ResearchIntent", system_message)
        self.assertIn("ПРЕДЫДУЩИЙ ResearchIntent", user_message)
        self.assertIn("ОТВЕТЫ ПОЛЬЗОВАТЕЛЯ", user_message)
        self.assertIn("2010-2024", user_message)
        self.assertIn('"intent_type": "comparative"', user_message)


class _SequencedLLMClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.requests: list[dict] = []
        self.chat = _SequencedChat(self)


class _SequencedChat:
    def __init__(self, owner: _SequencedLLMClient) -> None:
        self.completions = _SequencedCompletions(owner)


class _SequencedCompletions:
    def __init__(self, owner: _SequencedLLMClient) -> None:
        self.owner = owner

    def create(self, **kwargs):
        self.owner.requests.append(kwargs)
        if not self.owner.contents:
            raise AssertionError("No fake LLM responses left.")
        content = self.owner.contents.pop(0)

        class Message:
            pass

        class Choice:
            pass

        class Response:
            pass

        message = Message()
        message.content = content
        choice = Choice()
        choice.message = message
        response = Response()
        response.choices = [choice]
        return response


if __name__ == "__main__":
    unittest.main()
