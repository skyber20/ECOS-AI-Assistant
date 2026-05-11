import io
import unittest
from unittest.mock import patch

from agent.intent_parser import (
    Complexity,
    DatasetSpec,
    IndicatorSpec,
    IntentType,
    NextAction,
    ResearchIntent,
    TimeRange,
)
from agent.main import run_interactive_research_flow
from agent.orchestrator import OrchestrationResult, OrchestrationStatus


class MainFlowTest(unittest.TestCase):
    def test_reasks_and_starts_design_only_with_refined_intent(self) -> None:
        initial_intent = ResearchIntent(
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
            confidence=0.5,
            next_action=NextAction.ASK_CLARIFICATION,
        )
        refined_intent = ResearchIntent(
            original_query="Дай данные по инфляции.",
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
                    definition="индекс потребительских цен, декабрь к декабрю предыдущего года",
                    unit="%",
                    role="primary",
                )
            ],
            dataset_spec=DatasetSpec(
                row_grain="год",
                columns=["год", "инфляция", "источник"],
                rows_approx="5",
                frequency="годовая",
            ),
            confidence=0.9,
            next_action=NextAction.PROCEED_WITH_ASSUMPTIONS,
        )
        design_result = OrchestrationResult(
            status=OrchestrationStatus.DESIGN_READY,
            intent=refined_intent,
        )
        agent = _FakeResearchAgent(initial_intent, refined_intent, design_result)

        with patch("agent.main.LangGraphResearchAgent", return_value=agent):
            result = run_interactive_research_flow(
                "Дай данные по инфляции.",
                input_stream=io.StringIO("Россия\n2020-2024\n"),
                error_stream=io.StringIO(),
            )

        self.assertEqual(result.status, OrchestrationStatus.DESIGN_READY)
        self.assertEqual(agent.refine_calls, 1)
        self.assertEqual(agent.continue_calls, 1)
        self.assertIs(agent.continued_intent, refined_intent)


class _FakeResearchAgent:
    def __init__(
        self,
        initial_intent: ResearchIntent,
        refined_intent: ResearchIntent,
        design_result: OrchestrationResult,
    ) -> None:
        self.initial_intent = initial_intent
        self.refined_intent = refined_intent
        self.design_result = design_result
        self.refine_calls = 0
        self.continue_calls = 0
        self.continued_intent = None

    def parse_intent(self, query: str) -> ResearchIntent:
        return self.initial_intent

    def refine_intent(
        self,
        intent: ResearchIntent,
        answers: list,
    ) -> ResearchIntent:
        self.refine_calls += 1
        return self.refined_intent

    def continue_from_intent(self, intent: ResearchIntent) -> OrchestrationResult:
        self.continue_calls += 1
        self.continued_intent = intent
        return self.design_result


if __name__ == "__main__":
    unittest.main()
