import io
import unittest
from unittest.mock import patch

from intent_parser import (
    Complexity,
    DatasetSpec,
    IndicatorSpec,
    IntentType,
    NextAction,
    ResearchIntent,
    TimeRange,
)
from main import run_interactive_research_flow
from orchestrator import OrchestrationResult, OrchestrationStatus


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

        with (
            patch("main.parse_research_intent", return_value=initial_intent),
            patch("main.refine_intent_with_clarifications", return_value=refined_intent) as refine,
            patch("main.continue_research_flow", return_value=design_result) as continue_flow,
        ):
            result = run_interactive_research_flow(
                "Дай данные по инфляции.",
                input_stream=io.StringIO("Россия\n2020-2024\n"),
                error_stream=io.StringIO(),
            )

        self.assertEqual(result.status, OrchestrationStatus.DESIGN_READY)
        self.assertEqual(refine.call_count, 1)
        continue_flow.assert_called_once()
        self.assertIs(continue_flow.call_args.args[0], refined_intent)


if __name__ == "__main__":
    unittest.main()
