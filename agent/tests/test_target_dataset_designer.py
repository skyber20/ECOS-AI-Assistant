import json
import unittest

from agent.intent_parser import (
    Complexity,
    DatasetSpec,
    IndicatorSpec,
    IntentType,
    NextAction,
    ResearchIntent,
    TimeRange,
)
from agent.research_designer import ResearchStudyDesign
from agent.target_dataset_designer import (
    _parse_target_dataset_json,
    build_target_dataset_messages,
)


class TargetDatasetDesignerTest(unittest.TestCase):
    def test_target_dataset_prompt_contains_intent_design_and_schema(self) -> None:
        intent = _sample_intent()
        design = _sample_design(intent)

        messages = build_target_dataset_messages(intent, design)
        system_content = messages[0]["content"]
        user_content = messages[1]["content"]

        self.assertIn("структуры целевого датасета", system_content)
        self.assertIn("JSON Schema", system_content)
        self.assertIn("ResearchIntent", user_content)
        self.assertIn("ResearchStudyDesign", user_content)
        self.assertIn('"intent_type": "simple_data"', user_content)

    def test_parses_target_dataset_and_applies_intent_defaults(self) -> None:
        intent = _sample_intent()
        design = _sample_design(intent)
        payload = {
            "dataset_name": "ipc_russia_2020_2024",
            "dataset_purpose": "Годовой ряд ИПЦ России для анализа динамики.",
            "row_grain": "год",
            "primary_key": ["year", "country"],
            "dimensions": [
                {
                    "name": "year",
                    "title": "Год",
                    "role": "time",
                    "data_type": "year",
                    "unit": None,
                    "definition": "Год наблюдения.",
                    "nullable": False,
                    "is_required": True,
                }
            ],
            "indicators": [
                {
                    "name": "cpi_dec_to_dec",
                    "title": "ИПЦ декабрь к декабрю",
                    "role": "indicator",
                    "data_type": "float",
                    "unit": "%",
                    "definition": "Индекс потребительских цен, декабрь к декабрю предыдущего года.",
                    "nullable": True,
                    "is_required": True,
                }
            ],
            "metadata_columns": [
                {
                    "name": "source_name",
                    "title": "Источник",
                    "role": "source_metadata",
                    "data_type": "string",
                    "unit": None,
                    "definition": "Название источника данных.",
                    "nullable": False,
                    "is_required": True,
                }
            ],
            "validation_rules": ["primary_key должен быть уникален"],
            "can_build": True,
        }

        structure = _parse_target_dataset_json(
            json.dumps(payload, ensure_ascii=False),
            intent,
            design,
        )

        self.assertEqual(structure.original_query, intent.original_query)
        self.assertEqual(structure.frequency, "годовая")
        self.assertEqual(structure.geography_coverage, ["Россия"])
        self.assertEqual(structure.expected_row_count, "5")
        self.assertEqual(structure.indicators[0].unit, "%")


def _sample_intent() -> ResearchIntent:
    return ResearchIntent(
        original_query="Покажи динамику ИПЦ России за 2020-2024.",
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


def _sample_design(intent: ResearchIntent) -> ResearchStudyDesign:
    return ResearchStudyDesign(
        original_query=intent.original_query,
        design_summary="Дизайн динамического ряда ИПЦ России.",
        methodology_notes="Использовать годовую частоту.",
        required_row_grain=["год"],
    )


if __name__ == "__main__":
    unittest.main()
