import json
import unittest

from intent_parser import Complexity, IntentType, NextAction, ResearchIntent, TimeRange
from research_designer import _parse_design_json, build_research_design_messages


class ResearchDesignerSchemaTest(unittest.TestCase):
    def test_design_prompt_contains_original_query_and_first_stage_json(self) -> None:
        intent = ResearchIntent(
            original_query="Как связан уровень урбанизации и рождаемость по странам мира?",
            intent_type=IntentType.RESEARCH,
            complexity=Complexity.COMPLEX,
            topic="Связь урбанизации и рождаемости",
            geography=["мир"],
            indicators=["урбанизация", "рождаемость"],
            confidence=0.9,
            next_action=NextAction.PROCEED_WITH_ASSUMPTIONS,
        )

        messages = build_research_design_messages(intent)
        user_content = messages[1]["content"]

        self.assertIn("=== ИСХОДНЫЙ ЗАПРОС ===", user_content)
        self.assertIn(intent.original_query, user_content)
        self.assertIn("=== JSON ПЕРВОГО ЭТАПА: ResearchIntent ===", user_content)
        self.assertIn('"intent_type": "research"', user_content)
        self.assertIn('"indicators"', user_content)

    def test_parses_design_with_hypotheses(self) -> None:
        intent = ResearchIntent(
            original_query="Как связан уровень урбанизации и рождаемость по странам мира?",
            intent_type=IntentType.RESEARCH,
            complexity=Complexity.COMPLEX,
            topic="Связь урбанизации и рождаемости",
            geography=["мир"],
            time_range=TimeRange(raw=None, start_year=None, end_year=None, is_explicit=False),
            indicators=["урбанизация", "рождаемость"],
            confidence=0.9,
            next_action=NextAction.PROCEED_WITH_ASSUMPTIONS,
        )
        payload = {
            "original_query": intent.original_query,
            "design_summary": "Панельное или кросс-страновое исследование связи урбанизации и рождаемости.",
            "hypotheses": [
                {
                    "id": "H1",
                    "statement": "Более высокий уровень урбанизации связан с более низкой рождаемостью.",
                    "null_hypothesis": "Связь между урбанизацией и рождаемостью отсутствует.",
                    "direction": "negative",
                    "variables": ["урбанизация", "рождаемость"],
                    "expected_effect": "отрицательная корреляция",
                }
            ],
            "required_measurements": [
                {
                    "name": "урбанизация",
                    "definition": "доля городского населения",
                    "unit": "%",
                    "role": "primary",
                    "source_candidates": ["World Bank"],
                    "is_critical": True,
                }
            ],
            "grouping_rules": [
                {
                    "dimension": "группа дохода",
                    "levels": ["низкий доход", "средний доход", "высокий доход"],
                    "rationale": "Доход может менять силу связи.",
                    "min_observations": 20,
                }
            ],
            "derived_metrics": [],
            "statistical_methods": [
                {
                    "method": "correlation",
                    "purpose": "оценить направление и силу связи",
                    "variables": ["урбанизация", "рождаемость"],
                    "hypothesis_id": "H1",
                    "expected_output": "коэффициент корреляции",
                    "assumptions": [],
                }
            ],
            "visualizations": [
                {
                    "id": "V1",
                    "title": "Урбанизация и рождаемость по странам",
                    "chart_type": "scatter",
                    "x_axis": "урбанизация",
                    "y_axis": "рождаемость",
                    "grouping": "группа дохода",
                    "metrics_used": ["урбанизация", "рождаемость"],
                    "hypothesis_id": "H1",
                    "interpretation_guide": "Нисходящий тренд поддерживает H1.",
                }
            ],
            "data_quality_checks": [
                {
                    "check": "пропуски",
                    "method": "посчитать долю пропусков",
                    "action_if_failed": "исключить страны с недостаточным покрытием",
                }
            ],
            "methodology_notes": "Если период не задан, использовать последнее доступное наблюдение.",
            "required_row_grain": ["страна-год"],
            "can_continue": True,
            "blocking_reasons": [],
        }

        design = _parse_design_json(json.dumps(payload, ensure_ascii=False), intent)

        self.assertTrue(design.can_continue)
        self.assertEqual(design.hypotheses[0].id, "H1")
        self.assertEqual(design.visualizations[0].chart_type, "scatter")

    def test_no_data_design_is_blocked(self) -> None:
        intent = ResearchIntent(
            original_query="Дай зарплаты IT КНДР",
            intent_type=IntentType.NO_DATA,
            complexity=Complexity.MEDIUM,
            indicators=["зарплаты IT"],
            confidence=0.7,
            next_action=NextAction.REPORT_NO_DATA,
        )
        payload = {
            "original_query": intent.original_query,
            "design_summary": "Дизайн невозможен без верифицированных данных.",
            "hypotheses": [],
            "required_measurements": [],
            "grouping_rules": [],
            "derived_metrics": [],
            "statistical_methods": [],
            "visualizations": [],
            "data_quality_checks": [],
            "methodology_notes": "Нет открытых структурированных данных.",
            "required_row_grain": [],
            "can_continue": True,
            "blocking_reasons": [],
        }

        design = _parse_design_json(json.dumps(payload, ensure_ascii=False), intent)

        self.assertFalse(design.can_continue)
        self.assertTrue(design.blocking_reasons)


if __name__ == "__main__":
    unittest.main()
