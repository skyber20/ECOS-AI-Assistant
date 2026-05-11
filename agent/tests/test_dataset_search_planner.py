import unittest

from agent.dataset_search_planner import (
    BuildStrategy,
    DatasetCandidate,
    EmptyDatasetRegistry,
    InMemoryDatasetRegistry,
    build_dataset_search_request,
    create_dataset_match_report,
)
from agent.intent_parser import (
    Complexity,
    DatasetSpec,
    IndicatorSpec,
    IntentType,
    NextAction,
    ResearchIntent,
    SourceCandidate,
    TimeRange,
)
from agent.research_designer import ResearchStudyDesign, RequiredMeasurement
from agent.target_dataset_designer import DatasetColumn, TargetDatasetStructure


class DatasetSearchPlannerTest(unittest.TestCase):
    def test_builds_search_request_from_target_dataset_contract(self) -> None:
        intent = _sample_intent()
        design = _sample_design(intent)
        target = _sample_target_structure(intent)

        request = build_dataset_search_request(intent, design, target)

        self.assertEqual(request.target_dataset_name, "ipc_russia_2020_2024")
        self.assertEqual(request.row_grain, "год")
        self.assertEqual(request.primary_key, ["year", "country"])
        self.assertEqual(request.geography_coverage, ["Россия"])
        self.assertEqual(request.required_indicators[0].name, "cpi_dec_to_dec")
        self.assertIn("Росстат", request.source_requirements)
        self.assertIn("primary_key должен быть уникален", request.validation_rules)

    def test_empty_registry_returns_source_discovery_plan(self) -> None:
        intent = _sample_intent()
        design = _sample_design(intent)
        target = _sample_target_structure(intent)

        report = create_dataset_match_report(
            intent=intent,
            design=design,
            target=target,
            registry=EmptyDatasetRegistry(),
        )

        self.assertEqual(report.registry_backend, "empty_registry")
        self.assertEqual(report.build_plan.strategy, BuildStrategy.NEEDS_SOURCE_DISCOVERY)
        self.assertFalse(report.build_plan.can_build)
        self.assertTrue(report.rag_required)
        self.assertTrue(report.build_plan.predecessor_requirements)

    def test_ready_candidate_becomes_ready_dataset_plan(self) -> None:
        intent = _sample_intent()
        design = _sample_design(intent)
        target = _sample_target_structure(intent)
        registry = InMemoryDatasetRegistry(
            [
                DatasetCandidate(
                    dataset_id="rosstat_cpi_ready",
                    title="ИПЦ России по годам",
                    source_name="Росстат",
                    row_grain="год",
                    time_coverage="2020-2024",
                    geography_coverage=["Россия"],
                    frequency="годовая",
                    columns=["year", "country", "cpi_dec_to_dec", "source_name"],
                    indicators=["cpi_dec_to_dec"],
                    dimensions=["year", "country"],
                    join_keys=["year", "country"],
                )
            ]
        )

        report = create_dataset_match_report(
            intent=intent,
            design=design,
            target=target,
            registry=registry,
        )

        self.assertEqual(report.build_plan.strategy, BuildStrategy.USE_READY_DATASET)
        self.assertTrue(report.build_plan.can_build)
        self.assertEqual(report.build_plan.ready_dataset_id, "rosstat_cpi_ready")
        self.assertFalse(report.rag_required)


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
        source_candidates=[
            SourceCandidate(
                name="Росстат",
                dataset_or_indicator="ИПЦ",
                role="primary",
            )
        ],
        confidence=0.9,
        next_action=NextAction.PROCEED_WITH_ASSUMPTIONS,
    )


def _sample_design(intent: ResearchIntent) -> ResearchStudyDesign:
    return ResearchStudyDesign(
        original_query=intent.original_query,
        design_summary="Дизайн динамического ряда ИПЦ России.",
        required_measurements=[
            RequiredMeasurement(
                name="ИПЦ",
                definition="Индекс потребительских цен.",
                unit="%",
                role="primary",
                source_candidates=["Росстат"],
            )
        ],
        methodology_notes="Использовать годовую частоту.",
        required_row_grain=["год"],
    )


def _sample_target_structure(intent: ResearchIntent) -> TargetDatasetStructure:
    return TargetDatasetStructure(
        original_query=intent.original_query,
        dataset_name="ipc_russia_2020_2024",
        dataset_purpose="Годовой ряд ИПЦ России для анализа динамики.",
        row_grain="год",
        primary_key=["year", "country"],
        time_coverage="2020-2024",
        geography_coverage=["Россия"],
        frequency="годовая",
        dimensions=[
            DatasetColumn(
                name="year",
                title="Год",
                role="time",
                data_type="year",
                definition="Год наблюдения.",
                nullable=False,
                is_required=True,
            ),
            DatasetColumn(
                name="country",
                title="Страна",
                role="geography",
                data_type="string",
                definition="Страна наблюдения.",
                nullable=False,
                is_required=True,
            ),
        ],
        indicators=[
            DatasetColumn(
                name="cpi_dec_to_dec",
                title="ИПЦ декабрь к декабрю",
                role="indicator",
                data_type="float",
                unit="%",
                definition="Индекс потребительских цен, декабрь к декабрю предыдущего года.",
                nullable=True,
                is_required=True,
                source="Росстат",
            )
        ],
        metadata_columns=[
            DatasetColumn(
                name="source_name",
                title="Источник",
                role="source_metadata",
                data_type="string",
                definition="Название источника данных.",
                nullable=False,
                is_required=True,
            )
        ],
        source_requirements=["официальная статистика"],
        validation_rules=["primary_key должен быть уникален"],
    )


if __name__ == "__main__":
    unittest.main()
