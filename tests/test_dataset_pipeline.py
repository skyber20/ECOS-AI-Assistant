import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import assembly_planner
import hybrid_candidate_retriever
from artifact_writer import write_orchestration_artifacts
from assembly_planner import plan_dataset_build
from catalog_bm25_indexer import tokenize_query
from dataset_structure import DatasetColumn, TargetDatasetStructure, build_target_dataset_structure
from hybrid_candidate_retriever import retrieve_candidate_datasets
from parser.intent_parser import (
    Complexity,
    DatasetSpec,
    IndicatorSpec,
    IntentType,
    NextAction,
    ResearchIntent,
    SourceCandidate,
    TimeRange,
)
from designer.research_designer import ResearchStudyDesign
from orchestrator import OrchestrationResult, OrchestrationStatus
from script_generator import generate_build_script


def sample_intent() -> ResearchIntent:
    return ResearchIntent(
        original_query="Дай динамику ВРП Архангельской области за 2015-2024",
        intent_type=IntentType.SIMPLE_DATA,
        complexity=Complexity.EASY,
        topic="динамика ВРП Архангельской области",
        objects=["Архангельская область"],
        geography=["Архангельская область"],
        time_range=TimeRange(
            raw="2015-2024",
            start_year=2015,
            end_year=2024,
            is_explicit=True,
        ),
        frequency="годовая",
        indicators=["ВРП"],
        indicator_specs=[
            IndicatorSpec(
                name="ВРП",
                definition="валовой региональный продукт в текущих основных ценах",
                unit="млн руб.",
                role="primary",
            )
        ],
        granularity="регион-год",
        dataset_spec=DatasetSpec(
            row_grain="регион-год",
            columns=["регион", "год", "ВРП", "источник", "дата_выгрузки"],
            rows_approx="10",
            frequency="годовая",
        ),
        source_candidates=[
            SourceCandidate(
                name="Росстат",
                dataset_or_indicator="Валовой региональный продукт",
                role="official source",
                notes="Официальный кандидат источника для регионального показателя.",
            )
        ],
        confidence=0.95,
        next_action=NextAction.PROCEED_WITH_ASSUMPTIONS,
    )


def sample_design(intent: ResearchIntent) -> ResearchStudyDesign:
    return ResearchStudyDesign.model_validate(
        {
            "original_query": intent.original_query,
            "design_summary": "Ряд ВРП по Архангельской области.",
            "hypotheses": [],
            "required_measurements": [
                {
                    "name": "ВРП",
                    "definition": "валовой региональный продукт в текущих основных ценах",
                    "unit": "млн руб.",
                    "role": "primary",
                    "source_candidates": ["Росстат"],
                    "is_critical": True,
                }
            ],
            "grouping_rules": [],
            "derived_metrics": [],
            "statistical_methods": [],
            "visualizations": [],
            "data_quality_checks": [],
            "methodology_notes": "Сравнивать только в одной методологии.",
            "required_row_grain": ["регион-год"],
            "can_continue": True,
            "blocking_reasons": [],
        }
    )


class DatasetPipelineTest(unittest.TestCase):
    def test_builds_target_dataset_structure_from_intent_and_design(self) -> None:
        intent = sample_intent()
        design = sample_design(intent)

        structure = build_target_dataset_structure(intent, design)
        columns = [column.name for column in structure.columns]

        self.assertEqual(structure.row_grain, "регион-год")
        self.assertEqual(structure.primary_key, ["geo", "year"])
        self.assertIn("vrp", columns)
        self.assertIn("source_url", columns)
        self.assertEqual(structure.expected_frequency, "годовая")

    def test_planner_falls_back_when_registry_is_missing(self) -> None:
        intent = sample_intent()
        design = sample_design(intent)
        structure = build_target_dataset_structure(intent, design)

        with patch("assembly_planner.CATALOG_RECORDS_PATH", Path("/missing/catalog.jsonl")):
            plan = plan_dataset_build(intent, design, structure)

        self.assertFalse(plan.registry_checked)
        self.assertTrue(plan.predecessor_datasets)
        self.assertTrue(plan.blocking_reasons)
        self.assertTrue(any(entry.decision == "unavailable" for entry in plan.source_audit))

    def test_retriever_returns_empty_list_when_catalog_is_missing(self) -> None:
        with patch("hybrid_candidate_retriever.CATALOG_RECORDS_PATH", Path("/missing/catalog.jsonl")):
            self.assertEqual(retrieve_candidate_datasets("ВРП Архангельской области"), [])

    def test_generated_script_writes_empty_template_and_metadata(self) -> None:
        intent = sample_intent()
        design = sample_design(intent)
        structure = build_target_dataset_structure(intent, design)
        with patch("assembly_planner.CATALOG_RECORDS_PATH", Path("/missing/catalog.jsonl")):
            plan = plan_dataset_build(intent, design, structure)
        script = generate_build_script(structure, plan)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            script_path = tmpdir_path / script.filename
            output_dir = tmpdir_path / "out"
            script_path.write_text(script.content, encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--output-dir",
                    str(output_dir),
                    "--project-root",
                    str(Path.cwd()),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            self.assertEqual(payload["row_count"], 0)
            self.assertTrue((output_dir / "target_dataset.csv").exists())
            metadata = json.loads((output_dir / "target_dataset.metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["target_structure"]["row_grain"], "регион-год")

    def test_artifact_writer_writes_full_manifest(self) -> None:
        intent = sample_intent()
        design = sample_design(intent)
        structure = build_target_dataset_structure(intent, design)
        with patch("assembly_planner.CATALOG_RECORDS_PATH", Path("/missing/catalog.jsonl")):
            plan = plan_dataset_build(intent, design, structure)
        script = generate_build_script(structure, plan)
        result = OrchestrationResult(
            status=OrchestrationStatus.DESIGN_READY,
            intent=intent,
            research_design=design,
            dataset_structure=structure,
            build_plan=plan,
            build_script=script,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = write_orchestration_artifacts(result, tmpdir)
            artifact_types = {artifact.artifact_type for artifact in artifacts}

            self.assertIn("research_report", artifact_types)
            self.assertIn("research_intent", artifact_types)
            self.assertIn("build_script", artifact_types)
            report = (Path(tmpdir) / "00_research_report.md").read_text(encoding="utf-8")
            self.assertIn("Шаг 1. Определение исследования", report)
            self.assertIn("Шаг 5. Собранный артефакт", report)
            self.assertTrue((Path(tmpdir) / "manifest.json").exists())

    def test_generated_script_normalizes_wide_fedstat_like_rows(self) -> None:
        structure = build_target_dataset_structure(sample_intent(), sample_design(sample_intent()))
        plan = plan_dataset_build(sample_intent(), sample_design(sample_intent()), structure, use_registry=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            source_path = tmpdir_path / "wide.json"
            source_path.write_text(
                json.dumps(
                    [
                        {
                            "column00": "Классификатор объектов административно-территориального деления (ОКАТО)",
                            "column01": "Классификатор видов экономической деятельности (ОКВЭД2)",
                            "column02": "Вид цены",
                            "column03": 2020.0,
                            "column04": 2021.0,
                        },
                        {
                            "column00": "11000000000 Архангельская область",
                            "column01": "101.АГ Всего по обследуемым видам экономической деятельности",
                            "column02": "1 Текущие цены",
                            "column03": 10.0,
                            "column04": 12.0,
                        },
                        {
                            "column00": "11100000000 Ненецкий автономный округ (Архангельская область)",
                            "column01": "101.АГ Всего по обследуемым видам экономической деятельности",
                            "column02": "1 Текущие цены",
                            "column03": 20.0,
                            "column04": 22.0,
                        },
                        {
                            "column00": "11000000000 Архангельская область",
                            "column01": "101.АГ Всего по обследуемым видам экономической деятельности",
                            "column02": "9 Постоянные цены 2016 года",
                            "column03": 30.0,
                            "column04": 32.0,
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            plan.predecessor_datasets[0].data_path = str(source_path)
            script = generate_build_script(structure, plan)
            script_path = tmpdir_path / script.filename
            output_dir = tmpdir_path / "out"
            script_path.write_text(script.content, encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--output-dir",
                    str(output_dir),
                    "--project-root",
                    str(Path.cwd()),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            rows = (output_dir / "target_dataset.csv").read_text(encoding="utf-8").splitlines()
            metadata = json.loads((output_dir / "target_dataset.metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["row_count"], 2)
            self.assertEqual(payload["coverage"]["available_years"], [2020, 2021])
            self.assertIn(2015, metadata["coverage"]["missing_years"])
            self.assertIn(2024, metadata["coverage"]["missing_years"])
            self.assertIn("11000000000 Архангельская область,2020,10.0", rows[1])

    def test_generated_script_does_not_fill_dimension_like_columns_with_indicator_value(self) -> None:
        structure = TargetDatasetStructure(
            row_grain="страна-год",
            primary_key=["geo", "year"],
            columns=[
                DatasetColumn(
                    name="geo",
                    title="География",
                    role="dimension",
                    dtype="string",
                    nullable=False,
                ),
                DatasetColumn(
                    name="year",
                    title="Год",
                    role="dimension",
                    dtype="integer",
                    nullable=False,
                ),
                DatasetColumn(
                    name="inflation_rate",
                    title="inflation_rate",
                    role="indicator",
                    dtype="number",
                    source_field="inflation_rate",
                ),
                DatasetColumn(
                    name="country_code",
                    title="country_code",
                    role="indicator",
                    dtype="string",
                    source_field="country_code",
                ),
                DatasetColumn(
                    name="year_2",
                    title="year",
                    role="indicator",
                    dtype="integer",
                    source_field="year",
                ),
            ],
            expected_frequency="годовая",
            time_range="2020",
            geography=["Россия"],
        )
        plan = plan_dataset_build(sample_intent(), sample_design(sample_intent()), structure, use_registry=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            source_path = tmpdir_path / "wb_inflation.json"
            source_path.write_text(
                json.dumps(
                    [
                        {
                            "country_name": "Russian Federation",
                            "countryiso3code": "RUS",
                            "date": 2020,
                            "value": 3.4,
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            plan.predecessor_datasets[0].data_path = str(source_path)
            script = generate_build_script(structure, plan)
            script_path = tmpdir_path / script.filename
            output_dir = tmpdir_path / "out"
            script_path.write_text(script.content, encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--output-dir",
                    str(output_dir),
                    "--project-root",
                    str(Path.cwd()),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            with (output_dir / "target_dataset.csv").open(encoding="utf-8") as file:
                row = next(csv.DictReader(file))

            self.assertEqual(row["inflation_rate"], "3.4")
            self.assertEqual(row["country_code"], "RUS")
            self.assertEqual(row["year_2"], "2020")

    def test_structure_filters_dimension_like_indicator_names(self) -> None:
        intent = sample_intent().model_copy(
            update={
                "indicators": ["Инфляция (ИПЦ)", "inflation_rate", "country_code", "year"],
                "indicator_specs": [
                    IndicatorSpec(
                        name="Инфляция (ИПЦ)",
                        definition="Годовая инфляция по ИПЦ.",
                        unit="%",
                        role="primary",
                    ),
                    IndicatorSpec(
                        name="inflation_rate",
                        definition="Annual CPI inflation.",
                        unit="%",
                        role="primary",
                    ),
                ],
                "granularity": "страна-год",
                "geography": ["Россия", "США"],
            }
        )
        design = ResearchStudyDesign.model_validate(
            {
                "original_query": intent.original_query,
                "design_summary": "Сравнить инфляцию.",
                "hypotheses": [],
                "required_measurements": [
                    {
                        "name": "inflation_rate",
                        "definition": "Annual CPI inflation.",
                        "unit": "%",
                        "role": "primary",
                        "source_candidates": ["World Bank"],
                        "is_critical": True,
                    },
                    {
                        "name": "country_code",
                        "definition": "ISO country code.",
                        "unit": "string",
                        "role": "primary",
                        "source_candidates": ["World Bank"],
                        "is_critical": False,
                    },
                    {
                        "name": "year",
                        "definition": "Observation year.",
                        "unit": "integer",
                        "role": "primary",
                        "source_candidates": ["World Bank"],
                        "is_critical": False,
                    },
                ],
                "grouping_rules": [],
                "derived_metrics": [],
                "statistical_methods": [],
                "visualizations": [],
                "data_quality_checks": [],
                "methodology_notes": "",
                "required_row_grain": ["страна-год"],
                "can_continue": True,
                "blocking_reasons": [],
            }
        )

        structure = build_target_dataset_structure(intent, design)
        indicator_columns = [column.name for column in structure.columns if column.role == "indicator"]

        self.assertEqual(indicator_columns, ["inflyatsiya_ipts"])
        self.assertNotIn("country_code", indicator_columns)
        self.assertNotIn("year_2", indicator_columns)

    def test_bm25_query_expands_domain_abbreviations_and_drops_years(self) -> None:
        tokens = tokenize_query("ВРП Архангельской области 2015 2024")

        self.assertIn("валовой", tokens)
        self.assertIn("региональный", tokens)
        self.assertNotIn("2015", tokens)

    def test_query_expansion_adds_world_bank_indicator_aliases(self) -> None:
        gdp_tokens = tokenize_query("ВВП США Россия 2020 2024")
        inflation_tokens = tokenize_query("ИПЦ США Россия")

        self.assertIn("gdp", gdp_tokens)
        self.assertIn("gross", gdp_tokens)
        self.assertIn("consumer", inflation_tokens)
        self.assertIn("cpi", inflation_tokens)

    def test_hybrid_ranking_keeps_canonical_wb_gdp_first(self) -> None:
        records = {
            "wb:NY.GDP.MKTP.CD": {
                "record_id": "wb:NY.GDP.MKTP.CD",
                "title": "GDP (current US$)",
                "source": "world_bank",
            },
            "wb:DP.DOD.DECD.CR.FC.Z1": {
                "record_id": "wb:DP.DOD.DECD.CR.FC.Z1",
                "title": (
                    "Gross PSD, Financial Public Corp., Domestic creditors, "
                    "Nominal Value, % of GDP"
                ),
                "source": "world_bank",
            },
            "fedstat:30946": {
                "record_id": "fedstat:30946",
                "title": "Валовой внутренний продукт в рыночных ценах",
                "source": "fedstat",
            },
        }

        with patch("hybrid_candidate_retriever._catalog_records", return_value=records):
            candidates = hybrid_candidate_retriever._merge_candidates(
                query="ВВП США Россия",
                alias_results=[{**records["wb:NY.GDP.MKTP.CD"], "alias_rank": 1}],
                vector_results=[
                    {
                        **records["wb:DP.DOD.DECD.CR.FC.Z1"],
                        "vector_rank": 1,
                        "vector_score": 1.0,
                    }
                ],
                bm25_results=[
                    {
                        **records["fedstat:30946"],
                        "bm25_rank": 1,
                        "bm25_score": -10.0,
                    }
                ],
            )

        self.assertEqual(candidates[0]["record_id"], "wb:NY.GDP.MKTP.CD")
        self.assertIn("alias", candidates[0]["matched_by"])

    def test_planner_heuristic_prefers_wb_alias_for_cross_country_gdp(self) -> None:
        intent = sample_intent().model_copy(
            update={
                "original_query": "ВВП США Россия 2020 2024",
                "topic": "динамика ВВП России и США",
                "objects": ["Россия", "США"],
                "geography": ["Россия", "США"],
                "time_range": TimeRange(
                    raw="2020-2024",
                    start_year=2020,
                    end_year=2024,
                    is_explicit=True,
                ),
                "indicators": ["ВВП"],
                "indicator_specs": [
                    IndicatorSpec(
                        name="ВВП",
                        definition="валовой внутренний продукт",
                        unit="US$",
                        role="primary",
                    )
                ],
            }
        )
        design = sample_design(intent)
        candidates = [
            {
                "record_id": "fedstat:30946",
                "title": "Валовой внутренний продукт в рыночных ценах",
                "source": "fedstat",
                "data_path": "dumps/fedstatru/fedstatru/data/parquet/30946.parquet",
            },
            {
                "record_id": "wb:NY.GDP.MKTP.CD",
                "title": "GDP (current US$)",
                "source": "world_bank",
                "data_path": "dumps/wb/wb/parquet/NY.GDP.MKTP.CD.parquet",
                "alias_group": "gdp",
            },
        ]

        selected = assembly_planner._heuristic_select_candidates(
            intent,
            design,
            candidates,
            limit=1,
        )

        self.assertEqual(selected[0].record_id, "wb:NY.GDP.MKTP.CD")


if __name__ == "__main__":
    unittest.main()
