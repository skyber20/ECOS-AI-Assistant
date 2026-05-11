import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from artifact_writer import write_orchestration_artifacts
from assembly_planner import plan_dataset_build
from catalog_bm25_indexer import tokenize_query
from dataset_structure import build_target_dataset_structure
from hybrid_candidate_retriever import retrieve_candidate_datasets
from intent_parser import (
    Complexity,
    DatasetSpec,
    IndicatorSpec,
    IntentType,
    NextAction,
    ResearchIntent,
    SourceCandidate,
    TimeRange,
)
from research_designer import ResearchStudyDesign
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

    def test_bm25_query_expands_domain_abbreviations_and_drops_years(self) -> None:
        tokens = tokenize_query("ВРП Архангельской области 2015 2024")

        self.assertIn("валовой", tokens)
        self.assertIn("региональный", tokens)
        self.assertNotIn("2015", tokens)


if __name__ == "__main__":
    unittest.main()
