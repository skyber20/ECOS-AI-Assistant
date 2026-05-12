import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from orchestrator import OrchestrationResult
from research_report import generate_research_report


DEFAULT_ARTIFACT_DIR = Path("artifacts/latest_run")
DEFAULT_DATASET_OUTPUT_DIR = DEFAULT_ARTIFACT_DIR / "generated_dataset"


class WrittenArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: str
    path: str


def write_orchestration_artifacts(
    result: OrchestrationResult,
    output_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    dataset_output_dir: str | Path | None = None,
) -> list[WrittenArtifact]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(dataset_output_dir or DEFAULT_DATASET_OUTPUT_DIR)

    artifacts: list[WrittenArtifact] = [
        write_research_report_artifact(result, directory, dataset_output_dir=dataset_dir),
        _write_json(directory / "01_research_intent.json", "research_intent", result.intent.model_dump(mode="json")),
    ]

    if result.dataset_rerank:
        artifacts.append(
            _write_json(
                directory / "02_dataset_rerank.json",
                "dataset_rerank",
                result.dataset_rerank.model_dump(mode="json"),
            )
        )
    if result.research_design:
        artifacts.append(
            _write_json(
                directory / "03_research_design.json",
                "research_design",
                result.research_design.model_dump(mode="json"),
            )
        )
    if result.dataset_structure:
        artifacts.append(
            _write_json(
                directory / "04_dataset_structure.json",
                "dataset_structure",
                result.dataset_structure.model_dump(mode="json"),
            )
        )
    if result.build_script:
        artifacts.append(
            _write_json(
                directory / "05_build_script.json",
                "build_script",
                result.build_script.model_dump(mode="json"),
            )
        )
        script_path = directory / result.build_script.filename
        script_path.write_text(result.build_script.content, encoding="utf-8")
        artifacts.append(
            WrittenArtifact(
                artifact_type="build_script_file",
                path=str(script_path.resolve()),
            )
        )
    if result.build_run:
        artifacts.append(
            _write_json(
                directory / "06_build_run.json",
                "build_run",
                result.build_run.model_dump(mode="json"),
            )
        )
        artifacts.extend(_build_output_artifacts(result.build_run.output))

    manifest = WrittenArtifact(
        artifact_type="manifest",
        path=str((directory / "manifest.json").resolve()),
    )
    _write_json(
        directory / "manifest.json",
        "manifest",
        [artifact.model_dump(mode="json") for artifact in [*artifacts, manifest]],
    )
    artifacts.append(manifest)
    return artifacts


def write_research_report_artifact(
    result: OrchestrationResult,
    output_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    dataset_output_dir: str | Path | None = None,
) -> WrittenArtifact:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "00_research_report.md"
    path.write_text(
        generate_research_report(
            result,
            dataset_output_dir=dataset_output_dir or DEFAULT_DATASET_OUTPUT_DIR,
        ),
        encoding="utf-8",
    )
    return WrittenArtifact(
        artifact_type="research_report",
        path=str(path.resolve()),
    )


def _write_json(path: Path, artifact_type: str, payload: Any) -> WrittenArtifact:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return WrittenArtifact(
        artifact_type=artifact_type,
        path=str(path.resolve()),
    )


def _build_output_artifacts(output: dict[str, Any] | None) -> list[WrittenArtifact]:
    if not isinstance(output, dict):
        return []

    keys = [
        "dataset",
        "target_dataset",
        "csv_path",
        "metadata",
        "meta_path",
        "source_manifest",
        "manifest",
        "manifest_path",
        "chart_data",
        "chart",
        "chart_path",
        "chart_data_path",
        "sql",
        "sql_path",
        "query",
    ]
    artifacts: list[WrittenArtifact] = []
    seen: set[str] = set()

    for key in keys:
        value = output.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = str(Path(value).resolve())
        if path in seen:
            continue
        seen.add(path)
        artifacts.append(WrittenArtifact(artifact_type=f"build_output_{key}", path=path))

    return artifacts
