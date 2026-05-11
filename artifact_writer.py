import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from orchestrator import OrchestrationResult
from research_report import generate_research_report


class WrittenArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: str
    path: str


def write_orchestration_artifacts(
    result: OrchestrationResult,
    output_dir: str | Path,
    build_run: dict[str, object] | None = None,
    dataset_output_dir: str | Path | None = None,
) -> list[WrittenArtifact]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    artifacts: list[WrittenArtifact] = []
    artifacts.append(
        write_research_report_artifact(
            result,
            directory,
            build_run=build_run,
            dataset_output_dir=dataset_output_dir,
        )
    )
    artifacts.append(
        _write_json(
            directory / "01_research_intent.json",
            "research_intent",
            result.intent.model_dump(mode="json"),
        )
    )
    if result.research_design:
        artifacts.append(
            _write_json(
                directory / "02_research_design.json",
                "research_design",
                result.research_design.model_dump(mode="json"),
            )
        )
    if result.dataset_structure:
        artifacts.append(
            _write_json(
                directory / "03_dataset_structure.json",
                "dataset_structure",
                result.dataset_structure.model_dump(mode="json"),
            )
        )
    if result.build_plan:
        artifacts.append(
            _write_json(
                directory / "04_build_plan.json",
                "build_plan",
                result.build_plan.model_dump(mode="json"),
            )
        )
    if result.build_script:
        script_path = directory / result.build_script.filename
        script_path.write_text(result.build_script.content, encoding="utf-8")
        artifacts.append(
            WrittenArtifact(
                artifact_type="build_script",
                path=str(script_path.resolve()),
            )
        )

    manifest_artifact = WrittenArtifact(
        artifact_type="manifest",
        path=str((directory / "manifest.json").resolve()),
    )
    _write_json(
        directory / "manifest.json",
        "manifest",
        [
            artifact.model_dump(mode="json")
            for artifact in [*artifacts, manifest_artifact]
        ],
    )
    artifacts.append(manifest_artifact)
    return artifacts


def write_research_report_artifact(
    result: OrchestrationResult,
    output_dir: str | Path,
    build_run: dict[str, object] | None = None,
    dataset_output_dir: str | Path | None = None,
) -> WrittenArtifact:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "00_research_report.md"
    path.write_text(
        generate_research_report(
            result,
            build_run=build_run,
            dataset_output_dir=dataset_output_dir,
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
