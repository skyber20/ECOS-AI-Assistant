from __future__ import annotations

import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from artifact_writer import WrittenArtifact, write_orchestration_artifacts
from dataset_reranker import DatasetRerankerError
from intent_parser import IntentParserError, ResearchIntent
from orchestrator import (
    ClarificationAnswer,
    continue_research_flow,
    refine_intent_with_clarifications,
    run_research_flow,
)
from research_designer import ResearchDesignerError
from script_generator import ScriptGeneratorError


ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", "artifacts/runs")).resolve()


class ResearchRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    provider: str | None = None
    model: str | None = None
    use_defaults: bool = False
    run_build_script: bool = True
    max_build_tries: int = Field(default=3, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must not be blank")
        return query

    @field_validator("provider", "model", mode="before")
    @classmethod
    def _empty_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class ContinueResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: ResearchIntent
    answers: list[ClarificationAnswer] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    use_defaults: bool = False
    run_build_script: bool = True
    max_build_tries: int = Field(default=3, ge=1, le=10)

    @field_validator("provider", "model", mode="before")
    @classmethod
    def _empty_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class ArtifactInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: str
    filename: str
    path: str
    download_url: str


class ResearchRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: str
    status: str
    message: str | None = None
    artifact_dir: str
    artifacts: list[ArtifactInfo]
    report_markdown: str | None = None
    result: dict[str, Any]


def create_app() -> FastAPI:
    app = FastAPI(
        title="ECOS AI Assistant API",
        version="0.1.0",
        description="FastAPI backend for the ECOS AI research assistant.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "name": "ECOS AI Assistant API",
            "health": "/health",
            "docs": "/docs",
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "artifact_root": str(ARTIFACT_ROOT),
            "catalog_records_exists": Path("data/catalog_records.jsonl").exists(),
            "catalog_bm25_exists": Path("data/catalog_bm25.sqlite").exists(),
            "dumps_exists": Path("dumps").exists(),
        }

    @app.post("/api/research", response_model=ResearchRunResponse)
    def start_research(payload: ResearchRunRequest) -> ResearchRunResponse:
        run_id, run_dir = _create_run_dir()
        dataset_dir = run_dir / "generated_dataset"
        try:
            result = run_research_flow(
                payload.query,
                provider=payload.provider,
                model=payload.model,
                use_defaults=payload.use_defaults,
                run_build_script=payload.run_build_script,
                build_output_dir=str(dataset_dir),
                max_build_tries=payload.max_build_tries,
            )
            artifacts = write_orchestration_artifacts(
                result,
                output_dir=run_dir,
                dataset_output_dir=dataset_dir,
            )
        except _known_runtime_errors() as exc:
            raise _http_error(exc) from exc

        return _response_from_result(run_id, run_dir, artifacts, result.model_dump(mode="json"))

    @app.post("/api/research/continue", response_model=ResearchRunResponse)
    def continue_research(payload: ContinueResearchRequest) -> ResearchRunResponse:
        run_id, run_dir = _create_run_dir()
        dataset_dir = run_dir / "generated_dataset"
        try:
            refined_intent = refine_intent_with_clarifications(
                payload.intent,
                payload.answers,
                provider=payload.provider,
                model=payload.model,
            )
            result = continue_research_flow(
                refined_intent,
                provider=payload.provider,
                model=payload.model,
                use_defaults=payload.use_defaults,
                run_build_script=payload.run_build_script,
                build_output_dir=str(dataset_dir),
                max_build_tries=payload.max_build_tries,
            )
            artifacts = write_orchestration_artifacts(
                result,
                output_dir=run_dir,
                dataset_output_dir=dataset_dir,
            )
        except _known_runtime_errors() as exc:
            raise _http_error(exc) from exc

        return _response_from_result(run_id, run_dir, artifacts, result.model_dump(mode="json"))

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run_dir = _run_dir(run_id)
        manifest = _read_manifest(run_dir)
        return {
            "run_id": run_id,
            "artifact_dir": str(run_dir),
            "artifacts": [
                _artifact_info(run_id, WrittenArtifact.model_validate(item)).model_dump(mode="json")
                for item in manifest
            ],
            "report_markdown": _read_report_from_manifest(manifest),
        }

    @app.get("/api/runs/{run_id}/artifacts/{artifact_type}")
    def download_artifact(run_id: str, artifact_type: str) -> FileResponse:
        run_dir = _run_dir(run_id)
        manifest = _read_manifest(run_dir)
        artifact = next(
            (
                WrittenArtifact.model_validate(item)
                for item in manifest
                if item.get("artifact_type") == artifact_type
            ),
            None,
        )
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found.")

        path = Path(artifact.path).resolve()
        _ensure_inside_run(path, run_dir)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact file not found.")

        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=path.name)

    return app


def _create_run_dir() -> tuple[str, Path]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{uuid4().hex[:8]}"
    run_dir = (ARTIFACT_ROOT / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def _run_dir(run_id: str) -> Path:
    if not run_id or run_id != Path(run_id).name:
        raise HTTPException(status_code=404, detail="Run not found.")
    run_dir = (ARTIFACT_ROOT / run_id).resolve()
    _ensure_inside_run(run_dir, ARTIFACT_ROOT)
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found.")
    return run_dir


def _read_manifest(run_dir: Path) -> list[dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Run manifest not found.")
    import json

    with manifest_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail="Run manifest is invalid.")
    return payload


def _response_from_result(
    run_id: str,
    run_dir: Path,
    artifacts: list[WrittenArtifact],
    result: dict[str, Any],
) -> ResearchRunResponse:
    status = str(result.get("status") or "")
    return ResearchRunResponse(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        message=result.get("message"),
        artifact_dir=str(run_dir),
        artifacts=[_artifact_info(run_id, artifact) for artifact in artifacts],
        report_markdown=_read_report(artifacts),
        result=result,
    )


def _artifact_info(run_id: str, artifact: WrittenArtifact) -> ArtifactInfo:
    path = Path(artifact.path)
    return ArtifactInfo(
        artifact_type=artifact.artifact_type,
        filename=path.name,
        path=artifact.path,
        download_url=f"/api/runs/{run_id}/artifacts/{artifact.artifact_type}",
    )


def _read_report(artifacts: list[WrittenArtifact]) -> str | None:
    for artifact in artifacts:
        if artifact.artifact_type != "research_report":
            continue
        path = Path(artifact.path)
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def _read_report_from_manifest(manifest: list[dict[str, Any]]) -> str | None:
    for item in manifest:
        if item.get("artifact_type") != "research_report":
            continue
        path = Path(str(item.get("path", ""))).resolve()
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def _ensure_inside_run(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc


def _http_error(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=500,
        detail={
            "type": exc.__class__.__name__,
            "message": str(exc),
        },
    )


def _known_runtime_errors() -> tuple[type[Exception], ...]:
    return (
        RuntimeError,
        ValueError,
        OpenAIError,
        IntentParserError,
        DatasetRerankerError,
        ResearchDesignerError,
        ScriptGeneratorError,
    )


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "*")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or ["*"]


app = create_app()
