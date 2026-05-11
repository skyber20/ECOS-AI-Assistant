import shutil
import sys
from pathlib import Path
from typing import TextIO

from openai import OpenAIError

from artifact_writer import WrittenArtifact, write_orchestration_artifacts
from dataset_reranker import DatasetRerankerError
from intent_parser import IntentParserError, parse_research_intent
from orchestrator import (
    ClarificationAnswer,
    ClarificationRequest,
    OrchestrationResult,
    OrchestrationStatus,
    continue_research_flow,
    prepare_intent_for_design,
    refine_intent_with_clarifications,
)
from research_designer import ResearchDesignerError
from script_generator import ScriptGeneratorError


USER_PROMPT = "Исследуй секрет рецепта маленькой инфляции в США"
LLM_PROVIDER: str | None = None
LLM_MODEL: str | None = None
USE_DEFAULTS = False
RUN_BUILD_SCRIPT = True
MAX_BUILD_TRIES = 3
MAX_CLARIFICATION_ROUNDS = 3

ARTIFACT_DIR = Path("artifacts/latest_run")
DATASET_OUTPUT_DIR = ARTIFACT_DIR / "generated_dataset"


def main() -> None:
    prompt = USER_PROMPT.strip()
    if not prompt:
        raise SystemExit("Ошибка: USER_PROMPT не должен быть пустым.")

    _reset_artifact_dir()

    try:
        result = _run_flow_with_clarifications(prompt)
        artifacts = write_orchestration_artifacts(
            result,
            output_dir=ARTIFACT_DIR,
            dataset_output_dir=DATASET_OUTPUT_DIR,
        )
    except (
        RuntimeError,
        ValueError,
        OpenAIError,
        IntentParserError,
        DatasetRerankerError,
        ResearchDesignerError,
        ScriptGeneratorError,
    ) as exc:
        raise SystemExit(f"Ошибка: {exc}") from exc

    _print_summary(result, artifacts)


def _run_flow_with_clarifications(query: str) -> OrchestrationResult:
    return run_interactive_research_flow(query)


def run_interactive_research_flow(
    query: str,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    error_stream: TextIO | None = None,
    provider: str | None = LLM_PROVIDER,
    model: str | None = LLM_MODEL,
    use_defaults: bool = USE_DEFAULTS,
    run_build_script: bool = RUN_BUILD_SCRIPT,
    build_output_dir: str | Path = DATASET_OUTPUT_DIR,
    max_build_tries: int = MAX_BUILD_TRIES,
    max_clarification_rounds: int = MAX_CLARIFICATION_ROUNDS,
) -> OrchestrationResult:
    prompt = query.strip()
    if not prompt:
        raise ValueError("query must not be empty")

    output = output_stream or error_stream or sys.stdout
    user_input = input_stream or sys.stdin

    print("Этап 1: формализация запроса", file=output)
    intent = parse_research_intent(
        prompt,
        provider=provider,
        model=model,
    )

    for round_number in range(1, max_clarification_rounds + 1):
        readiness = prepare_intent_for_design(intent, use_defaults=use_defaults)
        if readiness.status == OrchestrationStatus.READY_FOR_DESIGN:
            print("Этап 2: RAG, дизайн исследования, генерация и запуск сборки", file=output)
            return continue_research_flow(
                intent,
                provider=provider,
                model=model,
                use_defaults=use_defaults,
                run_build_script=run_build_script,
                build_output_dir=str(build_output_dir),
                max_build_tries=max_build_tries,
            )

        if readiness.status != OrchestrationStatus.NEEDS_CLARIFICATION:
            return readiness

        print(
            f"Нужно уточнение запроса, раунд {round_number}/{max_clarification_rounds}",
            file=output,
        )
        answers = _collect_clarification_answers(
            readiness.clarification_requests,
            input_stream=user_input,
            output_stream=output,
        )
        intent = refine_intent_with_clarifications(
            intent,
            answers,
            provider=provider,
            model=model,
        )

    readiness = prepare_intent_for_design(intent, use_defaults=use_defaults)
    if readiness.status == OrchestrationStatus.READY_FOR_DESIGN:
        return continue_research_flow(
            intent,
            provider=provider,
            model=model,
            use_defaults=use_defaults,
            run_build_script=run_build_script,
            build_output_dir=str(build_output_dir),
            max_build_tries=max_build_tries,
        )

    fields = ", ".join(item.field for item in readiness.clarification_requests)
    raise RuntimeError(f"Не удалось уточнить запрос. Остались поля: {fields or readiness.status.value}.")


def _collect_clarification_answers(
    requests: list[ClarificationRequest],
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> list[ClarificationAnswer]:
    user_input = input_stream or sys.stdin
    output = output_stream or sys.stdout
    answers: list[ClarificationAnswer] = []
    for index, request in enumerate(requests, start=1):
        print("", file=output)
        print(f"Уточнение {index}/{len(requests)}: {request.question}", file=output)
        print(f"Причина: {request.reason}", file=output)
        if request.default_assumption:
            print(f"Можно нажать Enter для default: {request.default_assumption}", file=output)

        print("Ответ: ", end="", file=output)
        output.flush()
        answer = user_input.readline().strip()
        used_default = False
        if not answer and request.default_assumption:
            answer = request.default_assumption
            used_default = True
        if not answer:
            raise RuntimeError("Уточнение обязательно, чтобы продолжить.")

        answers.append(
            ClarificationAnswer(
                field=request.field,
                question=request.question,
                answer=answer,
                used_default=used_default,
            )
        )

    return answers


def _reset_artifact_dir() -> None:
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)
    DATASET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _print_summary(
    result: OrchestrationResult,
    artifacts: list[WrittenArtifact],
) -> None:
    print("Готово.")
    print(f"Статус: {result.status.value}")
    print(f"Артефакты: {ARTIFACT_DIR.resolve()}")

    report_path = _artifact_path(artifacts, "research_report")
    if report_path:
        print(f"Отчет: {report_path}")

    script_path = _artifact_path(artifacts, "build_script_file")
    if script_path:
        print(f"Код сборки: {script_path}")

    if result.build_run:
        print(f"Датасет и результат запуска: {Path(result.build_run.output_dir).resolve()}")
        print(f"Статус сборки: {result.build_run.status}")

    if result.clarification_requests:
        print("Нужны уточнения:")
        for item in result.clarification_requests:
            print(f"- {item.question}")
            if item.default_assumption:
                print(f"  default: {item.default_assumption}")

    limitations = _limitations(result)
    if limitations:
        print("Ограничения:")
        for item in limitations:
            print(f"- {item}")


def _artifact_path(artifacts: list[WrittenArtifact], artifact_type: str) -> str | None:
    for artifact in artifacts:
        if artifact.artifact_type == artifact_type:
            return artifact.path
    return None


def _limitations(result: OrchestrationResult) -> list[str]:
    values: list[str] = []
    if result.dataset_rerank:
        for item in result.dataset_rerank.results:
            values.extend(item.possible_limitations)
    if result.build_script:
        values.extend(result.build_script.possible_limitations)
    if result.build_run and isinstance(result.build_run.output, dict):
        output_limitations = (
            result.build_run.output.get("limitations")
            or result.build_run.output.get("possible_limitations")
            or []
        )
        if isinstance(output_limitations, list):
            values.extend(str(item) for item in output_limitations)
        elif output_limitations:
            values.append(str(output_limitations))
    return list(dict.fromkeys(item for item in values if item.strip()))


if __name__ == "__main__":
    main()
