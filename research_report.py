import csv
from pathlib import Path
from typing import Any

from dataset_reranker import DatasetRerankResponse
from dataset_structure import DatasetColumn, TargetDatasetStructure
from intent_parser import Complexity, IntentType, ResearchIntent
from orchestrator import ClarificationRequest, OrchestrationResult, OrchestrationStatus
from research_designer import ResearchStudyDesign
from script_generator import BuildScriptRun, GeneratedBuildScript


DEFAULT_DATASET_OUTPUT_DIR = Path("artifacts/latest_run/generated_dataset")


COMPLEXITY_LABELS = {
    Complexity.EASY: "легкая",
    Complexity.MEDIUM: "средняя",
    Complexity.COMPLEX: "сложная",
}

INTENT_LABELS = {
    IntentType.SIMPLE_DATA: "простые данные",
    IntentType.COMPARATIVE: "сравнение",
    IntentType.RESEARCH: "исследование",
    IntentType.DERIVED: "производная метрика",
    IntentType.AMBIGUOUS: "неоднозначный запрос",
    IntentType.NO_DATA: "нет данных",
    IntentType.UNSUPPORTED: "неподдерживаемый запрос",
}


def generate_research_report(
    result: OrchestrationResult,
    dataset_output_dir: str | Path | None = None,
) -> str:
    lines: list[str] = []
    intent = result.intent

    lines.append("# Исследовательский отчет")
    lines.append("")
    lines.append(f"Запрос: {intent.original_query}")
    lines.append(f"Статус: {result.status.value}")
    lines.append(f"Тип: {INTENT_LABELS.get(intent.intent_type, intent.intent_type.value)}")
    lines.append(f"Сложность: {COMPLEXITY_LABELS.get(intent.complexity, intent.complexity.value)}")
    if result.message:
        lines.append(f"Сообщение: {result.message}")
    lines.append("")

    if result.status == OrchestrationStatus.NO_DATA or intent.intent_type == IntentType.NO_DATA:
        _append_definition(lines, intent, result.clarification_requests)
        _append_no_data(lines, intent, result.dataset_rerank)
        return "\n".join(lines).rstrip() + "\n"

    if result.status == OrchestrationStatus.NEEDS_CLARIFICATION:
        _append_definition(lines, intent, result.clarification_requests)
        lines.append("## Следующий шаг")
        lines.append("")
        lines.append("Нужно получить уточнения пользователя или явно применить default-допущения.")
        return "\n".join(lines).rstrip() + "\n"

    _append_definition(lines, intent, result.clarification_requests)
    _append_sources(lines, result.dataset_rerank)
    _append_design(lines, result.research_design, intent)
    _append_structure(lines, result.dataset_structure)
    _append_script(lines, result.build_script)
    _append_run(lines, result.build_run, dataset_output_dir)

    return "\n".join(lines).rstrip() + "\n"


def _append_definition(
    lines: list[str],
    intent: ResearchIntent,
    clarification_requests: list[ClarificationRequest],
) -> None:
    lines.append("## Шаг 1. Формальное описание")
    lines.append("")
    lines.append(f"Тема: {_value(intent.topic)}")
    lines.append(f"География: {_join(intent.geography or intent.objects or intent.entities)}")
    lines.append(f"Период: {_period(intent)}")
    lines.append(f"Частота: {_value(intent.frequency)}")
    lines.append(f"Показатели: {_join(intent.indicators)}")
    lines.append(f"Ракурс: {_value(intent.disciplinary_perspective)}")
    lines.append("")
    lines.append("Исследовательские вопросы:")
    _append_items(lines, intent.research_questions, "отдельно не заданы")
    lines.append("")
    lines.append("Уточнения и default-допущения:")
    if clarification_requests:
        for request in clarification_requests:
            lines.append(f"- {request.question}")
            if request.default_assumption:
                lines.append(f"  default: {request.default_assumption}")
            lines.append(f"  причина: {request.reason}")
    elif intent.assumptions_if_no_answer:
        _append_items(lines, intent.assumptions_if_no_answer, "не требуются")
    else:
        lines.append("- не требуются")
    lines.append("")


def _append_no_data(
    lines: list[str],
    intent: ResearchIntent,
    rerank: DatasetRerankResponse | None,
) -> None:
    lines.append("## Вердикт по данным")
    lines.append("")
    verdict = intent.data_availability.verdict or _no_results_reason(rerank)
    lines.append(f"Вердикт: {_value(verdict, 'данные не найдены в доступных metadata')}")
    lines.append("Причины:")
    reasons = [*intent.data_availability.reasons]
    if _no_results_reason(rerank):
        reasons.append(_no_results_reason(rerank))
    _append_items(lines, reasons, "причины не детализированы")
    if intent.data_availability.alternatives:
        lines.append("")
        lines.append("Возможные альтернативы:")
        _append_items(lines, intent.data_availability.alternatives, "не предложены")


def _append_sources(lines: list[str], rerank: DatasetRerankResponse | None) -> None:
    lines.append("## Шаг 2. Источники")
    lines.append("")
    if not rerank:
        lines.append("RAG-результат отсутствует.")
        lines.append("")
        return

    if rerank.results:
        lines.append("Выбранные датасеты:")
        for item in rerank.results:
            lines.append(f"- {item.title or item.record_id}")
            lines.append(f"  record_id: {item.record_id}")
            if item.dataset_id:
                lines.append(f"  dataset_id: {item.dataset_id}")
            if item.source:
                lines.append(f"  источник: {item.source}")
            if item.data_path:
                lines.append(f"  data_path: {item.data_path}")
            if item.source_url:
                lines.append(f"  source_url: {item.source_url}")
            lines.append(f"  релевантность: {item.relevance.value}")
            lines.append(f"  уверенность: {item.usefulness_confidence.value}")
            lines.append(f"  почему выбран: {item.why_matched}")
            if item.possible_limitations:
                lines.append(f"  ограничения: {'; '.join(item.possible_limitations)}")
    else:
        lines.append(f"Выбранные датасеты: нет. {_value(rerank.no_results_reason)}")

    if rerank.rejected_similar_candidates:
        lines.append("")
        lines.append("Похожие, но отклоненные:")
        for item in rerank.rejected_similar_candidates[:8]:
            lines.append(f"- {item.title or item.record_id}: {item.reason}")
    lines.append("")


def _append_design(
    lines: list[str],
    design: ResearchStudyDesign | None,
    intent: ResearchIntent,
) -> None:
    lines.append("## Шаг 3. Дизайн исследования")
    lines.append("")
    if not design:
        lines.append("Дизайн не сформирован.")
        lines.append("")
        return

    lines.append(f"Кратко: {design.design_summary}")
    lines.append("")
    lines.append("Гипотезы:")
    if design.hypotheses:
        for item in design.hypotheses:
            prefix = f"{item.id}: " if item.id else ""
            lines.append(f"- {prefix}{item.statement}")
    elif intent.research_design and intent.research_design.hypotheses:
        _append_items(lines, intent.research_design.hypotheses, "не заданы")
    else:
        lines.append("- не требуются для простого запроса")

    lines.append("")
    lines.append("Нужные измерения:")
    if design.required_measurements:
        for item in design.required_measurements:
            parts = [item.name]
            if item.unit:
                parts.append(f"единица: {item.unit}")
            if item.role:
                parts.append(f"роль: {item.role}")
            if item.definition:
                parts.append(item.definition)
            lines.append("- " + "; ".join(parts))
    else:
        _append_items(lines, intent.indicators, "не заданы")

    lines.append("")
    lines.append("Производные метрики:")
    if design.derived_metrics:
        for item in design.derived_metrics:
            lines.append(f"- {item.name}: {item.formula}")
            if item.inputs:
                lines.append(f"  inputs: {_join(item.inputs)}")
            if item.normalization:
                lines.append(f"  нормализация: {item.normalization}")
    elif intent.derived_metrics:
        for item in intent.derived_metrics:
            lines.append(f"- {item.name}: {_value(item.formula)}")
    else:
        lines.append("- не требуются")

    lines.append("")
    lines.append("Визуализации:")
    if design.visualizations:
        for item in design.visualizations:
            axes = []
            if item.x_axis:
                axes.append(f"x={item.x_axis}")
            if item.y_axis:
                axes.append(f"y={item.y_axis}")
            suffix = f" ({', '.join(axes)})" if axes else ""
            lines.append(f"- {item.title}: {item.chart_type}{suffix}")
    else:
        lines.append("- не заданы")

    if design.methodology_notes or design.data_quality_checks or design.blocking_reasons:
        lines.append("")
        lines.append("Ограничения и проверки:")
        items = []
        if design.methodology_notes:
            items.append(design.methodology_notes)
        items.extend(check.action_if_failed for check in design.data_quality_checks)
        items.extend(design.blocking_reasons)
        _append_items(lines, items, "нет")
    lines.append("")


def _append_structure(lines: list[str], structure: TargetDatasetStructure | None) -> None:
    lines.append("## Шаг 4. Структура датасета")
    lines.append("")
    if not structure:
        lines.append("Структура не сформирована.")
        lines.append("")
        return

    lines.append(f"Зернистость строки: {structure.row_grain}")
    lines.append(f"Первичный ключ: {_join(structure.primary_key)}")
    lines.append(f"Частота: {_value(structure.expected_frequency)}")
    lines.append(f"Период: {_value(structure.time_range)}")
    lines.append("")
    _append_columns(lines, "Измерения", structure.columns, "dimension")
    _append_columns(lines, "Индикаторы", structure.columns, "indicator")
    _append_columns(lines, "Производные метрики", structure.columns, "derived_metric")
    _append_columns(lines, "Метаданные", structure.columns, "metadata")
    if structure.design_notes:
        lines.append("")
        lines.append("Примечания:")
        _append_items(lines, structure.design_notes, "нет")
    lines.append("")


def _append_script(lines: list[str], script: GeneratedBuildScript | None) -> None:
    lines.append("## Шаг 5. Сгенерированный код")
    lines.append("")
    if not script:
        lines.append("Скрипт не сформирован.")
        lines.append("")
        return

    lines.append(f"Файл: {script.filename}")
    lines.append(f"Язык: {script.language}")
    lines.append(f"Entrypoint: {script.entrypoint}")
    lines.append(f"Запуск: {script.usage}")
    if script.inputs:
        lines.append(f"Входы: {_join(script.inputs)}")
    if script.outputs:
        lines.append("Выходы:")
        for item in script.outputs:
            path = f", path={item.path}" if item.path else ""
            description = f", {item.description}" if item.description else ""
            lines.append(f"- {item.name}: {item.kind}{path}{description}")
    if script.possible_limitations:
        lines.append("Ограничения скрипта:")
        _append_items(lines, script.possible_limitations, "нет")
    lines.append("")


def _append_run(
    lines: list[str],
    build_run: BuildScriptRun | None,
    dataset_output_dir: str | Path | None,
) -> None:
    lines.append("## Шаг 6. Запуск и результат")
    lines.append("")
    output_dir = Path(dataset_output_dir or DEFAULT_DATASET_OUTPUT_DIR)
    if not build_run:
        lines.append(f"Скрипт еще не запускался. Ожидаемая папка результата: {output_dir}")
        lines.append("")
        return

    lines.append(f"Статус запуска: {build_run.status}")
    lines.append(f"Попыток: {len(build_run.attempts)}")
    lines.append(f"Repair применялся: {'да' if build_run.repaired else 'нет'}")
    if build_run.final_error:
        lines.append(f"Финальная ошибка: {build_run.final_error}")

    output = _normalized_output(build_run.output)
    if output:
        lines.append("")
        lines.append("Файлы результата:")
        for key in ["dataset", "metadata", "manifest", "chart_data"]:
            if output.get(key):
                lines.append(f"- {key}: {output[key]}")
        if output.get("row_count") is not None:
            lines.append(f"Строк: {output['row_count']}")
        if output.get("limitations"):
            lines.append("Ограничения результата:")
            _append_items(lines, output["limitations"], "нет")

        sample = _csv_sample(Path(str(output["dataset"]))) if output.get("dataset") else []
        if sample:
            lines.append("")
            lines.append("Первые строки CSV:")
            lines.append("```csv")
            lines.extend(sample)
            lines.append("```")
    lines.append("")


def _append_columns(
    lines: list[str],
    title: str,
    columns: list[DatasetColumn],
    role: str,
) -> None:
    selected = [item for item in columns if item.role == role]
    lines.append(f"{title}:")
    if not selected:
        lines.append("- нет")
        return
    for item in selected:
        unit = f", unit={item.unit}" if item.unit else ""
        nullable = "nullable" if item.nullable else "not null"
        lines.append(f"- {item.name}: {item.title} ({item.dtype}{unit}, {nullable})")


def _normalized_output(output: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {}
    return {
        "dataset": _first(output, "dataset", "target_dataset", "csv_path"),
        "metadata": _first(output, "metadata", "meta_path"),
        "manifest": _first(output, "source_manifest", "manifest", "manifest_path"),
        "chart_data": _first(output, "chart_data", "chart", "chart_path", "chart_data_path"),
        "row_count": _first(output, "row_count", "rows_count", "rows_generated"),
        "limitations": _list(_first(output, "limitations", "possible_limitations")),
    }


def _csv_sample(path: Path, limit: int = 3) -> list[str]:
    if not path.exists():
        return []
    rows: list[str] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index > limit:
                break
            rows.append(",".join(row))
    return rows


def _first(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _append_items(lines: list[str], items: list[str], empty: str) -> None:
    if not items:
        lines.append(f"- {empty}")
        return
    for item in items:
        lines.append(f"- {item}")


def _period(intent: ResearchIntent) -> str:
    if not intent.time_range:
        return "не задано"
    if intent.time_range.raw:
        return intent.time_range.raw
    if intent.time_range.start_year and intent.time_range.end_year:
        return f"{intent.time_range.start_year}-{intent.time_range.end_year}"
    if intent.time_range.start_year:
        return f"с {intent.time_range.start_year}"
    if intent.time_range.end_year:
        return f"по {intent.time_range.end_year}"
    return "не задано"


def _no_results_reason(rerank: DatasetRerankResponse | None) -> str | None:
    return rerank.no_results_reason if rerank else None


def _join(values: list[Any]) -> str:
    cleaned = [str(item).strip() for item in values if item is not None and str(item).strip()]
    return "; ".join(cleaned) if cleaned else "не задано"


def _value(value: Any, default: str = "не задано") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default
