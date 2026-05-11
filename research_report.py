import csv
from pathlib import Path
from typing import Any

from assembly_planner import DatasetBuildPlan, SourceDatasetReference
from dataset_structure import DatasetColumn, TargetDatasetStructure
from parser.intent_parser import Complexity, IntentType, ResearchIntent
from orchestrator import ClarificationRequest, OrchestrationResult, OrchestrationStatus
from designer.research_designer import ResearchStudyDesign


COMPLEXITY_LABELS = {
    Complexity.EASY: "легкий",
    Complexity.MEDIUM: "средний",
    Complexity.COMPLEX: "сложный",
}

INTENT_LABELS = {
    IntentType.SIMPLE_DATA: "простые данные",
    IntentType.COMPARATIVE: "сравнительный запрос",
    IntentType.RESEARCH: "исследовательский запрос",
    IntentType.DERIVED: "производная метрика",
    IntentType.AMBIGUOUS: "неоднозначный запрос",
    IntentType.NO_DATA: "нет данных",
    IntentType.UNSUPPORTED: "неподдерживаемый запрос",
}


def generate_research_report(
    result: OrchestrationResult,
    build_run: dict[str, object] | None = None,
    dataset_output_dir: str | Path | None = None,
) -> str:
    """Render a human-readable research trail in the style of the demo examples."""
    lines: list[str] = []
    intent = result.intent

    lines.append("# Исследовательский вывод")
    lines.append("")
    lines.append(f"Запрос пользователя: {intent.original_query}")
    lines.append(
        f"Тип запроса: {INTENT_LABELS.get(intent.intent_type, intent.intent_type.value)} "
        f"({intent.intent_type.value})"
    )
    lines.append(
        f"Сложность: {COMPLEXITY_LABELS.get(intent.complexity, intent.complexity.value)}"
    )
    if result.message:
        lines.append(f"Статус: {result.message}")
    lines.append("")

    if result.status == OrchestrationStatus.NO_DATA or intent.intent_type == IntentType.NO_DATA:
        _append_no_data_report(lines, intent)
        return "\n".join(lines).rstrip() + "\n"

    if result.status == OrchestrationStatus.NEEDS_CLARIFICATION:
        _append_definition(lines, intent, result.clarification_requests)
        lines.append("## Следующий шаг")
        lines.append("")
        lines.append(
            "Система не начинает сборку данных до ответа пользователя или применения явного default-поведения."
        )
        return "\n".join(lines).rstrip() + "\n"

    _append_definition(lines, intent, result.clarification_requests)
    _append_design(lines, result.research_design, intent)
    _append_structure(lines, result.dataset_structure)
    _append_plan(lines, result.build_plan, intent)
    _append_built_artifact(lines, result, build_run, dataset_output_dir)

    return "\n".join(lines).rstrip() + "\n"


def _append_no_data_report(lines: list[str], intent: ResearchIntent) -> None:
    lines.append("## Шаг 1. Определение исследования")
    lines.append("")
    lines.append(f"Тема: {_value(intent.topic)}")
    lines.append(f"География: {_join_or_dash(intent.geography or intent.objects)}")
    lines.append(f"Период: {_period_label(intent)}")
    lines.append(f"Намерение: запрос на данные с вероятно отсутствующим открытым статистическим рядом")
    lines.append("")
    lines.append("## Вердикт по данным")
    lines.append("")
    lines.append(f"Вердикт: {_value(intent.data_availability.verdict, 'данные недоступны в верифицированных источниках')}")
    lines.append("Пояснения:")
    _append_bullets(lines, intent.data_availability.reasons, "причины отсутствия данных не детализированы")
    lines.append("")
    lines.append("Что предлагается взамен:")
    _append_bullets(lines, intent.data_availability.alternatives, "можно собрать близкие проверяемые показатели или исследовательские публикации с явной пометкой их природы")


def _append_definition(
    lines: list[str],
    intent: ResearchIntent,
    clarification_requests: list[ClarificationRequest],
) -> None:
    lines.append("## Шаг 1. Определение исследования")
    lines.append("")
    lines.append(f"Тема: {_value(intent.topic)}")
    lines.append(f"География: {_join_or_dash(intent.geography or intent.objects or intent.entities)}")
    lines.append(f"Период: {_period_label(intent)}")
    lines.append(f"Частота: {_value(intent.frequency)}")
    lines.append(f"Дисциплинарный ракурс: {_value(intent.disciplinary_perspective)}")
    lines.append("Намерение: " + _intent_purpose(intent))
    lines.append("")
    lines.append("Сущности:")
    lines.append(f"- indicators: {_join_or_dash(intent.indicators)}")
    lines.append(f"- objects: {_join_or_dash(intent.objects or intent.geography or intent.entities)}")
    lines.append(f"- period: {_period_label(intent)}")
    lines.append(f"- frequency: {_value(intent.frequency)}")
    lines.append("")
    lines.append("Исследовательские вопросы:")
    _append_bullets(lines, intent.research_questions, "исследовательские вопросы не заданы отдельно; используется задача из исходного запроса")
    lines.append("")
    lines.append("Уточняющие вопросы пользователю и default-поведение:")
    if clarification_requests:
        for request in clarification_requests:
            lines.append(f"- Вопрос: {request.question}")
            if request.default_assumption:
                lines.append(f"  default при отсутствии ответа: {request.default_assumption}")
            lines.append(f"  причина: {request.reason}")
    elif intent.clarifying_questions:
        for index, question in enumerate(intent.clarifying_questions):
            default = intent.assumptions_if_no_answer[index] if index < len(intent.assumptions_if_no_answer) else None
            lines.append(f"- Вопрос: {question}")
            if default:
                lines.append(f"  default при отсутствии ответа: {default}")
    elif intent.assumptions_if_no_answer:
        _append_bullets(lines, intent.assumptions_if_no_answer, "не требуются")
    else:
        lines.append("- не требуются; запрос содержит достаточные географию, период, показатель и частоту")
    lines.append("")


def _append_design(
    lines: list[str],
    design: ResearchStudyDesign | None,
    intent: ResearchIntent,
) -> None:
    lines.append("## Шаг 2. Дизайн исследования")
    lines.append("")
    if design is None:
        lines.append("Дизайн исследования еще не сформирован.")
        lines.append("")
        return

    lines.append(f"Краткое описание: {_value(design.design_summary)}")
    lines.append("Гипотезы:")
    if design.hypotheses:
        for hypothesis in design.hypotheses:
            prefix = f"{hypothesis.id}: " if hypothesis.id else ""
            lines.append(f"- {prefix}{hypothesis.statement}")
    elif intent.research_design and intent.research_design.hypotheses:
        _append_bullets(lines, intent.research_design.hypotheses, "не требуются")
    else:
        lines.append("- для простого ряда гипотезы не обязательны; основной ожидаемый результат - корректный временной ряд")
    lines.append("")

    lines.append("Измерения и индикаторы:")
    if design.required_measurements:
        for measurement in design.required_measurements:
            parts = [
                measurement.name,
                _value(measurement.definition, None),
                f"единица: {measurement.unit}" if measurement.unit else None,
                f"роль: {measurement.role}" if measurement.role else None,
                f"источники-кандидаты: {', '.join(measurement.source_candidates)}" if measurement.source_candidates else None,
            ]
            lines.append("- " + "; ".join(part for part in parts if part))
    else:
        for spec in intent.indicator_specs:
            parts = [
                spec.name,
                _value(spec.definition, None),
                f"единица: {spec.unit}" if spec.unit else None,
                f"роль: {spec.role}" if spec.role else None,
            ]
            lines.append("- " + "; ".join(part for part in parts if part))
        if not intent.indicator_specs:
            _append_bullets(lines, intent.indicators, "индикаторы не заданы")
    lines.append("")

    lines.append("Методы группировки:")
    if design.grouping_rules:
        for rule in design.grouping_rules:
            levels = f"; уровни: {', '.join(rule.levels)}" if rule.levels else ""
            lines.append(f"- {rule.dimension}{levels}; {rule.rationale}")
    elif intent.research_design and intent.research_design.grouping:
        _append_bullets(lines, intent.research_design.grouping, "группировки не требуются")
    else:
        lines.append("- временной ряд или панель в зернистости целевого датасета")
    lines.append("")

    lines.append("Производные метрики:")
    if design.derived_metrics:
        for metric in design.derived_metrics:
            lines.append(f"- {metric.name}")
            lines.append(f"  formula: {metric.formula}")
            lines.append(f"  inputs: {_join_or_dash(metric.inputs)}")
            if metric.normalization:
                lines.append(f"  нормализация: {metric.normalization}")
            if metric.aggregation_rules:
                lines.append(f"  правило агрегирования: {'; '.join(metric.aggregation_rules)}")
    elif intent.derived_metrics:
        for metric in intent.derived_metrics:
            lines.append(f"- {metric.name}")
            if metric.formula:
                lines.append(f"  formula: {metric.formula}")
            lines.append(f"  inputs: {_join_or_dash(metric.inputs)}")
            if metric.normalization:
                lines.append(f"  нормализация: {metric.normalization}")
            if metric.aggregation_rules:
                lines.append(f"  правило агрегирования: {'; '.join(metric.aggregation_rules)}")
    else:
        lines.append("- не требуются")
    lines.append("")

    lines.append("Ожидаемые визуализации:")
    if design.visualizations:
        for visualization in design.visualizations:
            axes = " x=" + visualization.x_axis if visualization.x_axis else ""
            axes += " y=" + visualization.y_axis if visualization.y_axis else ""
            lines.append(f"- {visualization.title} ({visualization.chart_type}{axes})")
    elif intent.research_design and intent.research_design.expected_visuals:
        _append_bullets(lines, intent.research_design.expected_visuals, "не заданы")
    else:
        lines.append("- таблица значений; базовый график в соответствии с зернистостью ряда")
    lines.append("")

    lines.append("Особые случаи:")
    edge_cases = []
    if design.methodology_notes:
        edge_cases.append(design.methodology_notes)
    edge_cases.extend(check.action_if_failed for check in design.data_quality_checks)
    edge_cases.extend(design.blocking_reasons)
    _append_bullets(lines, edge_cases, "нет специальных ограничений, кроме сохранения пропусков и источников")
    lines.append("")


def _append_structure(lines: list[str], structure: TargetDatasetStructure | None) -> None:
    lines.append("## Шаг 3. Структура целевого датасета")
    lines.append("")
    if structure is None:
        lines.append("Структура целевого датасета еще не сформирована.")
        lines.append("")
        return

    lines.append(f"Гранулярность строки: {structure.row_grain}")
    lines.append(f"Первичный ключ: {_join_or_dash(structure.primary_key)}")
    lines.append(f"Ожидаемая частота: {_value(structure.expected_frequency)}")
    lines.append(f"Период: {_value(structure.time_range)}")
    lines.append("")
    _append_columns(lines, "Измерения:", structure.columns, "dimension")
    _append_columns(lines, "Индикаторы:", structure.columns, "indicator")
    _append_columns(lines, "Производные метрики:", structure.columns, "derived_metric")
    _append_columns(lines, "Метаданные:", structure.columns, "metadata")
    if structure.design_notes:
        lines.append("Примечания к структуре:")
        _append_bullets(lines, structure.design_notes, "нет")
    lines.append("")


def _append_plan(
    lines: list[str],
    plan: DatasetBuildPlan | None,
    intent: ResearchIntent,
) -> None:
    lines.append("## Шаг 4. План сборки")
    lines.append("")
    if plan is None:
        lines.append("План сборки еще не сформирован.")
        lines.append("")
        return

    lines.append(
        "Проверка реестра: "
        + ("выполнена" if plan.registry_checked else "не выполнена или реестр недоступен")
    )
    if plan.registry_assets:
        assets = ", ".join(
            f"{name}={'есть' if exists else 'нет'}"
            for name, exists in plan.registry_assets.items()
        )
        lines.append(f"Состояние локальных индексов: {assets}")
    lines.append(f"Готовый датасет найден: {'да' if plan.can_use_existing_dataset else 'нет'}")
    lines.append(f"Можно собрать из локальных источников: {'да' if plan.can_build_from_local_sources else 'нет'}")
    lines.append("")

    lines.append("Реестровые датасеты-предшественники:")
    if plan.predecessor_datasets:
        for source in plan.predecessor_datasets:
            _append_source_reference(lines, source, intent)
    else:
        lines.append("- не найдены")
    lines.append("")

    lines.append("Рассмотренные и отвергнутые источники:")
    audit_entries = [
        entry for entry in plan.source_audit
        if entry.decision in {"considered", "rejected", "unavailable"}
    ][:8]
    if audit_entries:
        for entry in audit_entries:
            score = f"; score={entry.score:.3f}" if entry.score is not None else ""
            lines.append(f"- {entry.decision}: {entry.title}{score}. {entry.reason}")
    else:
        lines.append("- отдельный аудит источников не сформирован")
    lines.append("")

    lines.append("Операции сборки:")
    for step in plan.assembly_steps:
        lines.append(f"- {step.id}: {step.description}")
    if not plan.assembly_steps:
        lines.append("- не заданы")
    lines.append("")

    lines.append("Обработка пропусков и ограничения:")
    missing_rules = [
        "пропуски сохраняются как NULL; автоматическая интерполяция не выполняется",
        *plan.blocking_reasons,
        *plan.warnings,
    ]
    _append_bullets(lines, missing_rules, "пропуски не ожидаются")
    lines.append("")


def _append_built_artifact(
    lines: list[str],
    result: OrchestrationResult,
    build_run: dict[str, object] | None,
    dataset_output_dir: str | Path | None,
) -> None:
    lines.append("## Шаг 5. Собранный артефакт")
    lines.append("")
    output_dir = Path(dataset_output_dir or "artifacts/generated_dataset")
    lines.append("Команда сборки из папки со скриптом:")
    lines.append("`python build_target_dataset.py --output-dir artifacts/generated_dataset`")
    lines.append("")

    if result.build_script:
        lines.append(f"Скрипт: {result.build_script.filename}")
        lines.append(f"Ожидаемые файлы: {_join_or_dash(result.build_script.produces)}")

    output = _build_output(build_run)
    dataset_path = Path(str(output.get("dataset") or output_dir / "target_dataset.csv"))
    metadata_path = Path(str(output.get("metadata") or output_dir / "target_dataset.metadata.json"))
    manifest_path = Path(str(output.get("source_manifest") or output_dir / "source_manifest.json"))
    row_count = output.get("row_count")

    if build_run:
        lines.append(f"Имя файла: {dataset_path.name}")
        lines.append("Формат: CSV (UTF-8)")
        lines.append(f"Объем: {_volume_label(dataset_path, row_count)}")
        coverage = output.get("coverage")
        if isinstance(coverage, dict):
            available_years = coverage.get("available_years")
            missing_years = coverage.get("missing_years")
            if available_years:
                lines.append(f"Фактическое покрытие лет: {_join_or_dash(list(available_years))}")
            if missing_years:
                lines.append(f"Отсутствующие годы относительно запроса: {_join_or_dash(list(missing_years))}")
        lines.append(f"Метаданные: {metadata_path}")
        lines.append(f"Source manifest: {manifest_path}")
        sample = _csv_sample(dataset_path)
        if sample:
            lines.append("")
            lines.append("Пример строк:")
            lines.append("```csv")
            lines.extend(sample)
            lines.append("```")
    else:
        lines.append(f"Сборка еще не запущена. При запуске датасет будет записан в: {output_dir}")
    lines.append("")


def _append_source_reference(
    lines: list[str],
    source: SourceDatasetReference,
    intent: ResearchIntent,
) -> None:
    filters = []
    if intent.geography or intent.objects:
        filters.append("география = " + _join_or_dash(intent.geography or intent.objects))
    if intent.time_range:
        filters.append("период = " + _period_label(intent))
    if intent.frequency:
        filters.append("частота = " + intent.frequency)
    if intent.indicators:
        filters.append("показатели = " + _join_or_dash(intent.indicators))

    lines.append(f"- Источник: {_value(source.source_name or source.source)}")
    lines.append(f"  датасет: {source.title}")
    if source.dataset_id:
        lines.append(f"  dataset_id: {source.dataset_id}")
    if source.data_path:
        lines.append(f"  локальный файл: {source.data_path}")
    if source.source_url:
        lines.append(f"  ссылка: {source.source_url}")
    if filters:
        lines.append(f"  фильтры: {'; '.join(filters)}")
    lines.append(f"  уверенность: {source.usefulness_confidence:.2f}")
    lines.append(f"  почему выбран: {source.why_selected}")
    if source.possible_limitations:
        lines.append(f"  ограничения: {'; '.join(source.possible_limitations)}")


def _append_columns(
    lines: list[str],
    title: str,
    columns: list[DatasetColumn],
    role: str,
) -> None:
    selected = [column for column in columns if column.role == role]
    lines.append(title)
    if not selected:
        lines.append("- не требуются")
        return
    for column in selected:
        unit = f", {column.unit}" if column.unit else ""
        nullable = "nullable" if column.nullable else "not null"
        lines.append(
            f"- {column.name} ({column.dtype}{unit}, {nullable}) — {column.title}"
        )


def _append_bullets(lines: list[str], items: list[str], empty: str) -> None:
    if not items:
        lines.append(f"- {empty}")
        return
    for item in items:
        lines.append(f"- {item}")


def _build_output(build_run: dict[str, object] | None) -> dict[str, object]:
    if not build_run:
        return {}
    output = build_run.get("output")
    if isinstance(output, dict):
        return output
    return {}


def _csv_sample(path: Path, limit: int = 3) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = []
        for index, row in enumerate(reader):
            if index > limit:
                break
            rows.append(",".join("" if value is None else str(value) for value in row))
    return rows


def _volume_label(path: Path, row_count: object) -> str:
    parts = []
    if row_count is not None:
        parts.append(f"{row_count} строк")
    if path.exists():
        parts.append(f"{path.stat().st_size} байт")
    return ", ".join(parts) if parts else "объем пока неизвестен"


def _period_label(intent: ResearchIntent) -> str:
    if intent.time_range is None:
        return "не задан"
    if intent.time_range.raw:
        return intent.time_range.raw
    start = intent.time_range.start_year
    end = intent.time_range.end_year
    if start and end:
        return f"{start}-{end}"
    if start:
        return f"с {start}"
    if end:
        return f"по {end}"
    return "не задан"


def _intent_purpose(intent: ResearchIntent) -> str:
    if intent.intent_type == IntentType.SIMPLE_DATA:
        return "получить временной ряд одного или нескольких показателей по заданному объекту"
    if intent.intent_type == IntentType.COMPARATIVE:
        return "сравнить показатель между объектами в единой зернистости"
    if intent.intent_type == IntentType.RESEARCH:
        return "проверить исследовательские вопросы и гипотезы на собранном датасете"
    if intent.intent_type == IntentType.DERIVED:
        return "рассчитать производную метрику на основе исходных показателей"
    if intent.intent_type == IntentType.AMBIGUOUS:
        return "снять неоднозначности перед сборкой данных"
    return intent.intent_type.value


def _join_or_dash(values: list[Any]) -> str:
    cleaned = [str(value) for value in values if value is not None and str(value).strip()]
    return "; ".join(cleaned) if cleaned else "не задано"


def _value(value: Any, default: str | None = "не задано") -> str:
    if value is None:
        return default or ""
    text = str(value).strip()
    return text if text else (default or "")
