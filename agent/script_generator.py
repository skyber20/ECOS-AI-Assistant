import json
import subprocess
import sys
import tempfile
import textwrap
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from dataset_reranker import DatasetRerankResponse
from dataset_structure import TargetDatasetStructure
from intent_parser import (
    IntentParserError,
    LLMSettings,
    ResearchIntent,
    _create_json_completion,
    _load_json_object,
    create_llm_settings,
    none_to_empty_list,
)
from research_designer import ResearchStudyDesign


class GeneratedOutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["dataset", "metadata", "manifest", "table", "chart", "sql", "text", "json"]
    path: str | None = None
    description: str | None = None


class GeneratedBuildScript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = "build_target_dataset.py"
    language: Literal["python"] = "python"
    entrypoint: str = "run"
    content: str
    usage: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[GeneratedOutputSpec] = Field(default_factory=list)
    possible_limitations: list[str] = Field(default_factory=list)

    @field_validator("inputs", "outputs", "possible_limitations", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class ScriptExecutionAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int
    ok: bool
    returncode: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    output: dict[str, Any] | None = None
    error: str | None = None


class BuildScriptRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["succeeded", "failed"]
    output_dir: str
    attempts: list[ScriptExecutionAttempt] = Field(default_factory=list)
    output: dict[str, Any] | None = None
    final_error: str | None = None
    repaired: bool = False

    @field_validator("attempts", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


CODE_GENERATION_PROMPT = """Ты генерируешь исполняемый Python-скрипт для шага сборки и анализа датасета в прототипе ECOS AI Assistant.

На входе один JSON-контекст:
- intent: формальное описание запроса;
- research_design: сжатый дизайн исследования, включая гипотезы, метрики и визуализации;
- target_structure: целевая структура датасета;
- source_datasets: датасеты, выбранные RAG/reranker только по metadata;
- execution_contract: как должен запускаться код.

Правила:
- Верни только валидный JSON по схеме GeneratedBuildScript без Markdown.
- language всегда "python", filename "build_target_dataset.py", entrypoint "run".
- content должен содержать функцию run(context, output_dir), которая возвращает dict.
- Не добавляй CLI, argparse, input, сетевые запросы, комментарии и docstring.
- Код может использовать стандартную библиотеку и optional libraries из execution_contract.available_libraries.
- Если source_datasets содержит file_format=".parquet" и execution_contract.available_libraries.pyarrow=true, обязательно реализуй чтение parquet через pyarrow.parquet.read_table(...). Импорт pyarrow должен быть внутри try/except и с graceful fallback.
- Не используй pandas, polars, duckdb, matplotlib, seaborn или другие библиотеки, которых нет в execution_contract.available_libraries.
- При чтении parquet через pyarrow конвертируй table.to_pylist() и работай со списком dict без pandas.
- Для чтения локального источника используй source.resolved_data_path, если он заполнен, иначе source.data_path относительно context.project_root.
- Все числа и строки наблюдений должны извлекаться только детерминированным кодом из файлов source_datasets.data_path.
- Не придумывай значения, колонки, периоды, географию и coverage. Если данных или схемы не хватает, запиши это в possible_limitations и output, а не выдумывай.
- Не полагайся на названия raw-колонок, которых нет в metadata. Если код читает файл, он должен сам определить доступные заголовки/ключи и работать осторожно.
- Если формат файла не поддержан доступными библиотеками, не падай: создай пустой target_dataset.csv с заголовками целевой структуры и limitation.
- Всегда формируй source_manifest.json и target_dataset.metadata.json.
- Если получается собрать строки, target_dataset.csv должен содержать только колонки target_structure.columns.
- Если дизайн просит графики или SQL/table result, сформируй дополнительный артефакт без неподтвержденных значений: chart_data.csv, chart_spec.json, sql_query.sql, analysis_summary.json или аналогичный файл.
- Все source_name, source_url, source_dataset_id должны сохранять lineage из metadata, когда возможно.
- usage коротко объясняет, как вызвать run(context, output_dir) из Python.
- possible_limitations пиши по-русски, предметно и только из metadata/context.
"""


REPAIR_PROMPT = """Ты исправляешь Python-скрипт сборки датасета.

На входе:
- исходный JSON-контекст;
- предыдущий GeneratedBuildScript;
- ошибка запуска.

Правила:
- Верни только полный валидный JSON GeneratedBuildScript без Markdown.
- Сохрани исследовательский смысл, source_datasets и target_structure.
- Исправляй только причину ошибки запуска.
- Не добавляй CLI, argparse, input, сетевые запросы, комментарии и docstring.
- Не используй pandas, polars, duckdb, matplotlib, seaborn или другие библиотеки, которых нет в execution_contract.available_libraries.
- Если ошибка связана с ModuleNotFoundError для pandas или другой неразрешенной библиотеки, убери эту библиотеку и перепиши код на standard library / pyarrow.to_pylist().
- Не выдумывай данные и колонки.
- Если ошибку нельзя исправить без знания схемы файла или зависимости, сделай graceful fallback с пустым CSV, metadata, manifest и limitation.
"""


class ScriptGeneratorError(RuntimeError):
    pass


def generate_build_script(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
    structure: TargetDatasetStructure,
    dataset_rerank: DatasetRerankResponse | None,
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> GeneratedBuildScript:
    context = build_generation_context(intent, design, structure, dataset_rerank)
    return _render_template_build_script(context)


def generate_and_run_build_script(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
    structure: TargetDatasetStructure,
    dataset_rerank: DatasetRerankResponse | None,
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
    output_dir: str | Path = "artifacts/latest_run/generated_dataset",
    max_tries: int = 3,
) -> tuple[GeneratedBuildScript, BuildScriptRun]:
    context = build_generation_context(intent, design, structure, dataset_rerank)
    llm_settings = settings or create_llm_settings(provider=provider, model=model)
    script = generate_build_script(
        intent=intent,
        design=design,
        structure=structure,
        dataset_rerank=dataset_rerank,
        settings=llm_settings,
    )
    attempts: list[ScriptExecutionAttempt] = []
    repaired = False

    for attempt_number in range(1, max_tries + 1):
        attempt = _execute_script(script, context, output_dir, attempt_number)
        attempts.append(attempt)
        if attempt.ok:
            return script, BuildScriptRun(
                status="succeeded",
                output_dir=str(Path(output_dir).resolve()),
                attempts=attempts,
                output=attempt.output,
                repaired=repaired,
            )
        if attempt_number >= max_tries:
            break
        script = _repair_script(script, context, attempt, llm_settings)
        repaired = True

    final_error = attempts[-1].error if attempts else "Скрипт не запускался."
    return script, BuildScriptRun(
        status="failed",
        output_dir=str(Path(output_dir).resolve()),
        attempts=attempts,
        final_error=final_error,
        repaired=repaired,
    )


def build_generation_context(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
    structure: TargetDatasetStructure,
    dataset_rerank: DatasetRerankResponse | None,
) -> dict[str, Any]:
    return {
        "project_root": str(Path.cwd()),
        "intent": _intent_payload(intent),
        "research_design": _design_payload(design),
        "target_structure": structure.model_dump(mode="json"),
        "source_datasets": _source_payloads(dataset_rerank),
        "execution_contract": {
            "entrypoint": "run(context, output_dir)",
            "max_tries": 3,
            "available_libraries": _available_libraries(),
            "source_file_formats": _source_file_formats(dataset_rerank),
            "default_outputs": [
                "target_dataset.csv",
                "target_dataset.metadata.json",
                "source_manifest.json",
            ],
        },
    }


def _render_template_build_script(context: dict[str, Any]) -> GeneratedBuildScript:
    limitations: list[str] = []
    for source in context.get("source_datasets", []):
        limitations.extend(source.get("possible_limitations") or [])
        if not source.get("resolved_data_path"):
            title = source.get("title") or source.get("record_id") or "источник"
            limitations.append(f"Для источника '{title}' нет локального data_path.")

    return GeneratedBuildScript(
        content=_template_script_content(),
        usage="Из Python: import build_target_dataset; build_target_dataset.run(context, output_dir).",
        inputs=["context"],
        outputs=[
            GeneratedOutputSpec(name="target_dataset.csv", kind="dataset", path="target_dataset.csv"),
            GeneratedOutputSpec(name="target_dataset.metadata.json", kind="metadata", path="target_dataset.metadata.json"),
            GeneratedOutputSpec(name="source_manifest.json", kind="manifest", path="source_manifest.json"),
            GeneratedOutputSpec(name="chart_data.csv", kind="table", path="chart_data.csv"),
        ],
        possible_limitations=list(dict.fromkeys(item for item in limitations if str(item).strip())),
    )


def _template_script_content() -> str:
    return textwrap.dedent(
        r'''
        import csv
        import gzip
        import json
        import math
        import re
        from datetime import datetime, timezone
        from pathlib import Path


        GEO_ALIASES = {
            "россия": ["россия", "российская федерация", "russian federation", "russia", "rus"],
            "российская федерация": ["россия", "российская федерация", "russian federation", "russia", "rus"],
            "сша": ["сша", "соединенные штаты", "соединенные штаты америки", "united states", "united states of america", "usa"],
            "соединенные штаты": ["сша", "соединенные штаты", "соединенные штаты америки", "united states", "united states of america", "usa"],
            "казахстан": ["казахстан", "kazakhstan", "kaz"],
            "бангладеш": ["бангладеш", "bangladesh", "bgd"],
            "мир": ["мир", "world", "wld"],
        }


        def run(context, output_dir):
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            target_structure = context.get("target_structure") or {}
            columns = target_structure.get("columns") or []
            fieldnames = [column.get("name") for column in columns if column.get("name")]
            source_datasets = context.get("source_datasets") or []

            limitations = []
            source_manifest = []
            rows = []

            for source in source_datasets:
                path = resolve_data_path(context, source)
                source_rows, read_error = read_source_rows(path)
                normalized_rows = normalize_source_rows(source_rows)
                matched_rows = []

                if read_error:
                    limitations.append(read_error)
                if not path:
                    limitations.append(f"Для источника '{source.get('title') or source.get('record_id')}' нет локального data_path.")

                for item in source.get("possible_limitations") or []:
                    limitations.append(str(item))

                for source_row in normalized_rows:
                    if not row_matches_target(source_row, context):
                        continue
                    projected = project_row(source_row, source, context)
                    if has_observation_value(projected, columns):
                        matched_rows.append(projected)

                rows.extend(matched_rows)
                source_manifest.append({
                    "record_id": source.get("record_id"),
                    "dataset_id": source.get("dataset_id"),
                    "title": source.get("title"),
                    "source": source.get("source"),
                    "data_path": str(path) if path else None,
                    "source_url": source.get("source_url"),
                    "rows_read": len(normalized_rows),
                    "rows_matched": len(matched_rows),
                    "why_matched": source.get("why_matched"),
                    "possible_limitations": source.get("possible_limitations") or [],
                })

            rows = dedupe_rows(rows, target_structure.get("primary_key") or [])
            rows.sort(key=row_sort_key)
            apply_derived_metrics(rows, columns, context)
            coverage = build_coverage(rows, context)

            dataset_path = output_path / "target_dataset.csv"
            metadata_path = output_path / "target_dataset.metadata.json"
            manifest_path = output_path / "source_manifest.json"
            chart_path = output_path / "chart_data.csv"

            write_csv(dataset_path, fieldnames, rows)
            write_json(manifest_path, source_manifest)
            chart_rows = build_chart_rows(rows, columns, context)
            write_csv(chart_path, ["visualization_id", "title", "geo", "year", "metric", "value"], chart_rows)

            limitations = unique_text(limitations)
            write_json(metadata_path, {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "row_count": len(rows),
                "coverage": coverage,
                "possible_limitations": limitations,
                "target_structure": target_structure,
                "source_manifest": source_manifest,
                "outputs": {
                    "dataset": str(dataset_path),
                    "metadata": str(metadata_path),
                    "source_manifest": str(manifest_path),
                    "chart_data": str(chart_path),
                },
            })

            return {
                "dataset": str(dataset_path),
                "metadata": str(metadata_path),
                "source_manifest": str(manifest_path),
                "chart_data": str(chart_path),
                "row_count": len(rows),
                "coverage": coverage,
                "limitations": limitations,
            }


        def resolve_data_path(context, source):
            value = source.get("resolved_data_path") or source.get("data_path")
            if not value:
                return None
            path = Path(value)
            if not path.is_absolute():
                path = Path(context.get("project_root") or ".") / path
            return path


        def read_source_rows(path):
            if path is None:
                return [], None
            if not path.exists():
                return [], f"Файл источника не найден: {path}"

            suffix = path.suffix.lower()
            suffixes = [item.lower() for item in path.suffixes]
            try:
                if suffix == ".csv":
                    with path.open(newline="", encoding="utf-8-sig") as file:
                        return list(csv.DictReader(file)), None
                if suffixes[-2:] == [".jsonl", ".gz"]:
                    rows = []
                    with gzip.open(path, mode="rt", encoding="utf-8") as file:
                        for line in file:
                            if line.strip():
                                rows.append(json.loads(line))
                    return rows, None
                if suffix == ".jsonl":
                    rows = []
                    with path.open(encoding="utf-8") as file:
                        for line in file:
                            if line.strip():
                                rows.append(json.loads(line))
                    return rows, None
                if suffix == ".json":
                    with path.open(encoding="utf-8") as file:
                        payload = json.load(file)
                    if isinstance(payload, list):
                        return [row for row in payload if isinstance(row, dict)], None
                    if isinstance(payload, dict):
                        for value in payload.values():
                            if isinstance(value, list):
                                return [row for row in value if isinstance(row, dict)], None
                    return [], None
                if suffix == ".parquet":
                    try:
                        import pyarrow.parquet as pq
                    except ImportError:
                        return [], "Библиотека pyarrow недоступна, parquet не прочитан."
                    table = pq.read_table(path)
                    return [dict(row) for row in table.to_pylist()], None
            except Exception as exc:
                return [], f"Ошибка чтения источника {path}: {exc}"

            return [], f"Формат источника не поддержан: {path.suffix}"


        def normalize_source_rows(rows):
            if not rows:
                return []
            first = rows[0]
            column_keys = sorted(
                [key for key in first if re.fullmatch(r"column\d+", str(key))],
                key=column_sort_key,
            )
            if not column_keys:
                return rows

            year_columns = []
            dimension_columns = []
            for key in column_keys:
                year = int_year(first.get(key))
                if year is not None:
                    year_columns.append((key, year))
                elif first.get(key):
                    dimension_columns.append((key, str(first[key]).strip()))

            if not year_columns:
                return rows

            normalized = []
            for source_row in rows[1:]:
                base = {
                    dimension_name: clean_cell(source_row.get(key))
                    for key, dimension_name in dimension_columns
                }
                for key, year in year_columns:
                    value = source_row.get(key)
                    if value is None or str(value).strip() == "":
                        continue
                    normalized.append({**base, "year": year, "value": value})
            return normalized


        def column_sort_key(name):
            match = re.search(r"\d+", str(name))
            return int(match.group(0)) if match else 10_000


        def row_matches_target(source_row, context):
            start_year, end_year = target_year_bounds(context)
            year = row_year(source_row)
            if (start_year is not None or end_year is not None) and year is None:
                return False
            if start_year is not None and year < start_year:
                return False
            if end_year is not None and year > end_year:
                return False

            variants_by_label = target_geo_variants(context)
            if variants_by_label:
                values = source_geo_values(source_row)
                if not values:
                    return False
                if not any(geo_value_matches(value, variants_by_label) for value in values):
                    return False

            return True


        def project_row(source_row, source, context):
            now = datetime.now(timezone.utc).isoformat()
            target_structure = context.get("target_structure") or {}
            columns = target_structure.get("columns") or []
            row = {}
            for column in columns:
                name = column.get("name")
                if name:
                    row[name] = value_for_column(column, source_row, source, context)
            row["source_name"] = row.get("source_name") or source.get("title") or source.get("source")
            row["source_url"] = row.get("source_url") or source.get("source_url")
            row["source_dataset_id"] = row.get("source_dataset_id") or source.get("dataset_id") or source.get("record_id")
            row["downloaded_at"] = row.get("downloaded_at") or now
            return row


        def value_for_column(column, source_row, source, context):
            name = column.get("name")
            role = column.get("role")
            source_field = column.get("source_field")

            direct = lookup_value(source_row, name)
            if direct is not None:
                return clean_cell(direct)

            if source_field:
                for field in str(source_field).split(","):
                    value = lookup_value(source_row, field.strip())
                    if value is not None:
                        return clean_cell(value)

            if name == "geo":
                return display_geo(source_row, context)
            if name == "year":
                return row_year(source_row)
            if name == "period":
                return clean_cell(first_non_empty(source_row.get("period"), source_row.get("date"), source_row.get("time")))

            if role == "indicator":
                if not indicator_allows_source_value(column, source):
                    return None
                value = first_non_empty(
                    lookup_value(source_row, "value"),
                    lookup_value(source_row, "Value"),
                    lookup_value(source_row, "VALUE"),
                    lookup_value(source_row, "obs_value"),
                )
                return numeric_or_clean(value)

            return None


        def indicator_allows_source_value(column, source):
            column_text = " ".join(str(column.get(key) or "") for key in ("name", "title", "source_field", "description")).lower()
            source_text = " ".join(str(source.get(key) or "") for key in ("title", "description", "unit")).lower()
            source_is_rate = "%" in source_text or "growth" in source_text or "annual" in source_text or "темп" in source_text
            if source_is_rate and any(token in column_text for token in ("index_level", "уровень", "абсолют", "индексная единица")):
                return False
            if source_is_rate:
                return any(token in column_text for token in ("инфля", "inflation", "rate", "yoy", "годов", "annual", "темп", "рост", "growth", "ипц", "cpi"))
            return True


        def apply_derived_metrics(rows, columns, context):
            if not rows:
                return
            indicator_name = first_indicator_column(columns)
            if not indicator_name:
                return

            by_geo = {}
            for row in rows:
                geo = row.get("geo") or ""
                by_geo.setdefault(geo, []).append(row)

            for column in columns:
                if column.get("role") != "derived_metric":
                    continue
                name = column.get("name")
                text = " ".join(str(column.get(key) or "") for key in ("name", "title", "description", "source_field")).lower()
                if not name:
                    continue
                if any(token in text for token in ("cumulative", "накоп")):
                    for geo_rows in by_geo.values():
                        rates = [to_float(row.get(indicator_name)) for row in geo_rows]
                        rates = [value for value in rates if value is not None]
                        cumulative = cumulative_rate(rates)
                        for row in geo_rows:
                            row[name] = cumulative
                elif any(token in text for token in ("volatility", "stddev", "standard", "волат", "отклон")):
                    for geo_rows in by_geo.values():
                        rates = [to_float(row.get(indicator_name)) for row in geo_rows]
                        rates = [value for value in rates if value is not None]
                        value = sample_std(rates)
                        for row in geo_rows:
                            row[name] = value
                elif any(token in text for token in ("ratio", "отнош")):
                    apply_ratio(rows, name, indicator_name)


        def first_indicator_column(columns):
            indicators = [column.get("name") for column in columns if column.get("role") == "indicator" and column.get("name")]
            preferred = [
                column.get("name")
                for column in columns
                if column.get("role") == "indicator"
                and any(token in " ".join(str(column.get(key) or "") for key in ("name", "title", "source_field")).lower() for token in ("inflation", "инфля", "rate", "yoy", "growth", "рост"))
            ]
            return (preferred or indicators or [None])[0]


        def apply_ratio(rows, output_name, indicator_name):
            by_year = {}
            for row in rows:
                year = row.get("year")
                geo = row.get("geo")
                value = to_float(row.get(indicator_name))
                if year is None or not geo or value is None:
                    continue
                by_year.setdefault(year, {})[geo] = value

            geos = list(dict.fromkeys(row.get("geo") for row in rows if row.get("geo")))
            if len(geos) < 2:
                return
            left, right = geos[0], geos[1]

            ratios = {}
            for year, values in by_year.items():
                denominator = values.get(right)
                numerator = values.get(left)
                if numerator is not None and denominator not in (None, 0):
                    ratios[year] = numerator / denominator

            for row in rows:
                if row.get("year") in ratios:
                    row[output_name] = ratios[row.get("year")]


        def build_chart_rows(rows, columns, context):
            visualizations = (context.get("research_design") or {}).get("visualizations") or []
            indicator_name = first_indicator_column(columns)
            if not visualizations or not indicator_name:
                return []
            chart_rows = []
            for visualization in visualizations:
                metric = visualization.get("y_axis") or indicator_name
                metric_column = resolve_metric_column(metric, columns) or indicator_name
                for row in rows:
                    value = row.get(metric_column)
                    if value in (None, ""):
                        continue
                    chart_rows.append({
                        "visualization_id": visualization.get("id"),
                        "title": visualization.get("title"),
                        "geo": row.get("geo"),
                        "year": row.get("year"),
                        "metric": metric_column,
                        "value": value,
                    })
            return chart_rows


        def resolve_metric_column(metric, columns):
            if not metric:
                return None
            metric_key = normalize_key(metric)
            for column in columns:
                values = [column.get("name"), column.get("title"), column.get("source_field")]
                if any(normalize_key(value) == metric_key for value in values if value):
                    return column.get("name")
            for column in columns:
                values = [column.get("name"), column.get("title"), column.get("source_field")]
                if any(metric_key in normalize_key(value) for value in values if value):
                    return column.get("name")
            return None


        def has_observation_value(row, columns):
            for column in columns:
                if column.get("role") != "indicator":
                    continue
                value = row.get(column.get("name"))
                if value not in (None, ""):
                    return True
            return False


        def build_coverage(rows, context):
            requested_years = target_years(context)
            available_years = sorted({
                year for year in (row_year(row) for row in rows) if year is not None
            })
            missing_years = [
                year for year in requested_years if year not in available_years
            ] if requested_years else []
            return {
                "requested_years": requested_years,
                "available_years": available_years,
                "missing_years": missing_years,
                "available_start_year": available_years[0] if available_years else None,
                "available_end_year": available_years[-1] if available_years else None,
            }


        def target_years(context):
            start_year, end_year = target_year_bounds(context)
            if start_year is None or end_year is None:
                return []
            return list(range(start_year, end_year + 1))


        def target_year_bounds(context):
            target_structure = context.get("target_structure") or {}
            intent = context.get("intent") or {}
            value = target_structure.get("time_range")
            if not value and isinstance(intent.get("time_range"), dict):
                time_range = intent["time_range"]
                start = time_range.get("start_year")
                end = time_range.get("end_year")
                if start or end:
                    return start, end
            years = [int(item) for item in re.findall(r"\d{4}", str(value or ""))]
            if len(years) >= 2:
                return min(years), max(years)
            if len(years) == 1:
                return years[0], years[0]
            return None, None


        def target_geo_variants(context):
            target_structure = context.get("target_structure") or {}
            intent = context.get("intent") or {}
            geos = []
            for value in target_structure.get("geography") or []:
                geos.append(value)
            for key in ("geography", "objects", "entities"):
                for value in intent.get(key) or []:
                    geos.append(value)
            result = {}
            for geo in geos:
                label = str(geo).strip()
                if not label:
                    continue
                normalized = normalize_geo(label)
                variants = [normalized]
                variants.extend(GEO_ALIASES.get(normalized, []))
                result[label] = [normalize_geo(item) for item in variants if str(item).strip()]
            return result


        def display_geo(source_row, context):
            variants_by_label = target_geo_variants(context)
            values = source_geo_values(source_row)
            for value in values:
                matched_label = geo_value_matches(value, variants_by_label)
                if matched_label:
                    return matched_label
            return clean_cell(values[0]) if values else None


        def geo_value_matches(value, variants_by_label):
            normalized = normalize_geo(value)
            for label, variants in variants_by_label.items():
                if normalized in variants:
                    return label
            return None


        def source_geo_values(source_row):
            candidates = [
                lookup_value(source_row, "geo"),
                lookup_value(source_row, "country_name"),
                lookup_value(source_row, "country"),
                lookup_value(source_row, "countryiso3code"),
                lookup_value(source_row, "country_id"),
                lookup_value(source_row, "region"),
                lookup_value(source_row, "territory"),
            ]
            for key, value in source_row.items():
                lower_key = str(key).lower()
                if any(token in lower_key for token in ("окато", "территор", "регион", "страна", "субъект")):
                    candidates.append(value)
            return [clean_cell(value) for value in candidates if value is not None and str(value).strip()]


        def row_year(source_row):
            value = first_non_empty(
                lookup_value(source_row, "year"),
                lookup_value(source_row, "date"),
                lookup_value(source_row, "time"),
                lookup_value(source_row, "period"),
                lookup_value(source_row, "Год"),
            )
            if value is None:
                return None
            year = int_year(value)
            if year is not None:
                return year
            match = re.search(r"\d{4}", str(value))
            return int(match.group(0)) if match else None


        def int_year(value):
            if value is None:
                return None
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            text = str(value).strip()
            if re.fullmatch(r"\d{4}(?:\.0)?", text):
                year = int(float(text))
                if 1800 <= year <= 2200:
                    return year
            return None


        def dedupe_rows(rows, primary_key):
            if not primary_key:
                return rows
            result = []
            seen = set()
            for row in rows:
                key = tuple(row.get(item) for item in primary_key)
                source_key = row.get("source_dataset_id")
                full_key = (key, source_key)
                if full_key in seen:
                    continue
                seen.add(full_key)
                result.append(row)
            return result


        def row_sort_key(row):
            return (
                str(row.get("geo") or ""),
                row.get("year") if row.get("year") is not None else 10_000,
                str(row.get("source_dataset_id") or ""),
            )


        def lookup_value(row, requested_key):
            if not requested_key:
                return None
            if requested_key in row:
                return row[requested_key]
            requested_lower = str(requested_key).lower()
            for key, value in row.items():
                if str(key).lower() == requested_lower:
                    return value
            requested_normalized = normalize_key(requested_key)
            for key, value in row.items():
                if normalize_key(key) == requested_normalized:
                    return value
            return None


        def normalize_key(value):
            return re.sub(r"[^a-zа-я0-9]+", "", str(value).lower())


        def normalize_geo(value):
            text = str(value).strip().lower()
            text = re.sub(r"^\d+\s+", "", text)
            return " ".join(text.split())


        def first_non_empty(*values):
            for value in values:
                if value is not None and str(value).strip() != "":
                    return value
            return None


        def clean_cell(value):
            if isinstance(value, str):
                return " ".join(value.split())
            return value


        def numeric_or_clean(value):
            number = to_float(value)
            if number is not None:
                return number
            return clean_cell(value)


        def to_float(value):
            if value is None or value == "":
                return None
            if isinstance(value, (int, float)):
                return float(value)
            text = str(value).strip().replace(",", ".")
            try:
                return float(text)
            except ValueError:
                return None


        def cumulative_rate(values):
            if not values:
                return None
            product = 1.0
            for value in values:
                product *= 1 + value / 100.0
            return (product - 1) * 100


        def sample_std(values):
            if len(values) < 2:
                return None
            mean = sum(values) / len(values)
            return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


        def unique_text(values):
            result = []
            seen = set()
            for value in values:
                text = str(value).strip()
                if text and text not in seen:
                    result.append(text)
                    seen.add(text)
            return result


        def write_csv(path, fieldnames, rows):
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)


        def write_json(path, payload):
            with path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
        '''
    ).strip()


def _generation_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"{CODE_GENERATION_PROMPT}\n\n"
                "JSON Schema:\n"
                f"{json.dumps(GeneratedBuildScript.model_json_schema(), ensure_ascii=False, indent=2)}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, indent=2),
        },
    ]


def _repair_messages(
    script: GeneratedBuildScript,
    context: dict[str, Any],
    attempt: ScriptExecutionAttempt,
) -> list[dict[str, str]]:
    payload = {
        "context": context,
        "previous_script": script.model_dump(mode="json"),
        "failed_attempt": attempt.model_dump(mode="json"),
    }
    return [
        {
            "role": "system",
            "content": (
                f"{REPAIR_PROMPT}\n\n"
                "JSON Schema:\n"
                f"{json.dumps(GeneratedBuildScript.model_json_schema(), ensure_ascii=False, indent=2)}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def _parse_generated_script(content: str) -> GeneratedBuildScript:
    try:
        data = _load_json_object(content)
    except (IntentParserError, json.JSONDecodeError) as exc:
        raise ScriptGeneratorError(f"LLM вернула невалидный JSON скрипта: {exc}") from exc

    try:
        script = GeneratedBuildScript.model_validate(data)
    except ValidationError as exc:
        raise ScriptGeneratorError(f"JSON скрипта не соответствует контракту: {exc}") from exc

    if "def run(" not in script.content:
        raise ScriptGeneratorError("В сгенерированном скрипте нет функции run(context, output_dir).")
    _validate_script_content(script.content)
    return script


def _validate_script_content(content: str) -> None:
    forbidden_tokens = [
        "argparse",
        "input(",
        "import requests",
        "requests.",
        "from urllib",
        "urllib.",
        "if __name__",
    ]
    for token in forbidden_tokens:
        if token in content:
            raise ScriptGeneratorError(f"Сгенерированный скрипт содержит запрещенный фрагмент: {token}")


def _repair_script(
    script: GeneratedBuildScript,
    context: dict[str, Any],
    attempt: ScriptExecutionAttempt,
    settings: LLMSettings,
) -> GeneratedBuildScript:
    content = _create_json_completion(
        settings,
        _repair_messages(script, context, attempt),
    )
    return _parse_generated_script(content)


def _execute_script(
    script: GeneratedBuildScript,
    context: dict[str, Any],
    output_dir: str | Path,
    attempt_number: int,
    timeout_seconds: int = 120,
) -> ScriptExecutionAttempt:
    with tempfile.TemporaryDirectory(prefix="ecos_build_") as directory:
        temp_dir = Path(directory)
        script_path = temp_dir / script.filename
        context_path = temp_dir / "context.json"
        script_path.write_text(script.content, encoding="utf-8")
        context_path.write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")

        try:
            completed = subprocess.run(
                [
                    _python_executable(),
                    "-c",
                    _runner_code(),
                    str(script_path),
                    str(context_path),
                    str(Path(output_dir).resolve()),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return ScriptExecutionAttempt(
                attempt=attempt_number,
                ok=False,
                stdout=exc.stdout,
                stderr=exc.stderr,
                error=f"Таймаут запуска скрипта: {timeout_seconds} секунд.",
            )

    output = _parse_stdout_json(completed.stdout)
    error = None
    if completed.returncode != 0:
        error = completed.stderr or completed.stdout or f"Код завершился с returncode={completed.returncode}."
    elif output is None:
        error = "Скрипт завершился без валидного JSON-результата."

    return ScriptExecutionAttempt(
        attempt=attempt_number,
        ok=error is None,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        output=output,
        error=error,
    )


def _runner_code() -> str:
    return (
        "import importlib.util, json, sys\n"
        "script_path, context_path, output_dir = sys.argv[1:4]\n"
        "spec = importlib.util.spec_from_file_location('generated_build_script', script_path)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "with open(context_path, encoding='utf-8') as file:\n"
        "    context = json.load(file)\n"
        "result = module.run(context, output_dir)\n"
        "print(json.dumps(result or {}, ensure_ascii=False))\n"
    )


def _python_executable() -> str:
    venv_python = Path.cwd() / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _parse_stdout_json(stdout: str | None) -> dict[str, Any] | None:
    if not stdout:
        return None
    text = stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        for line in reversed(text.splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            break
        else:
            return None
    return payload if isinstance(payload, dict) else None


def _intent_payload(intent: ResearchIntent) -> dict[str, Any]:
    return {
        "original_query": intent.original_query,
        "intent_type": intent.intent_type.value,
        "complexity": intent.complexity.value,
        "topic": intent.topic,
        "objects": intent.objects,
        "geography": intent.geography,
        "time_range": intent.time_range.model_dump(mode="json") if intent.time_range else None,
        "frequency": intent.frequency,
        "disciplinary_perspective": intent.disciplinary_perspective,
        "indicators": intent.indicators,
        "indicator_specs": [item.model_dump(mode="json") for item in intent.indicator_specs],
        "entities": intent.entities,
        "granularity": intent.granularity,
        "research_questions": intent.research_questions,
        "derived_metrics": [item.model_dump(mode="json") for item in intent.derived_metrics],
        "dataset_spec": intent.dataset_spec.model_dump(mode="json") if intent.dataset_spec else None,
        "assumptions_if_no_answer": intent.assumptions_if_no_answer,
    }


def _design_payload(design: ResearchStudyDesign) -> dict[str, Any]:
    return {
        "design_summary": design.design_summary,
        "hypotheses": [
            {
                "id": item.id,
                "statement": item.statement,
                "variables": item.variables,
                "expected_effect": item.expected_effect,
            }
            for item in design.hypotheses[:6]
        ],
        "required_measurements": [
            item.model_dump(mode="json")
            for item in design.required_measurements[:16]
        ],
        "grouping_rules": [
            item.model_dump(mode="json")
            for item in design.grouping_rules[:8]
        ],
        "derived_metrics": [
            item.model_dump(mode="json")
            for item in design.derived_metrics[:10]
        ],
        "statistical_methods": [
            item.model_dump(mode="json")
            for item in design.statistical_methods[:8]
        ],
        "visualizations": [
            item.model_dump(mode="json")
            for item in design.visualizations[:10]
        ],
        "data_quality_checks": [
            item.model_dump(mode="json")
            for item in design.data_quality_checks[:10]
        ],
        "methodology_notes": design.methodology_notes,
        "required_row_grain": design.required_row_grain,
        "blocking_reasons": design.blocking_reasons,
    }


def _source_payloads(dataset_rerank: DatasetRerankResponse | None) -> list[dict[str, Any]]:
    if not dataset_rerank:
        return []

    sources: list[dict[str, Any]] = []
    for result in dataset_rerank.results:
        sources.append(
            {
                "record_id": result.record_id,
                "dataset_id": result.dataset_id,
                "title": result.title,
                "source": result.source,
                "description": result.description,
                "tags": result.tags,
                "data_path": result.data_path,
                "resolved_data_path": _resolved_data_path(result.data_path),
                "file_format": _data_path_format(result.data_path),
                "source_url": result.source_url,
                "unit": result.unit,
                "frequency": result.frequency,
                "relevance": result.relevance.value,
                "usefulness_confidence": result.usefulness_confidence.value,
                "why_matched": result.why_matched,
                "possible_limitations": result.possible_limitations,
            }
        )
    return sources


def _available_libraries() -> dict[str, bool]:
    return {
        "pyarrow": find_spec("pyarrow") is not None,
    }


def _source_file_formats(dataset_rerank: DatasetRerankResponse | None) -> dict[str, int]:
    formats: dict[str, int] = {}
    if not dataset_rerank:
        return formats

    for result in dataset_rerank.results:
        suffix = _data_path_format(result.data_path)
        if not suffix:
            continue
        formats[suffix] = formats.get(suffix, 0) + 1
    return formats


def _data_path_format(data_path: str | None) -> str | None:
    if not data_path:
        return None
    suffixes = Path(data_path).suffixes
    if not suffixes:
        return None
    if len(suffixes) >= 2 and suffixes[-2:] == [".jsonl", ".gz"]:
        return ".jsonl.gz"
    return suffixes[-1].lower()


def _resolved_data_path(data_path: str | None) -> str | None:
    if not data_path:
        return None
    path = Path(data_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path)
