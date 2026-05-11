import csv
import json
import re
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

from openai import BadRequestError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dataset_reranker import DatasetRerankResponse
from dataset_structure import TargetDatasetStructure
from intent_parser import (
    LLMSettings,
    ResearchIntent,
    create_llm_settings,
    none_to_empty_list,
    parse_llm_json_model,
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

    filename: str = "build_target_dataset.sql"
    language: Literal["sql"] = "sql"
    entrypoint: str = "duckdb_sql"
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


SQL_GENERATION_PROMPT = """Ты генерируешь DuckDB SQL для сборки целевого датасета ECOS AI Assistant.

На входе JSON-контекст:
- intent: формальное описание запроса пользователя;
- research_design: сжатый дизайн исследования;
- target_structure: целевые колонки итогового CSV;
- source_tables: локальные файлы, уже подключаемые executor как SQL-таблицы;
- source_datasets: metadata и lineage найденных RAG источников;
- execution_contract: правила SQL-запуска.

Правила:
- Верни только валидный JSON GeneratedBuildScript без Markdown.
- Не добавляй thinking, reasoning, пояснения, теги <think> или текст вне JSON.
- language = "sql", filename = "build_target_dataset.sql", entrypoint = "duckdb_sql".
- content должен быть одним DuckDB SELECT или WITH ... SELECT.
- Не пиши Python, CLI, DDL, DML, COPY, CREATE, INSERT, UPDATE, DELETE, DROP, ALTER, INSTALL, LOAD.
- Не вызывай read_parquet/read_csv/read_json и не указывай файловые пути. Файлы уже доступны как таблицы source_1, source_2 и так далее.
- Используй только таблицы и колонки из source_tables.
- Итоговый SELECT должен вернуть только колонки из target_structure.columns, в том же смысле и по возможности в том же порядке.
- Обязательные dimension-колонки, например geo/year, должны быть заполнены, иначе запрос считается неготовым.
- Для lineage используй литералы из source_tables: source_name, source_url, source_dataset_id.
- Все фильтры периода, географии, объектов, частоты и показателей бери из intent, target_structure и research_design.
- Не выдумывай отсутствующие показатели. Если source schema не содержит нужной колонки или источник дает не тот показатель, не синтезируй значения, укажи limitation.
- Если source_datasets содержит данные только в процентах роста, не называй их уровнем индекса.
- possible_limitations пиши по-русски, предметно, только из metadata/schema/context.
- usage коротко объясняет, что SQL выполняется executor через DuckDB поверх source_* таблиц.
"""


SQL_REPAIR_PROMPT = """Ты исправляешь DuckDB SQL для сборки целевого датасета.

На входе:
- исходный JSON-контекст;
- предыдущий GeneratedBuildScript;
- ошибка выполнения.

Правила:
- Верни полный валидный JSON GeneratedBuildScript без Markdown.
- Не добавляй thinking, reasoning, пояснения, теги <think> или текст вне JSON.
- Сохрани смысл запроса, target_structure и source_tables.
- Исправляй только причину ошибки или пустого результата.
- content должен быть одним SELECT или WITH ... SELECT.
- Не используй Python, CLI, DDL, DML, COPY, CREATE, INSERT, UPDATE, DELETE, DROP, ALTER, INSTALL, LOAD.
- Не читай файлы в SQL. Используй только source_* таблицы и их колонки.
- Не выдумывай отсутствующие значения.
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
    if not context["source_tables"]:
        return _empty_sql_script(context, ["Нет локальных source tables для SQL-сборки."])

    llm_settings = settings or create_llm_settings(provider=provider, model=model)
    content = _create_schema_completion(
        llm_settings,
        _generation_messages(context),
        GeneratedBuildScript,
        "GeneratedBuildScript",
    )
    return _parse_generated_script(content, llm_settings, context)


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

    final_error = attempts[-1].error if attempts else "SQL не запускался."
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
    source_datasets = _source_payloads(dataset_rerank)
    source_tables = _source_table_payloads(source_datasets)
    return {
        "project_root": str(Path.cwd()),
        "intent": _intent_payload(intent),
        "research_design": _design_payload(design),
        "target_structure": structure.model_dump(mode="json"),
        "source_datasets": source_datasets,
        "source_tables": source_tables,
        "execution_contract": {
            "engine": "duckdb",
            "entrypoint": "duckdb_sql",
            "max_tries": 3,
            "available_libraries": _available_libraries(),
            "default_outputs": [
                "target_dataset.csv",
                "target_dataset.metadata.json",
                "source_manifest.json",
                "executed_query.sql",
            ],
        },
    }


def _generation_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"{SQL_GENERATION_PROMPT}\n\n"
                "JSON Schema:\n"
                f"{json.dumps(GeneratedBuildScript.model_json_schema(), ensure_ascii=False, indent=2)}"
            ),
        },
        {"role": "user", "content": json.dumps(context, ensure_ascii=False, indent=2)},
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
                f"{SQL_REPAIR_PROMPT}\n\n"
                "JSON Schema:\n"
                f"{json.dumps(GeneratedBuildScript.model_json_schema(), ensure_ascii=False, indent=2)}"
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def _parse_generated_script(
    content: str,
    settings: LLMSettings,
    context: dict[str, Any],
) -> GeneratedBuildScript:
    script = parse_llm_json_model(
        content,
        GeneratedBuildScript,
        settings=settings,
        error_type=ScriptGeneratorError,
        context=context,
    )
    script.content = _normalized_sql(script.content)
    _validate_sql_content(script.content)
    return script


def _repair_script(
    script: GeneratedBuildScript,
    context: dict[str, Any],
    attempt: ScriptExecutionAttempt,
    settings: LLMSettings,
) -> GeneratedBuildScript:
    content = _create_schema_completion(
        settings,
        _repair_messages(script, context, attempt),
        GeneratedBuildScript,
        "GeneratedBuildScript",
    )
    return _parse_generated_script(content, settings, context)


def _create_schema_completion(
    settings: LLMSettings,
    messages: list[dict[str, str]],
    schema_model: type[BaseModel],
    schema_name: str,
) -> str:
    if not settings.model:
        raise ScriptGeneratorError("LLM model is not configured.")

    request: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema_model.model_json_schema(),
            },
        },
    }

    try:
        response = settings.client.chat.completions.create(**request)
    except BadRequestError:
        request["response_format"] = {"type": "json_object"}
        try:
            response = settings.client.chat.completions.create(**request)
        except BadRequestError:
            request.pop("response_format", None)
            response = settings.client.chat.completions.create(**request)

    content = response.choices[0].message.content
    if not content:
        raise ScriptGeneratorError("LLM вернула пустой SQL JSON.")
    return content


def _execute_script(
    script: GeneratedBuildScript,
    context: dict[str, Any],
    output_dir: str | Path,
    attempt_number: int,
) -> ScriptExecutionAttempt:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        output = _run_duckdb_sql(script, context, output_path)
    except Exception as exc:
        return ScriptExecutionAttempt(
            attempt=attempt_number,
            ok=False,
            error=str(exc),
        )

    row_count = output.get("row_count", 0)
    has_readable_sources = any(table.get("path_exists") for table in context.get("source_tables", []))
    if row_count == 0 and has_readable_sources:
        return ScriptExecutionAttempt(
            attempt=attempt_number,
            ok=False,
            output=output,
            error="SQL выполнился, но вернул 0 строк при наличии локальных источников.",
        )

    return ScriptExecutionAttempt(
        attempt=attempt_number,
        ok=True,
        output=output,
    )


def _run_duckdb_sql(
    script: GeneratedBuildScript,
    context: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    if find_spec("duckdb") is None:
        raise ScriptGeneratorError("Библиотека duckdb не установлена.")

    import duckdb

    _validate_sql_content(script.content)
    query = _normalized_sql(script.content)
    target_columns = context["target_structure"].get("columns") or []
    target_names = [column["name"] for column in target_columns]
    required_names = [
        column["name"]
        for column in target_columns
        if column.get("role") == "dimension" and not column.get("nullable", True)
    ]

    con = duckdb.connect(database=":memory:")
    try:
        for table in context.get("source_tables", []):
            if table.get("path_exists"):
                _create_source_view(con, table)

        cursor = con.execute(query)
        result_columns = [item[0] for item in cursor.description or []]
        missing_required = [name for name in required_names if name not in result_columns]
        if missing_required:
            raise ScriptGeneratorError(
                "SQL не вернул обязательные колонки: " + ", ".join(missing_required)
            )

        raw_rows = cursor.fetchall()
    finally:
        con.close()

    rows = [
        _align_row(dict(zip(result_columns, row)), target_names)
        for row in raw_rows
    ]

    dataset_path = output_path / "target_dataset.csv"
    metadata_path = output_path / "target_dataset.metadata.json"
    manifest_path = output_path / "source_manifest.json"
    sql_path = output_path / "executed_query.sql"
    chart_path = output_path / "chart_data.csv"

    sql_path.write_text(query + "\n", encoding="utf-8")
    _write_csv(dataset_path, target_names, rows)
    source_manifest = _source_manifest(context)
    _write_json(manifest_path, source_manifest)
    chart_rows = _chart_rows(rows, target_columns, context)
    _write_csv(chart_path, ["visualization_id", "title", "geo", "year", "metric", "value"], chart_rows)

    limitations = _run_limitations(script, context)
    coverage = _coverage(rows, context)
    _write_json(
        metadata_path,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(rows),
            "coverage": coverage,
            "possible_limitations": limitations,
            "target_structure": context["target_structure"],
            "source_manifest": source_manifest,
            "executed_sql": str(sql_path),
            "outputs": {
                "dataset": str(dataset_path),
                "metadata": str(metadata_path),
                "source_manifest": str(manifest_path),
                "sql": str(sql_path),
                "chart_data": str(chart_path),
            },
        },
    )

    return {
        "dataset": str(dataset_path),
        "metadata": str(metadata_path),
        "source_manifest": str(manifest_path),
        "sql": str(sql_path),
        "chart_data": str(chart_path),
        "row_count": len(rows),
        "coverage": coverage,
        "limitations": limitations,
    }


def _create_source_view(con: Any, table: dict[str, Any]) -> None:
    path = _sql_string(table["resolved_path"])
    name = _sql_identifier(table["table_name"])
    file_format = table.get("file_format")
    if file_format == ".parquet":
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet({path})")
        return
    if file_format in {".csv", ".tsv"}:
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto({path}, header=true)")
        return
    if file_format in {".json", ".jsonl", ".jsonl.gz"}:
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_json_auto({path})")
        return
    raise ScriptGeneratorError(f"Формат источника не поддержан SQL executor: {file_format}")


def _validate_sql_content(content: str) -> None:
    sql = _normalized_sql(content)
    lowered = sql.lower()
    if not (lowered.startswith("select ") or lowered.startswith("with ")):
        raise ScriptGeneratorError("SQL должен начинаться с SELECT или WITH.")
    if ";" in sql.rstrip(";"):
        raise ScriptGeneratorError("SQL должен содержать только один SELECT.")
    forbidden = [
        "read_parquet",
        "read_csv",
        "read_json",
        "copy ",
        "create ",
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "attach ",
        "install ",
        "load ",
        "pragma ",
    ]
    for token in forbidden:
        if token in lowered:
            raise ScriptGeneratorError(f"SQL содержит запрещенный фрагмент: {token.strip()}")


def _normalized_sql(content: str) -> str:
    text = content.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    match = re.search(r"\b(with|select)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start():]
    return text.strip().rstrip(";").strip()


def _empty_sql_script(context: dict[str, Any], limitations: list[str]) -> GeneratedBuildScript:
    target_columns = context["target_structure"].get("columns") or []
    select_items = [
        f"CAST(NULL AS VARCHAR) AS {_sql_identifier(column['name'])}"
        for column in target_columns
    ]
    sql = "SELECT " + ", ".join(select_items) + " WHERE FALSE"
    return GeneratedBuildScript(
        content=sql,
        usage="SQL выполняется executor через DuckDB. Локальных source tables нет.",
        inputs=[],
        outputs=_default_outputs(),
        possible_limitations=limitations,
    )


def _default_outputs() -> list[GeneratedOutputSpec]:
    return [
        GeneratedOutputSpec(name="target_dataset.csv", kind="dataset", path="target_dataset.csv"),
        GeneratedOutputSpec(name="target_dataset.metadata.json", kind="metadata", path="target_dataset.metadata.json"),
        GeneratedOutputSpec(name="source_manifest.json", kind="manifest", path="source_manifest.json"),
        GeneratedOutputSpec(name="executed_query.sql", kind="sql", path="executed_query.sql"),
        GeneratedOutputSpec(name="chart_data.csv", kind="table", path="chart_data.csv"),
    ]


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


def _source_table_payloads(source_datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for index, source in enumerate(source_datasets, start=1):
        resolved_path = source.get("resolved_data_path")
        file_format = source.get("file_format")
        if not resolved_path or file_format not in {".parquet", ".csv", ".tsv", ".json", ".jsonl", ".jsonl.gz"}:
            continue
        path = Path(resolved_path)
        table = {
            "table_name": f"source_{index}",
            "record_id": source.get("record_id"),
            "dataset_id": source.get("dataset_id"),
            "title": source.get("title"),
            "source": source.get("source"),
            "source_name": source.get("title") or source.get("source"),
            "source_url": source.get("source_url"),
            "resolved_path": str(path),
            "file_format": file_format,
            "path_exists": path.exists(),
            "columns": _inspect_schema(path, file_format) if path.exists() else [],
            "possible_limitations": source.get("possible_limitations") or [],
        }
        tables.append(table)
    return tables


def _inspect_schema(path: Path, file_format: str | None) -> list[dict[str, str | None]]:
    if file_format == ".parquet":
        try:
            import pyarrow.parquet as pq

            schema = pq.ParquetFile(path).schema_arrow
            return [
                {"name": field.name, "dtype": str(field.type)}
                for field in schema
            ]
        except Exception:
            return []
    if file_format in {".csv", ".tsv"}:
        delimiter = "\t" if file_format == ".tsv" else ","
        try:
            with path.open(newline="", encoding="utf-8-sig") as file:
                reader = csv.reader(file, delimiter=delimiter)
                return [{"name": name, "dtype": None} for name in next(reader, [])]
        except Exception:
            return []
    return []


def _available_libraries() -> dict[str, bool]:
    return {
        "duckdb": find_spec("duckdb") is not None,
        "pyarrow": find_spec("pyarrow") is not None,
        "pandas": find_spec("pandas") is not None,
    }


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


def _align_row(row: dict[str, Any], target_names: list[str]) -> dict[str, Any]:
    return {name: row.get(name) for name in target_names}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def _source_manifest(context: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "table_name": table.get("table_name"),
            "record_id": table.get("record_id"),
            "dataset_id": table.get("dataset_id"),
            "title": table.get("title"),
            "source": table.get("source"),
            "data_path": table.get("resolved_path"),
            "source_url": table.get("source_url"),
            "file_format": table.get("file_format"),
            "columns": table.get("columns"),
            "possible_limitations": table.get("possible_limitations") or [],
        }
        for table in context.get("source_tables", [])
    ]


def _run_limitations(script: GeneratedBuildScript, context: dict[str, Any]) -> list[str]:
    values: list[str] = []
    values.extend(script.possible_limitations)
    for source in context.get("source_datasets", []):
        values.extend(source.get("possible_limitations") or [])
        if source.get("data_path") and not source.get("resolved_data_path"):
            values.append(f"Не удалось разрешить data_path источника {source.get('title') or source.get('record_id')}.")
        if not source.get("data_path"):
            values.append(f"В metadata источника {source.get('title') or source.get('record_id')} нет локального data_path.")
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _coverage(rows: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    requested_years = _target_years(context)
    available_years = sorted(
        {
            int(row["year"])
            for row in rows
            if row.get("year") is not None and str(row.get("year")).isdigit()
        }
    )
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


def _target_years(context: dict[str, Any]) -> list[int]:
    time_range = context.get("target_structure", {}).get("time_range")
    years = [int(item) for item in re.findall(r"\d{4}", str(time_range or ""))]
    if len(years) >= 2:
        return list(range(min(years), max(years) + 1))
    return years


def _chart_rows(
    rows: list[dict[str, Any]],
    columns: list[dict[str, Any]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    visualizations = context.get("research_design", {}).get("visualizations") or []
    metric_columns = [
        column["name"]
        for column in columns
        if column.get("role") in {"indicator", "derived_metric"} and column.get("name")
    ]
    if not visualizations or not metric_columns:
        return []
    output: list[dict[str, Any]] = []
    for visualization in visualizations:
        metric = _resolve_metric_column(visualization.get("y_axis"), metric_columns)
        if not metric:
            metric = metric_columns[0]
        for row in rows:
            value = row.get(metric)
            if value in (None, ""):
                continue
            output.append(
                {
                    "visualization_id": visualization.get("id"),
                    "title": visualization.get("title"),
                    "geo": row.get("geo"),
                    "year": row.get("year"),
                    "metric": metric,
                    "value": value,
                }
            )
    return output


def _resolve_metric_column(metric: str | None, metric_columns: list[str]) -> str | None:
    if not metric:
        return None
    key = _normalize_key(metric)
    for column in metric_columns:
        if _normalize_key(column) == key or key in _normalize_key(column):
            return column
    return None


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", str(value).lower())


def _sql_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _sql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"
