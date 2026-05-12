import csv
import json
import logging
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

logger = logging.getLogger(__name__)

SUPPORTED_SOURCE_FORMATS = frozenset({".parquet", ".csv", ".tsv", ".json", ".jsonl", ".jsonl.gz"})
SAMPLE_ROW_LIMIT = 3
SAMPLE_VALUE_MAX_LENGTH = 120


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
- source_tables[].sample_rows: несколько первых строк только для понимания формы таблицы и типов значений.
- source_tables[].column_year_map: карта "год -> имя колонки" для этой таблицы (например, {"2016": "column09", "2017": "column10"}). Используй её чтобы найти колонку для нужного года.

Правила:
- Верни только валидный JSON GeneratedBuildScript без Markdown.
- Не добавляй thinking, reasoning, пояснения, теги <think> или текст вне JSON.
- language = "sql", filename = "build_target_dataset.sql", entrypoint = "duckdb_sql".
- content должен быть одним SELECT или WITH ... SELECT.

- **КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО ИСПОЛЬЗОВАТЬ:**
  * CREATE TABLE, CREATE VIEW, CREATE OR REPLACE
  * read_parquet(), read_csv(), read_json(), read_csv_auto()
  * COPY, INSERT, UPDATE, DELETE, DROP, ALTER
  * INSTALL, LOAD, ATTACH, DETACH
  * Любые файловые пути в кавычках (например, 'C:\\...\\file.parquet')

- **Ты работаешь ТОЛЬКО с уже готовыми таблицами source_1, source_2, source_3 и т.д.**
- **Не пытайся создать или загрузить таблицы — они УЖЕ загружены в DuckDB.**

- **СТРУКТУРА ИСТОЧНИКОВ (ВНИМАТЕЛЬНО ИЗУЧИ ПЕРЕД НАПИСАНИЕМ SQL):**
  * Колонки в source_tables названы как column00, column01, column02, ...
  * Первая строка sample_rows — это ЗАГОЛОВКИ, показывающие какой год в какой колонке. Например, если sample_rows[0] = {"column04": 2002.0, "column05": 2003.0}, то:
    - column04 содержит данные за 2002 год
    - column05 содержит данные за 2003 год
  * Вторая и последующие строки sample_rows — это уже реальные данные.
  * **Используй column_year_map чтобы найти колонку для конкретного года. Например, для 2016 года проверь column_year_map: если там {"2016": "column09"}, используй column09.**
  * **Используй ТОЛЬКО те колонки, которые реально перечислены в source_tables[].columns!**
  * **Не придумывай имена колонок (column14, column15 и т.д.), если их нет в columns — это вызовет ошибку.**
  * Если в column_year_map нет нужного года (например, 2016), значит в этой таблице нет данных за этот год. Не пытайся извлечь несуществующий год.

- **ПОШАГОВЫЙ АЛГОРИЗМ НАПИСАНИЯ SQL:**
  1. Посмотри target_structure.columns — это колонки, которые должны быть в результате.
  2. Посмотри source_tables[].column_year_map — для каждого источника найди, какие годы там есть.
  3. Выбери источник(и), где есть нужный год (например, 2016).
  4. Найди в sample_rows строки, где column01 содержит "643" (код России) или "Российская Федерация".
  5. Найди в sample_rows строки, где column00 содержит "Текущие цены" (для номинального ВВП).
  6. Используй UNPIVOT или UNION ALL чтобы преобразовать широкий формат (годы в колонках) в длинный (годы в строках).
  7. Итоговый SELECT должен вернуть колонки в порядке target_structure.columns.

- **ПРИМЕР ПРАВИЛЬНОГО UNPIVOT (когда годы в колонках column04, column05, ... column13):**
  ```sql
  WITH source_1_long AS (
    SELECT
      column01 AS geo_raw,
      column00 AS price_type,
      column02 AS unit,
      '2002' AS year, column04 AS value FROM source_1 WHERE column04 IS NOT NULL
    UNION ALL
    SELECT
      column01 AS geo_raw,
      column00 AS price_type,
      column02 AS unit,
      '2003' AS year, column05 AS value FROM source_1 WHERE column05 IS NOT NULL
    -- ... и так для каждого года, который РЕАЛЬНО ЕСТЬ в колонках
  )
  SELECT ... FROM source_1_long WHERE year = '2016'

  *** НОВЫЕ ВАЖНЫЕ ПРАВИЛА ДЛЯ УСТОЙЧИВОСТИ ***

  * **ФИЛЬТРАЦИЯ ЗАГОЛОВКОВ:** Первая строка в sample_rows — это всегда заголовок с годами. Она не содержит реальных данных. Чтобы исключить её, всегда добавляй в первый CTE фильтр по ключевым колонкам. Например, если данные о странах находятся в column01 и содержат код страны, добавь: `WHERE column01 LIKE '%643%' OR column01 LIKE '%Российская%'`. Это отсечёт строку-заголовок.

  * **ФИЛЬТРАЦИЯ ПО GEO:** Никогда не используй точное сравнение `=` для географических колонок (например, `WHERE column01 = 'Россия'`). Всегда используй `LIKE '%Россия%'` или `LIKE '%Russian%'`, так как значения могут быть составными (например, "643 Российская Федерация" или "Russian Federation").

  * **ФИЛЬТРАЦИЯ ЗНАЧЕНИЙ:** Чтобы дополнительно исключить технические строки, можно добавить проверку на числовой тип значения: `AND TRY_CAST(value AS DOUBLE) IS NOT NULL`.

sample_rows используй только чтобы понять форму таблицы и типы значений. Не считай sample_rows полным набором данных.

Итоговый SELECT должен вернуть только колонки из target_structure.columns, в том же смысле и по возможности в том же порядке.

Обязательные dimension-колонки (geo, year) должны быть заполнены, иначе запрос считается неготовым.

Для lineage (source_name, source_url, source_dataset_id) используй литералы из source_tables.

Все фильтры периода, географии, объектов, частоты и показателей бери из intent, target_structure и research_design.

Не добавляй фильтры качества вроде value > 0 или IS NOT NULL для показателей, если это прямо не следует из intent, metadata или schema.

Не выдумывай отсутствующие показатели. Если source schema не содержит нужной колонки или источник дает не тот показатель, не синтезируй значения, укажи limitation.

Если нужного года нет ни в одной source_table, верни пустой SELECT по target_structure.columns с WHERE FALSE и укажи причину в possible_limitations.

Если source_datasets содержит данные только в процентах роста, не называй их уровнем индекса.

possible_limitations пиши по-русски, предметно, только из metadata/schema/context.

usage коротко объясняет, что SQL выполняется executor через DuckDB поверх source_* таблиц.

ПРОВЕРЬ СЕБЯ перед ответом:

Все ли колонки в SQL реально существуют в source_tables[].columns?

Нет ли в SQL CREATE TABLE, read_parquet, файловых путей?

Правильно ли определена колонка для 2016 года через column_year_map?

Все ли целевые колонки из target_structure.columns присутствуют в финальном SELECT?

Если какую-то колонку невозможно получить — указал ли ты это в possible_limitations?

Использовал ли ты `LIKE` вместо `=` для фильтрации географических колонок?

Добавил ли ты фильтр для исключения строки-заголовка из source_x?
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
        logger.info("Script generator: нет локальных source tables, возвращаю пустой SQL")
        return _empty_sql_script(context, ["Нет локальных source tables для SQL-сборки."])

    # Логируем информацию об источниках
    source_info = []
    for t in context['source_tables']:
        if t.get('path_exists'):
            # Показываем schema и первые 2 семпл-строки
            columns = [c['name'] for c in t.get('columns', [])]
            sample = t.get('sample_rows', [])[:2]
            source_info.append(
                f"    {t['table_name']}: {t.get('title')}\n"
                f"      Колонки: {columns}\n"
                f"      Семплы: {sample}"
            )

    logger.info(
        f"Script generator: подготовка к генерации SQL.\n"
        f"  Источников с данными: {sum(1 for t in context['source_tables'] if t.get('path_exists'))}\n"
        f"  Детали:\n" + "\n".join(source_info)
    )

    llm_settings = settings or create_llm_settings(provider=provider, model=model)
    content = _create_schema_completion(
        llm_settings,
        _generation_messages(context),
        GeneratedBuildScript,
        "GeneratedBuildScript",
    )

    logger.info(
        f"Script generator: получен ответ от LLM.\n"
        f"  Длина контента: {len(content)} символов\n"
        f"  Первые 300 символов: {content[:300]}"
    )

    return _parse_generated_script(content, context)


def generate_and_run_build_script(
        intent: ResearchIntent,
        design: ResearchStudyDesign,
        structure: TargetDatasetStructure,
        dataset_rerank: DatasetRerankResponse | None,
        settings: LLMSettings | None = None,
        provider: str | None = None,
        model: str | None = None,
        output_dir: str | Path = "artifacts/latest_run/generated_dataset",
        max_tries: int = 1,
) -> tuple[GeneratedBuildScript, BuildScriptRun]:
    context = build_generation_context(intent, design, structure, dataset_rerank)
    script = generate_build_script(
        intent=intent,
        design=design,
        structure=structure,
        dataset_rerank=dataset_rerank,
        settings=settings,
    )

    attempts: list[ScriptExecutionAttempt] = []
    attempt = _execute_script(script, context, output_dir, 1)
    attempts.append(attempt)

    if attempt.ok:
        return script, BuildScriptRun(
            status="succeeded",
            output_dir=str(Path(output_dir).resolve()),
            attempts=attempts,
            output=attempt.output,
            repaired=False,
        )

    return script, BuildScriptRun(
        status="failed",
        output_dir=str(Path(output_dir).resolve()),
        attempts=attempts,
        final_error=attempt.error or "SQL не выполнился.",
        repaired=False,
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


def _parse_generated_script(
        content: str,
        context: dict[str, Any],
) -> GeneratedBuildScript:
    try:
        script = parse_llm_json_model(
            content,
            GeneratedBuildScript,
            error_type=ScriptGeneratorError,
            context=context,
        )
        script.content = _normalized_sql(script.content)
        _validate_sql_content(script.content)
        return script
    except Exception as exc:
        logger.error(f"Script generator: ошибка парсинга SQL: {exc}")
        return _empty_sql_script(
            context,
            [f"Ошибка парсинга сгенерированного SQL: {exc}"]
        )


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
    }

    # Пробуем с json_schema
    try:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": False,
                "schema": schema_model.model_json_schema(),
            },
        }
        logger.info(f"Script generator: пробую запрос с json_schema")
        response = settings.client.chat.completions.create(**request)
        logger.info(f"Script generator: запрос с json_schema выполнен успешно")
    except BadRequestError as e:
        logger.warning(
            f"Script generator: json_schema не поддерживается.\n"
            f"  Model: {settings.model}\n"
            f"  Response: {e.response.text[:500]}"
        )
        # Пробуем с json_object
        request["response_format"] = {"type": "json_object"}
        try:
            logger.info(f"Script generator: пробую запрос с json_object")
            response = settings.client.chat.completions.create(**request)
            logger.info(f"Script generator: запрос с json_object выполнен успешно")
        except BadRequestError as e2:
            logger.warning(
                f"Script generator: json_object тоже не поддерживается.\n"
                f"  Response: {e2.response.text[:500]}"
            )
            # Без response_format
            request.pop("response_format", None)
            logger.info(f"Script generator: пробую запрос без response_format")
            response = settings.client.chat.completions.create(**request)
            logger.info(f"Script generator: запрос без response_format выполнен успешно")

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
        logger.error(f"Script generator: ошибка выполнения SQL: {exc}")
        return ScriptExecutionAttempt(
            attempt=attempt_number,
            ok=False,
            error=str(exc),
        )

    row_count = output.get("row_count", 0)
    has_readable_sources = any(table.get("path_exists") for table in context.get("source_tables", []))

    if row_count == 0 and has_readable_sources and _query_uses_readable_source(script.content, context):
        logger.warning(
            f"Script generator: SQL выполнился, но вернул 0 строк.\n"
            f"  SQL первые 200 символов: {script.content[:200]}\n"
            f"  Источники с данными: {[t['table_name'] for t in context.get('source_tables', []) if t.get('path_exists')]}"
        )
        return ScriptExecutionAttempt(
            attempt=attempt_number,
            ok=False,
            output=output,
            error="SQL выполнился, но вернул 0 строк при наличии локальных источников.",
        )

    logger.info(f"Script generator: SQL выполнен успешно, получено {row_count} строк")
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
    relation = _source_relation_sql(path, table.get("file_format"))
    con.execute(f"CREATE VIEW {name} AS SELECT * FROM {relation}")


def _validate_sql_content(content: str) -> None:
    sql = content  # уже нормализован в _parse_generated_script

    # ДЕБАГ: логируем что пришло и что получилось
    logger.info(
        f"Script generator: валидация SQL.\n"
        f"  Исходный content (первые 200): {content[:200]}\n"
        f"  Длина: {len(content) if content else 0}"
    )

    if not sql:
        raise ScriptGeneratorError("SQL не может быть пустым.")

    # Извлекаем только SQL, убирая однострочные комментарии и пустые строки в конце
    lines = []
    for line in sql.split('\n'):
        stripped = line.strip()
        # Пропускаем строки-комментарии
        if stripped.startswith('--'):
            continue
        # Оставляем только непустые строки
        if stripped:
            lines.append(line.rstrip())

    # Убираем висящий дефис в конце последней строки
    if lines:
        last_line = lines[-1]
        if last_line.rstrip().endswith('--'):
            lines = lines[:-1]

    # Убираем висящий комментарий, который мог остаться после обрезки
    while lines and lines[-1].rstrip().endswith('--'):
        lines = lines[:-1]

    sql_no_comments = '\n'.join(lines).strip()

    # Убираем запятую в конце, если она осталась одна
    if sql_no_comments.endswith(','):
        sql_no_comments = sql_no_comments[:-1].strip()

    # Убираем висящие незавершённые операторы и комментарии в конце
    sql_no_comments = sql_no_comments.rstrip(',').rstrip('--').strip()

    # Убираем пустые строки в начале и конце
    sql_no_comments = '\n'.join(
        line for line in sql_no_comments.split('\n')
        if line.strip()
    ).strip()

    # ДЕБАГ: что получилось после удаления комментариев
    logger.info(
        f"Script generator: после удаления комментариев.\n"
        f"  Первые 200: {sql_no_comments[:200] if sql_no_comments else 'ПУСТО'}\n"
        f"  Последние 200: {sql_no_comments[-200:] if sql_no_comments else 'ПУСТО'}\n"
        f"  Длина: {len(sql_no_comments)}"
    )

    lowered = sql_no_comments.lower()
    if not lowered or not (lowered.startswith("select") or lowered.startswith("with")):
        raise ScriptGeneratorError(
            f"SQL должен начинаться с SELECT или WITH. "
            f"Начинается с: {sql_no_comments[:50] if sql_no_comments else 'ПУСТО'}"
        )

    # Проверяем запрещённые конструкции только в коде, не в комментариях
    forbidden = [
        "read_parquet",
        "read_csv",
        "read_json",
        "copy ",
        "create table",
        "create view",
        "create or replace",
        "insert into",
        "update ",
        "delete from",
        "drop table",
        "drop view",
        "alter table",
        "alter view",
        "attach ",
        "install ",
        "load ",
        "pragma ",
    ]
    for token in forbidden:
        if token in lowered:
            raise ScriptGeneratorError(f"SQL содержит запрещенный фрагмент: {token.strip()}")

    logger.info("Script generator: валидация SQL пройдена успешно")


def _query_uses_readable_source(content: str, context: dict[str, Any]) -> bool:
    lowered = _normalized_sql(content).lower()
    return any(
        table.get("path_exists")
        and re.search(rf"\b{re.escape(str(table.get('table_name', '')).lower())}\b", lowered)
        for table in context.get("source_tables", [])
    )


def _normalized_sql(content: str) -> str:
    text = content.strip()

    # Удаляем markdown-блоки
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # ДЕБАГ
    logger.info(f"Script generator: _normalized_sql после fenced (первые 200): {text[:200]}")

    # Ищем начало SQL: WITH или SELECT
    match = re.search(r"\b(with|select)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start():]

    # ДЕБАГ
    logger.info(f"Script generator: _normalized_sql после поиска WITH/SELECT (первые 200): {text[:200]}")

    # Удаляем завершающую точку с запятой и обрезаем пробелы
    text = text.strip()
    text = re.sub(r'\s*;\s*$', '', text)

    # Нормализуем переносы строк
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # ДЕБАГ
    logger.info(f"Script generator: _normalized_sql финал (первые 200): {text[:200]}")

    return text


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
        if not resolved_path or file_format not in SUPPORTED_SOURCE_FORMATS:
            continue
        path = Path(resolved_path)
        table = {
            "table_name": f"source_{index}",
            "record_id": source.get("record_id"),
            "dataset_id": source.get("dataset_id"),
            "source_dataset_id": source.get("dataset_id"),
            "title": source.get("title"),
            "source": source.get("source"),
            "source_name": source.get("title") or source.get("source"),
            "source_url": source.get("source_url"),
            "unit": source.get("unit"),
            "frequency": source.get("frequency"),
            "resolved_path": str(path),
            "file_format": file_format,
            "path_exists": path.exists(),
            "columns": _inspect_schema(path, file_format) if path.exists() else [],
            "sample_rows": _sample_rows(path, file_format)[:SAMPLE_ROW_LIMIT] if path.exists() else [],
            "possible_limitations": source.get("possible_limitations") or [],
        }
        tables.append(table)
    return tables


def _inspect_schema(path: Path, file_format: str | None) -> list[dict[str, str | None]]:
    if find_spec("duckdb") is not None:
        try:
            import duckdb

            relation = _source_relation_sql(_sql_string(str(path)), file_format)
            con = duckdb.connect(database=":memory:")
            try:
                rows = con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
            finally:
                con.close()
            return [{"name": row[0], "dtype": row[1]} for row in rows]
        except Exception:
            pass
    if file_format in {".csv", ".tsv"}:
        delimiter = "\t" if file_format == ".tsv" else ","
        try:
            with path.open(newline="", encoding="utf-8-sig") as file:
                reader = csv.reader(file, delimiter=delimiter)
                return [{"name": name, "dtype": None} for name in next(reader, [])]
        except Exception:
            return []
    return []


def _sample_rows(path: Path, file_format: str | None) -> list[dict[str, Any]]:
    if find_spec("duckdb") is None:
        return []
    try:
        import duckdb

        relation = _source_relation_sql(_sql_string(str(path)), file_format)
        con = duckdb.connect(database=":memory:")
        try:
            cursor = con.execute(f"SELECT * FROM {relation} LIMIT {SAMPLE_ROW_LIMIT}")
            columns = [item[0] for item in cursor.description or []]
            rows = cursor.fetchall()
        finally:
            con.close()
    except Exception:
        return []
    return [
        {name: _sample_value(value) for name, value in zip(columns, row)}
        for row in rows
    ]


def _sample_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    text = str(value).strip()
    if len(text) > SAMPLE_VALUE_MAX_LENGTH:
        return text[:SAMPLE_VALUE_MAX_LENGTH]
    return text


def _source_relation_sql(path: str, file_format: str | None) -> str:
    if file_format == ".parquet":
        return f"read_parquet({path})"
    if file_format == ".csv":
        return f"read_csv_auto({path}, header=true)"
    if file_format == ".tsv":
        return f"read_csv_auto({path}, delim='\\t', header=true)"
    if file_format in {".json", ".jsonl", ".jsonl.gz"}:
        return f"read_json_auto({path})"
    raise ScriptGeneratorError(f"Формат источника не поддержан SQL executor: {file_format}")


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
            values.append(
                f"В metadata источника {source.get('title') or source.get('record_id')} нет локального data_path.")
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