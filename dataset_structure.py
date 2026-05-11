import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intent_parser import ResearchIntent, none_to_empty_list
from research_designer import ResearchStudyDesign


CYRILLIC_TRANSLITERATION = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "y",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "sch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
)


class DatasetColumn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    title: str
    role: str = Field(description="dimension, indicator, derived_metric, metadata, quality.")
    dtype: str
    unit: str | None = None
    nullable: bool = True
    description: str | None = None
    source_field: str | None = None


class TargetDatasetStructure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_grain: str
    primary_key: list[str] = Field(default_factory=list)
    columns: list[DatasetColumn] = Field(default_factory=list)
    expected_frequency: str | None = None
    time_range: str | None = None
    geography: list[str] = Field(default_factory=list)
    design_notes: list[str] = Field(default_factory=list)

    @field_validator("primary_key", "columns", "geography", "design_notes", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


def build_target_dataset_structure(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
) -> TargetDatasetStructure:
    row_grain = _infer_row_grain(intent, design)
    columns: list[DatasetColumn] = []
    primary_key: list[str] = []

    for column in _dimension_columns(intent, row_grain):
        _append_unique_column(columns, column)
        if not column.nullable:
            primary_key.append(column.name)

    for index, column in enumerate(_indicator_columns(intent, design), start=1):
        fallback_name = f"indicator_{index}"
        _append_unique_column(columns, _with_fallback_name(column, fallback_name))

    for index, column in enumerate(_derived_metric_columns(intent, design), start=1):
        fallback_name = f"derived_metric_{index}"
        _append_unique_column(columns, _with_fallback_name(column, fallback_name))

    for column in _metadata_columns():
        _append_unique_column(columns, column)

    if not primary_key:
        primary_key = _fallback_primary_key(row_grain, columns)

    return TargetDatasetStructure(
        row_grain=row_grain,
        primary_key=primary_key,
        columns=columns,
        expected_frequency=intent.frequency or (intent.dataset_spec.frequency if intent.dataset_spec else None),
        time_range=_time_range_label(intent),
        geography=intent.geography,
        design_notes=_design_notes(intent, design, row_grain),
    )


def column_names(structure: TargetDatasetStructure) -> list[str]:
    return [column.name for column in structure.columns]


def slugify(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    text = value.strip().lower().translate(CYRILLIC_TRANSLITERATION)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text or not re.search(r"[a-z]", text):
        return fallback
    if text[0].isdigit():
        text = f"v_{text}"
    return text[:64]


def _infer_row_grain(intent: ResearchIntent, design: ResearchStudyDesign) -> str:
    if intent.dataset_spec and intent.dataset_spec.row_grain:
        return intent.dataset_spec.row_grain
    if intent.granularity:
        return intent.granularity
    if design.required_row_grain:
        return design.required_row_grain[0]

    has_geo = bool(intent.geography or intent.objects or intent.entities)
    has_time = bool(intent.time_range or intent.frequency)
    if has_geo and has_time:
        return "география-период"
    if has_geo:
        return "география"
    if has_time:
        return "период"
    return "наблюдение"


def _dimension_columns(intent: ResearchIntent, row_grain: str) -> list[DatasetColumn]:
    grain = row_grain.lower()
    columns: list[DatasetColumn] = []
    needs_geo = bool(intent.geography) or any(
        token in grain
        for token in ("географ", "страна", "регион", "субъект", "территор")
    )
    if needs_geo:
        columns.append(
            DatasetColumn(
                name="geo",
                title="География",
                role="dimension",
                dtype="string",
                nullable=False,
                description="Страна, регион или другая территория наблюдения.",
            )
        )

    frequency = (intent.frequency or "").lower()
    if "год" in grain or "year" in grain or "год" in frequency:
        columns.append(
            DatasetColumn(
                name="year",
                title="Год",
                role="dimension",
                dtype="integer",
                nullable=False,
                description="Год наблюдения.",
            )
        )
    elif any(token in grain or token in frequency for token in ("месяц", "квартал", "период")):
        columns.append(
            DatasetColumn(
                name="period",
                title="Период",
                role="dimension",
                dtype="string",
                nullable=False,
                description="Период наблюдения в формате, соответствующем частоте данных.",
            )
        )

    if "продукт" in grain or "товар" in grain:
        columns.append(
            DatasetColumn(
                name="product",
                title="Товар или продукт",
                role="dimension",
                dtype="string",
                nullable=False,
                description="Товарная или продуктовая группа наблюдения.",
            )
        )
    if "отрасл" in grain or "вид деятельности" in grain:
        columns.append(
            DatasetColumn(
                name="sector",
                title="Отрасль",
                role="dimension",
                dtype="string",
                nullable=False,
                description="Отрасль или вид экономической деятельности.",
            )
        )

    return columns


def _indicator_columns(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
) -> list[DatasetColumn]:
    columns: list[DatasetColumn] = []
    specs_by_name = {spec.name.lower(): spec for spec in intent.indicator_specs}
    names = list(dict.fromkeys([*intent.indicators, *[measurement.name for measurement in design.required_measurements]]))

    for index, name in enumerate(names, start=1):
        spec = specs_by_name.get(name.lower())
        measurement = _find_measurement(design, name)
        role = measurement.role if measurement else (spec.role if spec and spec.role else "primary")
        if role in {"grouping", "denominator"}:
            role = "indicator"
        else:
            role = "indicator"
        unit = (measurement.unit if measurement else None) or (spec.unit if spec else None)
        definition = (measurement.definition if measurement else None) or (spec.definition if spec else None)
        columns.append(
            DatasetColumn(
                name=slugify(name, f"indicator_{index}"),
                title=name,
                role=role,
                dtype="number",
                unit=unit,
                nullable=True,
                description=definition or f"Показатель: {name}.",
                source_field=name,
            )
        )

    return columns


def _derived_metric_columns(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
) -> list[DatasetColumn]:
    columns: list[DatasetColumn] = []
    names_seen: set[str] = set()

    for index, metric in enumerate(design.derived_metrics, start=1):
        names_seen.add(metric.name.lower())
        columns.append(
            DatasetColumn(
                name=slugify(metric.name, f"derived_metric_{index}"),
                title=metric.name,
                role="derived_metric",
                dtype="number",
                unit=metric.unit,
                nullable=True,
                description=f"{metric.formula}. {metric.interpretation}",
                source_field=", ".join(metric.inputs) if metric.inputs else None,
            )
        )

    offset = len(columns)
    for index, metric in enumerate(intent.derived_metrics, start=1):
        if metric.name.lower() in names_seen:
            continue
        columns.append(
            DatasetColumn(
                name=slugify(metric.name, f"derived_metric_{offset + index}"),
                title=metric.name,
                role="derived_metric",
                dtype="number",
                unit=None,
                nullable=True,
                description=metric.formula or metric.normalization,
                source_field=", ".join(metric.inputs) if metric.inputs else None,
            )
        )

    return columns


def _metadata_columns() -> list[DatasetColumn]:
    return [
        DatasetColumn(
            name="source_name",
            title="Источник",
            role="metadata",
            dtype="string",
            nullable=True,
            description="Название источника данных.",
        ),
        DatasetColumn(
            name="source_url",
            title="Ссылка на источник",
            role="metadata",
            dtype="string",
            nullable=True,
            description="URL исходного набора или страницы показателя.",
        ),
        DatasetColumn(
            name="source_dataset_id",
            title="Идентификатор исходного датасета",
            role="metadata",
            dtype="string",
            nullable=True,
            description="Идентификатор набора-предшественника в реестре или внешней системе.",
        ),
        DatasetColumn(
            name="downloaded_at",
            title="Дата сборки",
            role="metadata",
            dtype="datetime",
            nullable=True,
            description="Дата и время сборки итогового датасета.",
        ),
    ]


def _append_unique_column(columns: list[DatasetColumn], column: DatasetColumn) -> None:
    names = {item.name for item in columns}
    original = column.name
    if original not in names:
        columns.append(column)
        return

    suffix = 2
    while f"{original}_{suffix}" in names:
        suffix += 1
    payload = column.model_dump()
    payload["name"] = f"{original}_{suffix}"
    columns.append(DatasetColumn.model_validate(payload))


def _with_fallback_name(column: DatasetColumn, fallback: str) -> DatasetColumn:
    if column.name:
        return column
    payload = column.model_dump()
    payload["name"] = fallback
    return DatasetColumn.model_validate(payload)


def _find_measurement(design: ResearchStudyDesign, name: str):
    normalized = name.lower()
    for measurement in design.required_measurements:
        if measurement.name.lower() == normalized:
            return measurement
    return None


def _fallback_primary_key(row_grain: str, columns: list[DatasetColumn]) -> list[str]:
    dimension_columns = [column.name for column in columns if column.role == "dimension"]
    if dimension_columns:
        return dimension_columns
    return [slugify(row_grain, "observation_id")]


def _time_range_label(intent: ResearchIntent) -> str | None:
    if intent.time_range is None:
        return None
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
    return None


def _design_notes(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
    row_grain: str,
) -> list[str]:
    notes = [f"Зернистость строк: {row_grain}."]
    if intent.dataset_spec and intent.dataset_spec.columns:
        notes.append(
            "Исходный список ожидаемых колонок из ResearchIntent: "
            + ", ".join(intent.dataset_spec.columns)
            + "."
        )
    if design.methodology_notes:
        notes.append(design.methodology_notes)
    if intent.assumptions_if_no_answer:
        notes.append(
            "Допущения при отсутствии ответа пользователя: "
            + "; ".join(intent.assumptions_if_no_answer)
            + "."
        )
    return notes
