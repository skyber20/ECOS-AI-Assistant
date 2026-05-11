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
    role: str
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
    row_grain = _row_grain(intent, design)
    columns: list[DatasetColumn] = []
    primary_key: list[str] = []

    for column in _dimension_columns(intent, row_grain):
        _append_column(columns, column)
        if not column.nullable:
            primary_key.append(column.name)

    for index, column in enumerate(_measurement_columns(intent, design), start=1):
        _append_column(columns, _with_fallback_name(column, f"indicator_{index}"))

    for index, column in enumerate(_derived_columns(intent, design), start=1):
        _append_column(columns, _with_fallback_name(column, f"derived_{index}"))

    for column in _metadata_columns():
        _append_column(columns, column)

    if not primary_key:
        primary_key = [column.name for column in columns if column.role == "dimension"]

    return TargetDatasetStructure(
        row_grain=row_grain,
        primary_key=primary_key,
        columns=columns,
        expected_frequency=intent.frequency or (intent.dataset_spec.frequency if intent.dataset_spec else None),
        time_range=_time_range(intent),
        geography=list(dict.fromkeys([*intent.geography, *intent.objects, *intent.entities])),
        design_notes=_design_notes(intent, design, row_grain),
    )


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


def _row_grain(intent: ResearchIntent, design: ResearchStudyDesign) -> str:
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
    columns: list[DatasetColumn] = []
    grain = row_grain.lower()
    frequency = (intent.frequency or "").lower()

    if intent.geography or intent.objects or intent.entities or _contains(grain, "географ", "страна", "регион", "территор"):
        columns.append(
            DatasetColumn(
                name="geo",
                title="География",
                role="dimension",
                dtype="string",
                nullable=False,
                description="Территория или объект наблюдения.",
            )
        )

    if _contains(frequency, "месяц", "квартал") or _contains(grain, "месяц", "квартал"):
        columns.append(
            DatasetColumn(
                name="period",
                title="Период",
                role="dimension",
                dtype="string",
                nullable=False,
                description="Период наблюдения.",
            )
        )
    elif intent.time_range or _contains(grain, "год", "year") or "год" in frequency:
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
    elif intent.frequency or "период" in grain:
        columns.append(
            DatasetColumn(
                name="period",
                title="Период",
                role="dimension",
                dtype="string",
                nullable=False,
                description="Период наблюдения.",
            )
        )

    return columns


def _measurement_columns(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
) -> list[DatasetColumn]:
    specs = {spec.name.lower(): spec for spec in intent.indicator_specs}
    measurement_names = [
        measurement.name
        for measurement in design.required_measurements
        if _is_measurement_indicator(measurement)
    ]
    names = _unique(measurement_names or [
        *intent.indicators,
        *[spec.name for spec in intent.indicator_specs],
    ])
    columns: list[DatasetColumn] = []

    for index, name in enumerate(names, start=1):
        if _is_dimension_or_metadata_name(name):
            continue
        spec = specs.get(name.lower())
        measurement = _measurement(design, name)
        columns.append(
            DatasetColumn(
                name=slugify(name, f"indicator_{index}"),
                title=name,
                role="indicator",
                dtype="number",
                unit=(measurement.unit if measurement else None) or (spec.unit if spec else None),
                nullable=True,
                description=(measurement.definition if measurement else None) or (spec.definition if spec else None),
                source_field=name,
            )
        )

    return columns


def _derived_columns(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
) -> list[DatasetColumn]:
    columns: list[DatasetColumn] = []
    seen: set[str] = set()

    for metric in design.derived_metrics:
        if _is_dimension_or_metadata_name(metric.name):
            continue
        seen.add(metric.name.lower())
        columns.append(
            DatasetColumn(
                name=slugify(metric.name, f"derived_{len(columns) + 1}"),
                title=metric.name,
                role="derived_metric",
                dtype="number",
                unit=metric.unit,
                nullable=True,
                description=metric.formula_explanation or metric.formula,
                source_field=", ".join(metric.inputs) if metric.inputs else None,
            )
        )

    for metric in intent.derived_metrics:
        if metric.name.lower() in seen or _is_dimension_or_metadata_name(metric.name):
            continue
        columns.append(
            DatasetColumn(
                name=slugify(metric.name, f"derived_{len(columns) + 1}"),
                title=metric.name,
                role="derived_metric",
                dtype="number",
                nullable=True,
                description=metric.formula or metric.normalization,
                source_field=", ".join(metric.inputs) if metric.inputs else None,
            )
        )

    return columns


def _metadata_columns() -> list[DatasetColumn]:
    return [
        DatasetColumn(name="source_name", title="Источник", role="metadata", dtype="string"),
        DatasetColumn(name="source_url", title="Ссылка на источник", role="metadata", dtype="string"),
        DatasetColumn(name="source_dataset_id", title="Идентификатор исходного датасета", role="metadata", dtype="string"),
        DatasetColumn(name="downloaded_at", title="Дата сборки", role="metadata", dtype="datetime"),
    ]


def _design_notes(intent: ResearchIntent, design: ResearchStudyDesign, row_grain: str) -> list[str]:
    notes: list[str] = [f"Зернистость строк: {row_grain}."]
    if intent.dataset_spec and intent.dataset_spec.columns:
        notes.append("Ожидаемые колонки из запроса: " + ", ".join(intent.dataset_spec.columns))
    if design.methodology_notes:
        notes.append(design.methodology_notes)
    if intent.assumptions_if_no_answer:
        notes.append("Допущения: " + "; ".join(intent.assumptions_if_no_answer))
    notes.append("Физическая схема исходных файлов определяется только исполняемым скриптом по data_path.")
    return notes


def _append_column(columns: list[DatasetColumn], column: DatasetColumn) -> None:
    existing = {item.name for item in columns}
    if column.name not in existing:
        columns.append(column)
        return
    base = column.name
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    payload = column.model_dump()
    payload["name"] = f"{base}_{index}"
    columns.append(DatasetColumn.model_validate(payload))


def _with_fallback_name(column: DatasetColumn, fallback: str) -> DatasetColumn:
    if column.name:
        return column
    payload = column.model_dump()
    payload["name"] = fallback
    return DatasetColumn.model_validate(payload)


def _measurement(design: ResearchStudyDesign, name: str):
    key = name.lower()
    for measurement in design.required_measurements:
        if measurement.name.lower() == key:
            return measurement
    return None


def _time_range(intent: ResearchIntent) -> str | None:
    if intent.time_range is None:
        return None
    if intent.time_range.raw:
        return intent.time_range.raw
    if intent.time_range.start_year and intent.time_range.end_year:
        return f"{intent.time_range.start_year}-{intent.time_range.end_year}"
    if intent.time_range.start_year:
        return f"с {intent.time_range.start_year}"
    if intent.time_range.end_year:
        return f"по {intent.time_range.end_year}"
    return None


def _is_measurement_indicator(measurement: Any) -> bool:
    role = str(getattr(measurement, "role", "") or "").strip().lower()
    return role not in {"grouping", "dimension", "metadata", "id", "identifier"}


def _is_dimension_or_metadata_name(value: str) -> bool:
    key = slugify(value, "").lower()
    blocked = {
        "year",
        "god",
        "date",
        "period",
        "country",
        "country_code",
        "country_id",
        "countryiso3code",
        "geo",
        "geography",
        "source",
        "source_name",
        "source_url",
        "source_dataset_id",
        "data_retrieval_date",
        "downloaded_at",
    }
    return key in blocked


def _contains(text: str, *tokens: str) -> bool:
    return any(token in text for token in tokens)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value)).strip()
        key = text.lower()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result
