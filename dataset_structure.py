import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intent_parser import ResearchIntent, none_to_empty_list
from research_designer import ResearchStudyDesign


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

    for column in _dimension_columns(intent, row_grain):
        _append_column(columns, column)
    for column in _measurement_columns(intent, design):
        _append_column(columns, column)
    for column in _derived_columns(intent, design):
        _append_column(columns, column)
    for column in _metadata_columns():
        _append_column(columns, column)

    return TargetDatasetStructure(
        row_grain=row_grain,
        primary_key=[column.name for column in columns if column.role == "dimension" and not column.nullable],
        columns=columns,
        expected_frequency=intent.frequency or (intent.dataset_spec.frequency if intent.dataset_spec else None),
        time_range=_time_range(intent),
        geography=list(dict.fromkeys([*intent.geography, *intent.objects, *intent.entities])),
        design_notes=_design_notes(intent, design),
    )


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
    names = _unique([
        *intent.indicators,
        *[spec.name for spec in intent.indicator_specs],
        *[measurement.name for measurement in design.required_measurements],
    ])
    columns: list[DatasetColumn] = []

    for index, name in enumerate(names, start=1):
        spec = specs.get(name.lower())
        measurement = _measurement(design, name)
        columns.append(
            DatasetColumn(
                name=f"indicator_{index}",
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
        seen.add(metric.name.lower())
        columns.append(
            DatasetColumn(
                name=f"derived_{len(columns) + 1}",
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
        if metric.name.lower() in seen:
            continue
        columns.append(
            DatasetColumn(
                name=f"derived_{len(columns) + 1}",
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
    ]


def _design_notes(intent: ResearchIntent, design: ResearchStudyDesign) -> list[str]:
    notes: list[str] = []
    if intent.dataset_spec and intent.dataset_spec.columns:
        notes.append("Ожидаемые колонки из запроса: " + ", ".join(intent.dataset_spec.columns))
    if design.methodology_notes:
        notes.append(design.methodology_notes)
    if intent.assumptions_if_no_answer:
        notes.append("Допущения: " + "; ".join(intent.assumptions_if_no_answer))
    notes.append("Физическая схема исходных файлов не заявляется, пока explorer не прочитал data_path.")
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
