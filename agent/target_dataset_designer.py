import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent.intent_parser import (
    IntentParserError,
    IntentType,
    NextAction,
    ResearchIntent,
    _load_json_object,
    none_to_empty_list,
)
from agent.research_designer import ResearchStudyDesign


class DatasetColumn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Machine-friendly snake_case column name.")
    title: str | None = Field(default=None, description="Human-readable column title.")
    role: str = Field(
        description=(
            "Column role: identifier, time, geography, dimension, indicator, "
            "derived_metric, source_metadata, quality_flag."
        )
    )
    data_type: str = Field(
        description="Expected type: string, integer, float, decimal, date, year, boolean, category."
    )
    unit: str | None = None
    definition: str
    nullable: bool = True
    is_required: bool = True
    source: str | None = None
    calculation: str | None = None
    allowed_values: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("allowed_values", "depends_on", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class TargetDatasetStructure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str
    dataset_name: str
    dataset_purpose: str
    row_grain: str
    primary_key: list[str] = Field(default_factory=list)
    time_coverage: str | None = None
    geography_coverage: list[str] = Field(default_factory=list)
    frequency: str | None = None
    dimensions: list[DatasetColumn] = Field(default_factory=list)
    indicators: list[DatasetColumn] = Field(default_factory=list)
    derived_metrics: list[DatasetColumn] = Field(default_factory=list)
    metadata_columns: list[DatasetColumn] = Field(default_factory=list)
    quality_flags: list[DatasetColumn] = Field(default_factory=list)
    expected_row_count: str | None = None
    source_requirements: list[str] = Field(default_factory=list)
    validation_rules: list[str] = Field(default_factory=list)
    join_keys: list[str] = Field(default_factory=list)
    can_build: bool = True
    blocking_reasons: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator(
        "primary_key",
        "geography_coverage",
        "dimensions",
        "indicators",
        "derived_metrics",
        "metadata_columns",
        "quality_flags",
        "source_requirements",
        "validation_rules",
        "join_keys",
        "blocking_reasons",
        mode="before",
    )
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


TARGET_DATASET_PROMPT = """Ты модуль проектирования структуры целевого датасета для системы сборки социально-экономических данных.

На входе ты получаешь:
1. формализованный ResearchIntent;
2. ResearchStudyDesign.

Твоя задача: описать структуру итогового датасета, который нужен для исследования экономиста.

Определи:
- зернистость строки row_grain;
- primary_key и join_keys;
- измерения: география, время, объект наблюдения, группы сравнения;
- индикаторы с типами данных, единицами измерения и методологическими определениями;
- производные метрики с формулами и зависимостями;
- служебные поля источника, даты выгрузки, качества и проверок;
- правила валидации данных.

Правила:
- Верни только валидный JSON без Markdown.
- Не собирай данные и не придумывай числовые значения наблюдений.
- Структура должна быть трассируема к ResearchIntent и ResearchStudyDesign.
- Для каждой колонки укажи name, role, data_type, unit, definition, nullable, is_required.
- Для индикаторов и производных метрик unit и definition обязательны, если они известны из предыдущих этапов.
- Включай source_metadata колонки, например source_name, source_indicator_code, source_url, extracted_at, если они нужны для воспроизводимости.
- Если данные недоступны или дизайн заблокирован, can_build=false и объясни blocking_reasons.
- Используй snake_case для технических имен колонок и русский язык в описательных полях.
"""


class TargetDatasetDesignerError(RuntimeError):
    """Raised when the target dataset structure cannot be generated or parsed."""


def build_target_dataset_messages(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"{TARGET_DATASET_PROMPT}\n\n"
                "JSON Schema:\n"
                f"{json.dumps(TargetDatasetStructure.model_json_schema(), ensure_ascii=False, indent=2)}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Опиши структуру целевого датасета по ResearchIntent и ResearchStudyDesign.\n\n"
                "=== ResearchIntent ===\n"
                f"{intent.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
                "=== ResearchStudyDesign ===\n"
                f"{design.model_dump_json(indent=2, ensure_ascii=False)}"
            ),
        },
    ]


def _parse_target_dataset_json(
    content: str,
    intent: ResearchIntent,
    design: ResearchStudyDesign,
) -> TargetDatasetStructure:
    try:
        data = _load_json_object(content)
    except (IntentParserError, json.JSONDecodeError) as exc:
        raise TargetDatasetDesignerError(
            f"LLM returned invalid target dataset JSON: {exc}"
        ) from exc

    if not data.get("original_query"):
        data["original_query"] = intent.original_query

    try:
        structure = TargetDatasetStructure.model_validate(data)
    except ValidationError as exc:
        raise TargetDatasetDesignerError(
            f"LLM JSON does not match TargetDatasetStructure schema: {exc}"
        ) from exc

    return _apply_target_dataset_corrections(structure, intent, design)


def _apply_target_dataset_corrections(
    structure: TargetDatasetStructure,
    intent: ResearchIntent,
    design: ResearchStudyDesign,
) -> TargetDatasetStructure:
    if intent.dataset_spec and intent.dataset_spec.row_grain and not structure.row_grain:
        structure.row_grain = intent.dataset_spec.row_grain
    elif design.required_row_grain and not structure.row_grain:
        structure.row_grain = design.required_row_grain[0]
    elif intent.granularity and not structure.row_grain:
        structure.row_grain = intent.granularity

    if intent.frequency and not structure.frequency:
        structure.frequency = intent.frequency
    if intent.geography and not structure.geography_coverage:
        structure.geography_coverage = intent.geography
    if intent.dataset_spec and intent.dataset_spec.rows_approx and not structure.expected_row_count:
        structure.expected_row_count = intent.dataset_spec.rows_approx

    if (
        intent.intent_type == IntentType.NO_DATA
        or intent.next_action == NextAction.REPORT_NO_DATA
        or not design.can_continue
    ):
        structure.can_build = False
        if not structure.blocking_reasons:
            structure.blocking_reasons = (
                design.blocking_reasons
                or intent.data_availability.reasons
                or ["Целевой датасет нельзя построить без верифицированных данных."]
            )

    return structure
