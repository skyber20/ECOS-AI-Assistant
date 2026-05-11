import json
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator

from intent_parser import (
    IntentParserError,
    IntentType,
    LLMSettings,
    NextAction,
    ResearchIntent,
    _create_json_completion,
    _load_json_object,
    create_llm_settings,
    none_to_empty_list,
)


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Short id: H1, H2, ...")
    statement: str = Field(description="Проверяемая содержательная гипотеза.")
    null_hypothesis: str = Field(description="Нулевая гипотеза.")
    direction: str = Field(
        description="Ожидаемое направление: positive, negative, u_shaped, n_shaped, none."
    )
    variables: list[str] = Field(default_factory=list)
    expected_effect: str | None = Field(
        default=None,
        description="Короткое описание ожидаемого эффекта, если применимо.",
    )

    @field_validator("variables", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class RequiredMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    definition: str | None = None
    unit: str | None = None
    role: str = Field(description="primary, control, grouping, input, denominator.")
    source_candidates: list[str] = Field(default_factory=list)
    is_critical: bool = True

    @field_validator("source_candidates", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class GroupingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str
    levels: list[str] = Field(default_factory=list)
    rationale: str
    min_observations: int | None = None

    @field_validator("levels", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class StudyDerivedMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    formula: str
    formula_explanation: str
    inputs: list[str] = Field(default_factory=list)
    normalization: str | None = None
    aggregation_rules: list[str] = Field(default_factory=list)
    unit: str | None = None
    interpretation: str

    @field_validator("inputs", "aggregation_rules", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class StatisticalMethod(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    purpose: str
    variables: list[str] = Field(default_factory=list)
    hypothesis_id: str | None = None
    expected_output: str
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("variables", "assumptions", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class VisualizationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    chart_type: str
    x_axis: str | None = None
    y_axis: str | None = None
    grouping: str | None = None
    metrics_used: list[str] = Field(default_factory=list)
    hypothesis_id: str | None = None
    interpretation_guide: str

    @field_validator("metrics_used", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class DataQualityCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str
    method: str
    action_if_failed: str


class ResearchStudyDesign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str
    design_summary: str
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    required_measurements: list[RequiredMeasurement] = Field(default_factory=list)
    grouping_rules: list[GroupingRule] = Field(default_factory=list)
    derived_metrics: list[StudyDerivedMetric] = Field(default_factory=list)
    statistical_methods: list[StatisticalMethod] = Field(default_factory=list)
    visualizations: list[VisualizationSpec] = Field(default_factory=list)
    data_quality_checks: list[DataQualityCheck] = Field(default_factory=list)
    methodology_notes: str
    required_row_grain: list[str] = Field(default_factory=list)
    can_continue: bool = True
    blocking_reasons: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("blocking_reasons", "blocking_reasones"),
    )

    @field_validator(
        "hypotheses",
        "required_measurements",
        "grouping_rules",
        "derived_metrics",
        "statistical_methods",
        "visualizations",
        "data_quality_checks",
        "required_row_grain",
        "blocking_reasons",
        mode="before",
    )
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


RESEARCH_DESIGN_PROMPT = """Ты модуль дизайна количественного исследования для системы сборки социально-экономических датасетов.

На входе ты получаешь:
1. исходный пользовательский запрос;
2. JSON первого этапа ResearchIntent.

ResearchIntent - главный контракт для дизайна исследования. Исходный запрос используй только как дополнительный контекст для формулировок и нюансов, но не извлекай из него заново географию, период, показатели или тип задачи в обход JSON.

Твоя задача: построить исследовательский дизайн:
- проверяемые гипотезы;
- необходимые измерения и индикаторы;
- методы группировки;
- производные метрики с формулами, нормализацией и правилами агрегирования;
- статистические методы;
- ожидаемые визуализации;
- проверки качества данных.

Правила:
- Верни только валидный JSON без Markdown.
- Не придумывай числовые значения наблюдений.
- Не утверждай, что данные уже собраны. Источники - только кандидаты.
- Все гипотезы, измерения, группировки, формулы и визуализации должны быть трассируемы к полям ResearchIntent: indicators, indicator_specs, research_questions, research_design, derived_metrics, dataset_spec, geography, time_range, frequency, assumptions_if_no_answer.
- Если исходный запрос и ResearchIntent конфликтуют, следуй ResearchIntent и укажи ограничение в methodology_notes.
- Если intent_type=no_data, выставь can_continue=false и объясни blocking_reasons.
- Если intent_type=ambiguous, можно предложить дизайн только на основе assumptions_if_no_answer; явно укажи это в methodology_notes.
- Для simple_data достаточно 1-2 гипотез/аналитических ожиданий и визуализаций динамики.
- Для comparative добавь гипотезы о различиях между объектами и методы сравнения.
- Для research добавь 2-4 содержательные гипотезы, методы связи/контроля и контрольные переменные.
- Для derived обязательно перенеси или уточни формулы из intent. Формула должна быть проверяемой и гарантировать заявленную базу/нормализацию.
- required_measurements должны покрывать primary indicators, controls, grouping dimensions и inputs производных метрик.
- visualizations должны быть конкретными: тип графика, оси, метрики, как читать.
- Используй русский язык в текстовых полях.
"""


class ResearchDesignerError(RuntimeError):
    """Raised when a study design cannot be generated or parsed."""


def design_research(
    intent: ResearchIntent,
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> ResearchStudyDesign:
    llm_settings = settings or create_llm_settings(provider=provider, model=model)
    messages = build_research_design_messages(intent)

    content = _create_json_completion(llm_settings, messages)
    return _parse_design_json(content, intent)


def build_research_design_messages(intent: ResearchIntent) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"{RESEARCH_DESIGN_PROMPT}\n\n"
                "JSON Schema:\n"
                f"{json.dumps(ResearchStudyDesign.model_json_schema(), ensure_ascii=False, indent=2)}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Построй дизайн исследования по исходному запросу и JSON первого этапа.\n\n"
                "=== ИСХОДНЫЙ ЗАПРОС ===\n"
                f"{intent.original_query}\n\n"
                "=== JSON ПЕРВОГО ЭТАПА: ResearchIntent ===\n"
                f"{intent.model_dump_json(indent=2, ensure_ascii=False)}"
            ),
        },
    ]


def _parse_design_json(content: str, intent: ResearchIntent) -> ResearchStudyDesign:
    try:
        data = _load_json_object(content)
    except (IntentParserError, json.JSONDecodeError) as exc:
        raise ResearchDesignerError(f"LLM returned invalid research design JSON: {exc}") from exc

    if not data.get("original_query"):
        data["original_query"] = intent.original_query

    try:
        design = ResearchStudyDesign.model_validate(data)
    except ValidationError as exc:
        raise ResearchDesignerError(
            f"LLM JSON does not match ResearchStudyDesign schema: {exc}"
        ) from exc

    return _apply_design_corrections(design, intent)


def _apply_design_corrections(
    design: ResearchStudyDesign,
    intent: ResearchIntent,
) -> ResearchStudyDesign:
    if intent.next_action == NextAction.REPORT_NO_DATA or intent.intent_type == IntentType.NO_DATA:
        design.can_continue = False
        if not design.blocking_reasons:
            design.blocking_reasons = intent.data_availability.reasons or [
                "Данные недоступны в верифицированных источниках."
            ]

    if intent.dataset_spec and intent.dataset_spec.row_grain and not design.required_row_grain:
        design.required_row_grain = [intent.dataset_spec.row_grain]
    elif intent.granularity and not design.required_row_grain:
        design.required_row_grain = [intent.granularity]

    return design
