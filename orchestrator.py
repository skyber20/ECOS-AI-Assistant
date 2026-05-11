import warnings
from enum import Enum
from typing import Any, TypedDict

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.simplefilter("ignore", LangChainPendingDeprecationWarning)

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator

from assembly_planner import DatasetBuildPlan, plan_dataset_build
from dataset_structure import TargetDatasetStructure, build_target_dataset_structure
from designer.research_designer import ResearchStudyDesign, design_research
from parser.intent_parser import (
    IntentParserError,
    IntentType,
    LLMSettings,
    NextAction,
    ResearchIntent,
    SYSTEM_PROMPT,
    _create_json_completion,
    _parse_intent_json,
    create_llm_settings,
)
from script_generator import GeneratedBuildScript, generate_build_script


class OrchestrationStatus(str, Enum):
    NEEDS_CLARIFICATION = "needs_clarification"
    READY_FOR_DESIGN = "ready_for_design"
    DESIGN_READY = "design_ready"
    NO_DATA = "no_data"
    UNSUPPORTED = "unsupported"


class ClarificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    reason: str
    question: str
    default_assumption: str | None = None


class ClarificationAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    question: str
    answer: str
    used_default: bool = False


class OrchestrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: OrchestrationStatus
    intent: ResearchIntent
    clarification_requests: list[ClarificationRequest] = Field(default_factory=list)
    research_design: ResearchStudyDesign | None = None
    dataset_structure: TargetDatasetStructure | None = None
    build_plan: DatasetBuildPlan | None = None
    build_script: GeneratedBuildScript | None = None
    message: str | None = None

    @field_validator("clarification_requests", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return [] if value is None else value


class PipelineState(TypedDict, total=False):
    query: str
    intent: ResearchIntent
    use_defaults: bool
    use_registry: bool
    readiness: OrchestrationResult
    research_design: ResearchStudyDesign
    dataset_structure: TargetDatasetStructure
    build_plan: DatasetBuildPlan
    build_script: GeneratedBuildScript
    result: OrchestrationResult


REFINE_INTENT_PROMPT = """Ты обновляешь JSON первого этапа ResearchIntent после уточнений пользователя.

На входе:
1. предыдущий ResearchIntent;
2. вопросы системы и ответы пользователя.

Правила:
- Верни только полный валидный JSON ResearchIntent без Markdown.
- Не меняй смысл запроса без необходимости.
- Обнови поля, к которым относятся ответы: topic, objects, geography, time_range, frequency, indicators, indicator_specs, entities, granularity, dataset_spec, research_questions, derived_metrics.
- Удали закрытые ambiguities и clarifying_questions.
- Если все блокирующие уточнения закрыты, next_action = "proceed_with_assumptions".
- Если что-то все еще неясно, оставь next_action = "ask_clarification" и добавь новые clarifying_questions.
- assumptions_if_no_answer оставь только для оставшихся неуточненных полей.
- Не запускай дизайн исследования и не придумывай числовые значения данных.
"""


class ResearchPipelineAgent:
    """Graph agent for the ECOS research-data pipeline."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.model = model
        self.graph = self._build_graph()

    def run(
        self,
        query: str,
        use_defaults: bool = False,
        use_registry: bool = True,
    ) -> OrchestrationResult:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        state = self.graph.invoke(
            {
                "query": query,
                "use_defaults": use_defaults,
                "use_registry": use_registry,
            }
        )
        return state["result"]

    def continue_from_intent(
        self,
        intent: ResearchIntent,
        use_defaults: bool = False,
        use_registry: bool = True,
    ) -> OrchestrationResult:
        state = self.graph.invoke(
            {
                "intent": intent,
                "use_defaults": use_defaults,
                "use_registry": use_registry,
            }
        )
        return state["result"]

    def refine_intent(
        self,
        intent: ResearchIntent,
        answers: list[ClarificationAnswer],
    ) -> ResearchIntent:
        if not answers:
            return intent
        content = _create_json_completion(
            self._settings(),
            build_refine_intent_messages(intent, answers),
        )
        return _parse_intent_json(content, original_query=intent.original_query)

    def _settings(self) -> LLMSettings:
        if self.settings is None:
            self.settings = create_llm_settings(provider=self.provider, model=self.model)
        return self.settings

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        graph.add_node("start", lambda state: state)
        graph.add_node("parse_intent", self._parse_intent)
        graph.add_node("validate_intent", self._validate_intent)
        graph.add_node("design_research", self._design_research)
        graph.add_node("dataset_structure", self._dataset_structure)
        graph.add_node("build_plan", self._build_plan)
        graph.add_node("build_script", self._build_script)

        graph.add_edge(START, "start")
        graph.add_conditional_edges(
            "start",
            lambda state: "validate_intent" if state.get("intent") else "parse_intent",
            {
                "parse_intent": "parse_intent",
                "validate_intent": "validate_intent",
            },
        )
        graph.add_edge("parse_intent", "validate_intent")
        graph.add_conditional_edges(
            "validate_intent",
            self._route_after_validation,
            {"design_research": "design_research", END: END},
        )
        graph.add_conditional_edges(
            "design_research",
            lambda state: END if state.get("result") else "dataset_structure",
            {"dataset_structure": "dataset_structure", END: END},
        )
        graph.add_edge("dataset_structure", "build_plan")
        graph.add_edge("build_plan", "build_script")
        graph.add_edge("build_script", END)
        return graph.compile()

    def _parse_intent(self, state: PipelineState) -> PipelineState:
        content = _create_json_completion(
            self._settings(),
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state["query"]},
            ],
        )
        return {"intent": _parse_intent_json(content, original_query=state["query"])}

    def _validate_intent(self, state: PipelineState) -> PipelineState:
        readiness = prepare_intent_for_design(
            state["intent"],
            use_defaults=state.get("use_defaults", False),
        )
        if readiness.status != OrchestrationStatus.READY_FOR_DESIGN:
            return {"readiness": readiness, "result": readiness}
        return {"readiness": readiness}

    def _route_after_validation(self, state: PipelineState) -> str:
        return END if state.get("result") else "design_research"

    def _design_research(self, state: PipelineState) -> PipelineState:
        design = design_research(state["intent"], settings=self._settings())
        if design.can_continue:
            return {"research_design": design}
        return {
            "research_design": design,
            "result": OrchestrationResult(
                status=OrchestrationStatus.DESIGN_READY,
                intent=state["intent"],
                clarification_requests=_result_clarifications(state),
                research_design=design,
                message="Дизайн исследования построен, но дальнейшая сборка заблокирована.",
            ),
        }

    def _dataset_structure(self, state: PipelineState) -> PipelineState:
        return {
            "dataset_structure": build_target_dataset_structure(
                state["intent"],
                state["research_design"],
            )
        }

    def _build_plan(self, state: PipelineState) -> PipelineState:
        return {
            "build_plan": plan_dataset_build(
                intent=state["intent"],
                design=state["research_design"],
                structure=state["dataset_structure"],
                settings=self._settings(),
                use_registry=state.get("use_registry", True),
            )
        }

    def _build_script(self, state: PipelineState) -> PipelineState:
        script = generate_build_script(
            state["dataset_structure"],
            state["build_plan"],
        )
        return {
            "build_script": script,
            "result": OrchestrationResult(
                status=OrchestrationStatus.DESIGN_READY,
                intent=state["intent"],
                clarification_requests=_result_clarifications(state),
                research_design=state["research_design"],
                dataset_structure=state["dataset_structure"],
                build_plan=state["build_plan"],
                build_script=script,
                message="Сформированы дизайн исследования, структура датасета, план сборки и скрипт.",
            ),
        }


LangGraphResearchAgent = ResearchPipelineAgent


def run_research_flow(
    query: str,
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
    use_defaults: bool = False,
    use_registry: bool = True,
) -> OrchestrationResult:
    return ResearchPipelineAgent(settings=settings, provider=provider, model=model).run(
        query,
        use_defaults=use_defaults,
        use_registry=use_registry,
    )


def continue_research_flow(
    intent: ResearchIntent,
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
    use_defaults: bool = False,
    use_registry: bool = True,
) -> OrchestrationResult:
    return ResearchPipelineAgent(settings=settings, provider=provider, model=model).continue_from_intent(
        intent,
        use_defaults=use_defaults,
        use_registry=use_registry,
    )


def refine_intent_with_clarifications(
    intent: ResearchIntent,
    answers: list[ClarificationAnswer],
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> ResearchIntent:
    try:
        return ResearchPipelineAgent(settings=settings, provider=provider, model=model).refine_intent(
            intent,
            answers,
        )
    except IntentParserError as exc:
        raise RuntimeError(f"Failed to refine ResearchIntent after clarification: {exc}") from exc


def build_refine_intent_messages(
    intent: ResearchIntent,
    answers: list[ClarificationAnswer],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REFINE_INTENT_PROMPT},
        {
            "role": "user",
            "content": (
                "=== ПРЕДЫДУЩИЙ ResearchIntent ===\n"
                f"{intent.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
                "=== ОТВЕТЫ ПОЛЬЗОВАТЕЛЯ НА УТОЧНЕНИЯ ===\n"
                f"{_answers_to_json(answers)}"
            ),
        },
    ]


def prepare_intent_for_design(
    intent: ResearchIntent,
    use_defaults: bool = False,
) -> OrchestrationResult:
    clarification_requests = validate_intent_for_design(intent)

    if intent.intent_type == IntentType.NO_DATA or intent.next_action == NextAction.REPORT_NO_DATA:
        return OrchestrationResult(
            status=OrchestrationStatus.NO_DATA,
            intent=intent,
            message=intent.data_availability.verdict
            or "Данные недоступны в верифицированных источниках.",
        )

    if intent.intent_type == IntentType.UNSUPPORTED or intent.next_action == NextAction.UNSUPPORTED:
        return OrchestrationResult(
            status=OrchestrationStatus.UNSUPPORTED,
            intent=intent,
            clarification_requests=clarification_requests,
            message="Запрос не поддерживается текущим пайплайном.",
        )

    if clarification_requests and not use_defaults:
        return OrchestrationResult(
            status=OrchestrationStatus.NEEDS_CLARIFICATION,
            intent=intent,
            clarification_requests=clarification_requests,
            message="Нужно уточнить недостающие поля перед дизайном исследования.",
        )

    missing_defaults = [
        request
        for request in clarification_requests
        if use_defaults and not request.default_assumption
    ]
    if missing_defaults:
        return OrchestrationResult(
            status=OrchestrationStatus.NEEDS_CLARIFICATION,
            intent=intent,
            clarification_requests=missing_defaults,
            message="Нужно уточнить поля, для которых нет безопасного default assumption.",
        )

    return OrchestrationResult(
        status=OrchestrationStatus.READY_FOR_DESIGN,
        intent=intent,
        clarification_requests=[] if use_defaults else clarification_requests,
    )


def validate_intent_for_design(intent: ResearchIntent) -> list[ClarificationRequest]:
    requests: list[ClarificationRequest] = []

    if intent.next_action == NextAction.ASK_CLARIFICATION:
        requests.extend(_requests_from_model_questions(intent))
    if not intent.topic:
        requests.append(_request(intent, "topic", "Не определена тема исследования.", "Какую тему или явление нужно изучить?"))
    if _requires_geography(intent) and not intent.geography:
        requests.append(_request(intent, "geography", "Не указана страна, регион или группа объектов.", "Какая страна, регион или группа объектов вас интересует?"))
    if _requires_time(intent) and _is_missing_time(intent):
        requests.append(_request(intent, "time_range", "Не указан временной период.", "За какой период нужны данные или исследование?"))
    if not intent.indicators and not intent.indicator_specs:
        requests.append(_request(intent, "indicators", "Не определены показатели для анализа.", "Какие показатели или метрики нужно использовать?"))
    if _requires_indicator_methodology(intent):
        requests.append(_request(intent, "indicator_specs", "Для одного или нескольких показателей не хватает определения или единицы измерения.", "Какую методику и единицы измерения использовать для ключевых показателей?"))
    if _requires_frequency(intent) and not intent.frequency:
        requests.append(_request(intent, "frequency", "Не указана частота наблюдений.", "Какая частота данных нужна: годовая, квартальная или месячная?"))
    if _requires_dataset_spec(intent) and _has_incomplete_dataset_spec(intent):
        requests.append(_request(intent, "dataset_spec", "Не полностью определена структура целевого датасета.", "Какую зернистость строк и ключевые столбцы ожидаете в итоговом датасете?"))
    if intent.intent_type == IntentType.RESEARCH and not intent.research_questions:
        requests.append(_request(intent, "research_questions", "Для исследовательской задачи не сформулированы исследовательские вопросы.", "Какие исследовательские вопросы нужно проверить?"))
    if intent.intent_type == IntentType.DERIVED and not intent.derived_metrics:
        requests.append(_request(intent, "derived_metrics", "Запрос требует производную метрику, но формула не определена.", "Какую формулу или базу нормализации использовать для производной метрики?"))

    return _deduplicate_requests(requests)


FIELD_DEFAULT_TOKENS = {
    "topic": [],
    "geography": ["географ", "страна", "регион"],
    "time_range": ["период", "диапазон", "последн", "год"],
    "frequency": ["частот", "годовая", "месячная", "квартальная"],
    "indicators": ["показател", "метрик", "ипц", "инфляц"],
    "indicator_specs": ["метод", "методик", "единиц", "ипц", "%"],
    "methodology": ["метод", "методик", "декабрь", "базов"],
    "dataset_spec": ["зернист", "строк", "датасет", "колон"],
    "research_questions": ["вопрос", "связ", "зависим"],
    "derived_metrics": ["формул", "баз", "нормал"],
}


def _result_clarifications(state: PipelineState) -> list[ClarificationRequest]:
    if state.get("use_defaults"):
        return []
    return state["readiness"].clarification_requests


def _request(
    intent: ResearchIntent,
    field: str,
    reason: str,
    question: str,
) -> ClarificationRequest:
    return ClarificationRequest(
        field=field,
        reason=reason,
        question=question,
        default_assumption=_default_for_field(intent, field),
    )


def _answers_to_json(answers: list[ClarificationAnswer]) -> str:
    return "[\n" + ",\n".join(
        answer.model_dump_json(indent=2, ensure_ascii=False) for answer in answers
    ) + "\n]"


def _requests_from_model_questions(intent: ResearchIntent) -> list[ClarificationRequest]:
    ambiguities = intent.ambiguities or ["Запрос требует уточнения."]
    requests = []
    for index, question in enumerate(intent.clarifying_questions):
        field = _guess_field(question, ambiguities[index] if index < len(ambiguities) else "")
        requests.append(
            ClarificationRequest(
                field=field,
                reason=ambiguities[index] if index < len(ambiguities) else "Недостаточно данных.",
                question=question,
                default_assumption=_default_for_field(intent, field) or _default_at(intent, index),
            )
        )
    return requests


def _requires_geography(intent: ResearchIntent) -> bool:
    return intent.intent_type in {
        IntentType.SIMPLE_DATA,
        IntentType.COMPARATIVE,
        IntentType.RESEARCH,
        IntentType.DERIVED,
        IntentType.AMBIGUOUS,
    }


def _requires_time(intent: ResearchIntent) -> bool:
    return intent.intent_type in {
        IntentType.SIMPLE_DATA,
        IntentType.COMPARATIVE,
        IntentType.DERIVED,
    }


def _requires_frequency(intent: ResearchIntent) -> bool:
    return intent.intent_type in {
        IntentType.SIMPLE_DATA,
        IntentType.COMPARATIVE,
        IntentType.DERIVED,
    }


def _requires_dataset_spec(intent: ResearchIntent) -> bool:
    return intent.intent_type in {
        IntentType.SIMPLE_DATA,
        IntentType.COMPARATIVE,
        IntentType.DERIVED,
    }


def _has_incomplete_dataset_spec(intent: ResearchIntent) -> bool:
    if intent.dataset_spec is None:
        return True
    has_row_grain = bool(intent.dataset_spec.row_grain or intent.granularity)
    has_columns = bool(intent.dataset_spec.columns)
    has_frequency = bool(intent.dataset_spec.frequency or intent.frequency)
    return not (has_row_grain and has_columns and has_frequency)


def _requires_indicator_methodology(intent: ResearchIntent) -> bool:
    if intent.intent_type in {IntentType.NO_DATA, IntentType.UNSUPPORTED}:
        return False
    return any(not spec.definition or not spec.unit for spec in intent.indicator_specs)


def _is_missing_time(intent: ResearchIntent) -> bool:
    if intent.time_range is None:
        return True
    if intent.intent_type == IntentType.DERIVED and intent.time_range.start_year:
        return False
    return not intent.time_range.is_explicit and not (
        intent.time_range.start_year or intent.time_range.end_year or intent.time_range.raw
    )


def _guess_field(question: str, reason: str) -> str:
    text = f"{question} {reason}".lower()
    field_markers = {
        "frequency": ["частота", "месяч", "квартал", "годовая"],
        "methodology": ["метод", "методик", "формул"],
        "geography": ["страна", "регион", "географ", "территор"],
        "time_range": ["период", "год", "время", "диапазон"],
        "indicators": ["показател", "метрик", "ипц", "инфляц"],
    }
    return next(
        (field for field, markers in field_markers.items() if any(marker in text for marker in markers)),
        "other",
    )


def _default_at(intent: ResearchIntent, index: int) -> str | None:
    if index < len(intent.assumptions_if_no_answer):
        return intent.assumptions_if_no_answer[index]
    return None


def _default_for_field(intent: ResearchIntent, field: str) -> str | None:
    tokens = FIELD_DEFAULT_TOKENS.get(field, [])
    for assumption in intent.assumptions_if_no_answer:
        lower = assumption.lower()
        if any(token in lower for token in tokens):
            return assumption
    return None


def _deduplicate_requests(
    requests: list[ClarificationRequest],
) -> list[ClarificationRequest]:
    result: list[ClarificationRequest] = []
    seen_fields: set[str] = set()
    for request in requests:
        if request.field not in seen_fields:
            seen_fields.add(request.field)
            result.append(request)
    return result
