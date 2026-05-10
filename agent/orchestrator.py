from enum import Enum
import warnings
from typing import Any, Literal, TypedDict

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.simplefilter("ignore", LangChainPendingDeprecationWarning)
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.dataset_search_planner import (
    DatasetMatchReport,
    DatasetRegistry,
    EmptyDatasetRegistry,
    create_dataset_match_report,
)
from agent.intent_parser import (
    IntentType,
    LLMSettings,
    NextAction,
    ResearchIntent,
    SYSTEM_PROMPT,
    _create_json_completion,
    _parse_intent_json,
    create_llm_settings,
)
from research_designer import (
    ResearchStudyDesign,
    _parse_design_json,
    build_research_design_messages,
)
from target_dataset_designer import (
    TargetDatasetStructure,
    _parse_target_dataset_json,
    build_target_dataset_messages,
)


class OrchestrationStatus(str, Enum):
    NEEDS_CLARIFICATION = "needs_clarification"
    READY_FOR_DESIGN = "ready_for_design"
    DESIGN_READY = "design_ready"
    BUILD_PLAN_READY = "build_plan_ready"
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
    target_dataset_structure: TargetDatasetStructure | None = None
    dataset_match_report: DatasetMatchReport | None = None
    message: str | None = None

    @field_validator("clarification_requests", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return [] if value is None else value


DEFAULT_MAX_TOOL_RETRIES = 2
DEFAULT_MAX_GRAPH_STEPS = 16

GraphOperation = Literal[
    "parse_intent",
    "run",
    "continue",
    "refine_intent",
]


class ToolExecutionError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    attempt: int
    error: str
    raw_output: str | None = None


class ResearchAgentState(TypedDict, total=False):
    operation: GraphOperation
    query: str
    intent: ResearchIntent
    clarification_answers: list[ClarificationAnswer]
    settings: LLMSettings
    use_defaults: bool
    readiness: OrchestrationResult
    research_design: ResearchStudyDesign
    target_dataset_structure: TargetDatasetStructure
    dataset_match_report: DatasetMatchReport
    result: OrchestrationResult
    next_node: str
    refined_done: bool
    last_tool: str
    last_error: str | None
    last_raw_output: str | None
    tool_errors: list[ToolExecutionError]
    tool_attempts: dict[str, int]
    graph_steps: int
    max_tool_retries: int
    max_graph_steps: int


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

INTENT_REPAIR_PROMPT = """Ты исправляешь ответ инструмента parse_intent.

На входе:
1. исходный пользовательский запрос;
2. предыдущий ответ LLM;
3. ошибка парсинга или валидации.

Правила:
- Верни только полный валидный JSON ResearchIntent без Markdown.
- Исправь только формат, типы, enum-значения, лишние или пропущенные поля.
- Не меняй смысл исходного запроса без необходимости.
- Не добавляй поля вне схемы ResearchIntent.
"""

REFINE_REPAIR_PROMPT = """Ты исправляешь ответ инструмента refine_intent.

На входе:
1. предыдущий ResearchIntent;
2. ответы пользователя на уточнения;
3. предыдущий ответ LLM;
4. ошибка парсинга или валидации.

Правила:
- Верни только полный валидный JSON ResearchIntent без Markdown.
- Сохрани смысл уточнений пользователя.
- Исправь только формат, типы, enum-значения, лишние или пропущенные поля.
- Не запускай дизайн исследования и не придумывай числовые значения данных.
"""

DESIGN_REPAIR_PROMPT = """Ты исправляешь ответ инструмента design_research.

На входе:
1. исходный ResearchIntent;
2. предыдущий ответ LLM;
3. ошибка парсинга или валидации.

Правила:
- Верни только полный валидный JSON ResearchStudyDesign без Markdown.
- Исправь только формат, типы, enum-значения, лишние или пропущенные поля.
- Все содержательные решения должны оставаться трассируемыми к ResearchIntent.
- Не придумывай числовые значения наблюдений.
"""

TARGET_DATASET_REPAIR_PROMPT = """Ты исправляешь ответ инструмента target_dataset_structure.

На входе:
1. исходный ResearchIntent;
2. ResearchStudyDesign;
3. предыдущий ответ LLM;
4. ошибка парсинга или валидации.

Правила:
- Верни только полный валидный JSON TargetDatasetStructure без Markdown.
- Исправь только формат, типы, enum-значения, лишние или пропущенные поля.
- Сохрани структуру целевого датасета: row_grain, primary_key, dimensions, indicators, derived_metrics, metadata_columns и validation_rules.
- Не придумывай числовые значения наблюдений.
"""


class LangGraphResearchAgent:
    """LangGraph agent that coordinates research parsing, clarification and design."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        provider: str | None = None,
        model: str | None = None,
        dataset_registry: DatasetRegistry | None = None,
        max_tool_retries: int = DEFAULT_MAX_TOOL_RETRIES,
        max_graph_steps: int = DEFAULT_MAX_GRAPH_STEPS,
    ) -> None:
        self.settings = settings or create_llm_settings(provider=provider, model=model)
        self.dataset_registry = dataset_registry or EmptyDatasetRegistry()
        self.max_tool_retries = max_tool_retries
        self.max_graph_steps = max_graph_steps
        self.graph = self._build_graph()

    def parse_intent(self, query: str) -> ResearchIntent:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        state = self._invoke(
            {
                "operation": "parse_intent",
                "query": query,
            }
        )
        return state["intent"]

    def run(self, query: str, use_defaults: bool = False) -> OrchestrationResult:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        state = self._invoke(
            {
                "operation": "run",
                "query": query,
                "use_defaults": use_defaults,
            }
        )
        return state["result"]

    def continue_from_intent(
        self,
        intent: ResearchIntent,
        use_defaults: bool = False,
    ) -> OrchestrationResult:
        state = self._invoke(
            {
                "operation": "continue",
                "intent": intent,
                "use_defaults": use_defaults,
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

        state = self._invoke(
            {
                "operation": "refine_intent",
                "intent": intent,
                "clarification_answers": answers,
            }
        )
        return state["intent"]

    def _invoke(self, initial_state: ResearchAgentState) -> ResearchAgentState:
        state: ResearchAgentState = {
            "settings": self.settings,
            "max_tool_retries": self.max_tool_retries,
            "max_graph_steps": self.max_graph_steps,
            "tool_attempts": {},
            "tool_errors": [],
            "graph_steps": 0,
            **initial_state,
        }
        return self.graph.invoke(state)

    def _build_graph(self):
        graph = StateGraph(ResearchAgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("parse_intent", self._parse_intent_tool)
        graph.add_node("refine_intent", self._refine_intent_tool)
        graph.add_node("validate_intent", self._validate_intent_tool)
        graph.add_node("design_research", self._design_research_tool)
        graph.add_node("target_dataset_structure", self._target_dataset_tool)
        graph.add_node("search_and_build_plan", self._search_build_plan_tool)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._route_from_agent,
            {
                "parse_intent": "parse_intent",
                "refine_intent": "refine_intent",
                "validate_intent": "validate_intent",
                "design_research": "design_research",
                "target_dataset_structure": "target_dataset_structure",
                "search_and_build_plan": "search_and_build_plan",
                END: END,
            },
        )
        graph.add_edge("parse_intent", "agent")
        graph.add_edge("refine_intent", "agent")
        graph.add_edge("validate_intent", "agent")
        graph.add_edge("design_research", "agent")
        graph.add_edge("target_dataset_structure", "agent")
        graph.add_edge("search_and_build_plan", "agent")
        return graph.compile()

    def _agent_node(self, state: ResearchAgentState) -> ResearchAgentState:
        graph_steps = state.get("graph_steps", 0) + 1
        max_graph_steps = state.get("max_graph_steps", DEFAULT_MAX_GRAPH_STEPS)
        if graph_steps > max_graph_steps:
            raise RuntimeError(
                f"LangGraph research agent exceeded {max_graph_steps} graph steps."
            )

        if state.get("last_error"):
            last_tool = state.get("last_tool") or "unknown"
            attempts = state.get("tool_attempts", {}).get(last_tool, 0)
            max_retries = state.get("max_tool_retries", DEFAULT_MAX_TOOL_RETRIES)
            if attempts <= max_retries:
                return {
                    "graph_steps": graph_steps,
                    "next_node": last_tool,
                }
            raise RuntimeError(
                f"Tool {last_tool} failed after {attempts} attempts: "
                f"{state['last_error']}"
            )

        operation = state.get("operation", "run")
        if operation == "parse_intent":
            next_node = END if state.get("intent") else "parse_intent"
        elif operation == "refine_intent":
            next_node = END if state.get("refined_done") else "refine_intent"
        elif not state.get("intent"):
            next_node = "parse_intent"
        elif not state.get("readiness"):
            next_node = "validate_intent"
        elif state["readiness"].status != OrchestrationStatus.READY_FOR_DESIGN:
            return {
                "graph_steps": graph_steps,
                "result": state["readiness"],
                "next_node": END,
            }
        elif not state.get("research_design"):
            next_node = "design_research"
        elif not state.get("target_dataset_structure"):
            next_node = "target_dataset_structure"
        elif not state.get("dataset_match_report"):
            next_node = "search_and_build_plan"
        else:
            return {
                "graph_steps": graph_steps,
                "result": OrchestrationResult(
                    status=OrchestrationStatus.BUILD_PLAN_READY,
                    intent=state["intent"],
                    clarification_requests=(
                        []
                        if state.get("use_defaults")
                        else state["readiness"].clarification_requests
                    ),
                    research_design=state["research_design"],
                    target_dataset_structure=state["target_dataset_structure"],
                    dataset_match_report=state["dataset_match_report"],
                ),
                "next_node": END,
            }

        return {
            "graph_steps": graph_steps,
            "next_node": next_node,
        }

    def _route_from_agent(self, state: ResearchAgentState) -> str:
        return state.get("next_node", END)

    def _parse_intent_tool(self, state: ResearchAgentState) -> ResearchAgentState:
        tool = "parse_intent"
        query = state["query"]
        raw_output = None
        attempts = _increment_tool_attempts(state, tool)
        try:
            messages = _build_parse_intent_messages(state)
            raw_output = _create_json_completion(state["settings"], messages)
            intent = _parse_intent_json(raw_output, original_query=query)
            return _tool_success(
                state,
                {
                    "intent": intent,
                    "readiness": None,
                    "research_design": None,
                    "target_dataset_structure": None,
                    "dataset_match_report": None,
                }
            )
        except Exception as exc:
            return _tool_failure(state, tool, attempts, exc, raw_output)

    def _refine_intent_tool(self, state: ResearchAgentState) -> ResearchAgentState:
        tool = "refine_intent"
        raw_output = None
        attempts = _increment_tool_attempts(state, tool)
        try:
            messages = _build_refine_messages_for_state(state)
            raw_output = _create_json_completion(state["settings"], messages)
            intent = _parse_intent_json(
                raw_output,
                original_query=state["intent"].original_query,
            )
            return _tool_success(
                state,
                {
                    "intent": intent,
                    "refined_done": True,
                    "readiness": None,
                    "research_design": None,
                    "target_dataset_structure": None,
                    "dataset_match_report": None,
                }
            )
        except Exception as exc:
            return _tool_failure(state, tool, attempts, exc, raw_output)

    def _validate_intent_tool(self, state: ResearchAgentState) -> ResearchAgentState:
        readiness = prepare_intent_for_design(
            state["intent"],
            use_defaults=state.get("use_defaults", False),
        )
        return {
            "readiness": readiness,
            "last_tool": "",
            "last_error": None,
            "last_raw_output": None,
        }

    def _design_research_tool(self, state: ResearchAgentState) -> ResearchAgentState:
        tool = "design_research"
        raw_output = None
        attempts = _increment_tool_attempts(state, tool)
        try:
            messages = _build_design_messages_for_state(state)
            raw_output = _create_json_completion(state["settings"], messages)
            design = _parse_design_json(raw_output, state["intent"])
            return _tool_success(
                state,
                {
                    "research_design": design,
                    "target_dataset_structure": None,
                    "dataset_match_report": None,
                },
            )
        except Exception as exc:
            return _tool_failure(state, tool, attempts, exc, raw_output)

    def _target_dataset_tool(self, state: ResearchAgentState) -> ResearchAgentState:
        tool = "target_dataset_structure"
        raw_output = None
        attempts = _increment_tool_attempts(state, tool)
        try:
            messages = _build_target_dataset_messages_for_state(state)
            raw_output = _create_json_completion(state["settings"], messages)
            target_dataset_structure = _parse_target_dataset_json(
                raw_output,
                state["intent"],
                state["research_design"],
            )
            return _tool_success(
                state,
                {
                    "target_dataset_structure": target_dataset_structure,
                    "dataset_match_report": None,
                },
            )
        except Exception as exc:
            return _tool_failure(state, tool, attempts, exc, raw_output)

    def _search_build_plan_tool(self, state: ResearchAgentState) -> ResearchAgentState:
        tool = "search_and_build_plan"
        attempts = _increment_tool_attempts(state, tool)
        try:
            dataset_match_report = create_dataset_match_report(
                intent=state["intent"],
                design=state["research_design"],
                target=state["target_dataset_structure"],
                registry=self.dataset_registry,
            )
            return _tool_success(
                state,
                {"dataset_match_report": dataset_match_report},
            )
        except Exception as exc:
            return _tool_failure(state, tool, attempts, exc, None)


def _increment_tool_attempts(state: ResearchAgentState, tool: str) -> int:
    attempts = dict(state.get("tool_attempts", {}))
    attempt = attempts.get(tool, 0) + 1
    attempts[tool] = attempt
    state["tool_attempts"] = attempts
    return attempt


def _tool_success(
    state: ResearchAgentState,
    payload: ResearchAgentState,
) -> ResearchAgentState:
    return {
        **payload,
        "last_tool": "",
        "last_error": None,
        "last_raw_output": None,
        "tool_attempts": state.get("tool_attempts", {}),
    }


def _tool_failure(
    state: ResearchAgentState,
    tool: str,
    attempt: int,
    exc: Exception,
    raw_output: str | None,
) -> ResearchAgentState:
    error = _format_tool_error(exc)
    tool_errors = list(state.get("tool_errors", []))
    tool_errors.append(
        ToolExecutionError(
            tool=tool,
            attempt=attempt,
            error=error,
            raw_output=raw_output,
        )
    )
    return {
        "last_tool": tool,
        "last_error": error,
        "last_raw_output": raw_output,
        "tool_errors": tool_errors,
        "tool_attempts": state.get("tool_attempts", {}),
    }


def _format_tool_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _build_parse_intent_messages(state: ResearchAgentState) -> list[dict[str, str]]:
    query = state["query"]
    if not state.get("last_error"):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

    return [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\n{INTENT_REPAIR_PROMPT}",
        },
        {
            "role": "user",
            "content": _build_repair_user_message(
                task_context=f"=== ИСХОДНЫЙ ЗАПРОС ===\n{query}",
                raw_output=state.get("last_raw_output"),
                error=state["last_error"] or "",
                instruction="Верни исправленный полный JSON ResearchIntent.",
            ),
        },
    ]


def _build_refine_messages_for_state(
    state: ResearchAgentState,
) -> list[dict[str, str]]:
    if not state.get("last_error"):
        return build_refine_intent_messages(
            state["intent"],
            state.get("clarification_answers", []),
        )

    return [
        {
            "role": "system",
            "content": REFINE_REPAIR_PROMPT,
        },
        {
            "role": "user",
            "content": _build_repair_user_message(
                task_context=(
                    "=== ПРЕДЫДУЩИЙ ResearchIntent ===\n"
                    f"{state['intent'].model_dump_json(indent=2, ensure_ascii=False)}\n\n"
                    "=== ОТВЕТЫ ПОЛЬЗОВАТЕЛЯ НА УТОЧНЕНИЯ ===\n"
                    f"{_answers_to_json(state.get('clarification_answers', []))}"
                ),
                raw_output=state.get("last_raw_output"),
                error=state["last_error"] or "",
                instruction="Верни исправленный полный JSON ResearchIntent.",
            ),
        },
    ]


def _build_design_messages_for_state(
    state: ResearchAgentState,
) -> list[dict[str, str]]:
    if not state.get("last_error"):
        return build_research_design_messages(state["intent"])

    return [
        {
            "role": "system",
            "content": DESIGN_REPAIR_PROMPT,
        },
        {
            "role": "user",
            "content": _build_repair_user_message(
                task_context=(
                    "=== ResearchIntent ===\n"
                    f"{state['intent'].model_dump_json(indent=2, ensure_ascii=False)}"
                ),
                raw_output=state.get("last_raw_output"),
                error=state["last_error"] or "",
                instruction="Верни исправленный полный JSON ResearchStudyDesign.",
            ),
        },
    ]


def _build_target_dataset_messages_for_state(
    state: ResearchAgentState,
) -> list[dict[str, str]]:
    if not state.get("last_error"):
        return build_target_dataset_messages(
            state["intent"],
            state["research_design"],
        )

    return [
        {
            "role": "system",
            "content": TARGET_DATASET_REPAIR_PROMPT,
        },
        {
            "role": "user",
            "content": _build_repair_user_message(
                task_context=(
                    "=== ResearchIntent ===\n"
                    f"{state['intent'].model_dump_json(indent=2, ensure_ascii=False)}\n\n"
                    "=== ResearchStudyDesign ===\n"
                    f"{state['research_design'].model_dump_json(indent=2, ensure_ascii=False)}"
                ),
                raw_output=state.get("last_raw_output"),
                error=state["last_error"] or "",
                instruction="Верни исправленный полный JSON TargetDatasetStructure.",
            ),
        },
    ]


def _build_repair_user_message(
    task_context: str,
    raw_output: str | None,
    error: str,
    instruction: str,
) -> str:
    return (
        f"{task_context}\n\n"
        "=== ПРЕДЫДУЩИЙ ОШИБОЧНЫЙ ОТВЕТ LLM ===\n"
        f"{raw_output or '<пустой ответ или ошибка до получения ответа>'}\n\n"
        "=== ОШИБКА ИНСТРУМЕНТА ===\n"
        f"{error}\n\n"
        f"{instruction}"
    )


def build_refine_intent_messages(
    intent: ResearchIntent,
    answers: list[ClarificationAnswer],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": REFINE_INTENT_PROMPT,
        },
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
            clarification_requests=[],
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
    if clarification_requests and use_defaults:
        missing_defaults = [
            request
            for request in clarification_requests
            if not request.default_assumption
        ]
        if missing_defaults:
            return OrchestrationResult(
                status=OrchestrationStatus.NEEDS_CLARIFICATION,
                intent=intent,
                clarification_requests=missing_defaults,
                message=(
                    "Нужно уточнить поля, для которых нет безопасного default "
                    "assumption."
                ),
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
        requests.append(
            ClarificationRequest(
                field="topic",
                reason="Не определена тема исследования.",
                question="Какую тему или явление нужно изучить?",
                default_assumption=_default_at(intent, 0),
            )
        )

    if _requires_geography(intent) and not intent.geography:
        requests.append(
            ClarificationRequest(
                field="geography",
                reason="Не указана страна, регион или группа объектов.",
                question="Какая страна, регион или группа объектов вас интересует?",
                default_assumption=_default_containing(intent, ["географ", "страна", "регион"]),
            )
        )

    if _requires_time(intent) and _is_missing_time(intent):
        requests.append(
            ClarificationRequest(
                field="time_range",
                reason="Не указан временной период.",
                question="За какой период нужны данные или исследование?",
                default_assumption=_default_containing(intent, ["период", "последн", "год"]),
            )
        )

    if not intent.indicators and not intent.indicator_specs:
        requests.append(
            ClarificationRequest(
                field="indicators",
                reason="Не определены показатели для анализа.",
                question="Какие показатели или метрики нужно использовать?",
                default_assumption=_default_containing(intent, ["показател", "метрик"]),
            )
        )

    if _requires_indicator_methodology(intent):
        requests.append(
            ClarificationRequest(
                field="indicator_specs",
                reason="Для одного или нескольких показателей не хватает определения или единицы измерения.",
                question="Какую методику и единицы измерения использовать для ключевых показателей?",
                default_assumption=_default_containing(intent, ["метод", "единиц", "ипц", "%"]),
            )
        )

    if _requires_frequency(intent) and not intent.frequency:
        requests.append(
            ClarificationRequest(
                field="frequency",
                reason="Не указана частота наблюдений.",
                question="Какая частота данных нужна: годовая, квартальная или месячная?",
                default_assumption=_default_containing(intent, ["частот", "годовая", "месячная"]),
            )
        )

    if _requires_dataset_spec(intent) and _has_incomplete_dataset_spec(intent):
        requests.append(
            ClarificationRequest(
                field="dataset_spec",
                reason="Не полностью определена структура целевого датасета.",
                question="Какую зернистость строк и ключевые столбцы ожидаете в итоговом датасете?",
                default_assumption=_default_containing(intent, ["зернист", "строк", "датасет"]),
            )
        )

    if intent.intent_type == IntentType.RESEARCH and not intent.research_questions:
        requests.append(
            ClarificationRequest(
                field="research_questions",
                reason="Для исследовательской задачи не сформулированы исследовательские вопросы.",
                question="Какие исследовательские вопросы нужно проверить?",
                default_assumption=_default_containing(intent, ["вопрос", "связ", "зависим"]),
            )
        )

    if intent.intent_type == IntentType.DERIVED and not intent.derived_metrics:
        requests.append(
            ClarificationRequest(
                field="derived_metrics",
                reason="Запрос требует производную метрику, но формула не определена.",
                question="Какую формулу или базу нормализации использовать для производной метрики?",
                default_assumption=_default_containing(intent, ["формул", "баз", "нормал"]),
            )
        )

    return _deduplicate_requests(requests)


def _answers_to_json(answers: list[ClarificationAnswer]) -> str:
    return "[\n" + ",\n".join(
        answer.model_dump_json(indent=2, ensure_ascii=False) for answer in answers
    ) + "\n]"


def _requests_from_model_questions(intent: ResearchIntent) -> list[ClarificationRequest]:
    requests = []
    ambiguities = intent.ambiguities or ["Запрос требует уточнения."]

    for index, question in enumerate(intent.clarifying_questions):
        field = _guess_field(
            question,
            ambiguities[index] if index < len(ambiguities) else "",
        )
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
    if not intent.indicator_specs:
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
    if any(token in text for token in ["частота", "месяч", "квартал", "годовая"]):
        return "frequency"
    if any(token in text for token in ["метод", "методик", "формул"]):
        return "methodology"
    if any(token in text for token in ["страна", "регион", "географ", "территор"]):
        return "geography"
    if any(token in text for token in ["период", "год", "время", "диапазон"]):
        return "time_range"
    if any(token in text for token in ["показател", "метрик", "ипц", "инфляц"]):
        return "indicators"
    return "other"


def _default_at(intent: ResearchIntent, index: int) -> str | None:
    if index < len(intent.assumptions_if_no_answer):
        return intent.assumptions_if_no_answer[index]
    return None


def _default_containing(intent: ResearchIntent, tokens: list[str]) -> str | None:
    for assumption in intent.assumptions_if_no_answer:
        lower = assumption.lower()
        if any(token in lower for token in tokens):
            return assumption
    return None


def _default_for_field(intent: ResearchIntent, field: str) -> str | None:
    field_tokens = {
        "geography": ["географ", "страна", "регион"],
        "time_range": ["период", "диапазон", "последн"],
        "frequency": ["частот", "годовая", "месячная", "квартальная"],
        "indicators": ["показател", "метрик", "ипц", "инфляц"],
        "methodology": ["метод", "методик", "декабрь", "базов"],
        "dataset_spec": ["зернист", "строк", "датасет", "колон"],
        "derived_metrics": ["формул", "баз", "нормал"],
    }
    return _default_containing(intent, field_tokens.get(field, []))


def _deduplicate_requests(
    requests: list[ClarificationRequest],
) -> list[ClarificationRequest]:
    result: list[ClarificationRequest] = []
    seen_fields: set[str] = set()

    for request in requests:
        if request.field in seen_fields:
            continue
        seen_fields.add(request.field)
        result.append(request)

    return result
