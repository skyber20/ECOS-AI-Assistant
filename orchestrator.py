from enum import Enum
import warnings
from typing import Any, Literal, TypedDict

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.simplefilter("ignore", LangChainPendingDeprecationWarning)
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dataset_reranker import (
    DatasetRerankResponse,
    ExplorerDatasetHandoff,
    build_explorer_handoff,
    retrieve_and_rerank_datasets,
)
from dataset_structure import TargetDatasetStructure, build_target_dataset_structure
from intent_parser import (
    IntentType,
    LLMSettings,
    NextAction,
    ResearchIntent,
    SYSTEM_PROMPT,
    _create_json_completion,
    _parse_intent_json,
    create_llm_settings,
)
from research_designer import ResearchStudyDesign, design_research
from script_generator import (
    BuildScriptRun,
    GeneratedBuildScript,
    generate_and_run_build_script,
    generate_build_script,
)


class OrchestrationStatus(str, Enum):
    NEEDS_CLARIFICATION = "needs_clarification"
    READY_FOR_DESIGN = "ready_for_design"
    DESIGN_READY = "design_ready"
    BUILD_FAILED = "build_failed"
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
    dataset_rerank: DatasetRerankResponse | None = None
    explorer_datasets: list[ExplorerDatasetHandoff] = Field(default_factory=list)
    research_design: ResearchStudyDesign | None = None
    dataset_structure: TargetDatasetStructure | None = None
    build_script: GeneratedBuildScript | None = None
    build_run: BuildScriptRun | None = None
    message: str | None = None

    @field_validator("clarification_requests", "explorer_datasets", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return [] if value is None else value


GraphOperation = Literal["parse_intent", "run", "continue", "refine_intent"]


class ResearchAgentState(TypedDict, total=False):
    operation: GraphOperation
    query: str
    intent: ResearchIntent
    clarification_answers: list[ClarificationAnswer]
    settings: LLMSettings
    use_defaults: bool
    readiness: OrchestrationResult
    dataset_rerank: DatasetRerankResponse
    explorer_datasets: list[dict[str, str | None]]
    research_design: ResearchStudyDesign
    dataset_structure: TargetDatasetStructure
    build_script: GeneratedBuildScript
    build_run: BuildScriptRun
    result: OrchestrationResult


REFINE_INTENT_PROMPT = """Ты обновляешь JSON ResearchIntent после уточнений пользователя.

На входе:
1. предыдущий ResearchIntent;
2. вопросы системы и ответы пользователя.

Правила:
- Верни только полный валидный JSON ResearchIntent без Markdown.
- Не меняй смысл запроса без необходимости.
- Обнови только поля, к которым прямо относятся ответы пользователя.
- Удали закрытые ambiguities и clarifying_questions.
- Если все блокирующие уточнения закрыты, next_action = "proceed_with_assumptions".
- Если что-то все еще неясно, оставь next_action = "ask_clarification" и добавь новые clarifying_questions.
- assumptions_if_no_answer оставь только для оставшихся неуточненных полей.
- Не добавляй отдельные retrieval-запросы.
- Не запускай дизайн исследования и не придумывай числовые значения данных.
"""


class LangGraphResearchAgent:
    def __init__(
        self,
        settings: LLMSettings | None = None,
        provider: str | None = None,
        model: str | None = None,
        run_build_script: bool = True,
        build_output_dir: str = "artifacts/latest_run/generated_dataset",
        max_build_tries: int = 3,
    ) -> None:
        self.settings = settings or create_llm_settings(provider=provider, model=model)
        self.run_build_script = run_build_script
        self.build_output_dir = build_output_dir
        self.max_build_tries = max_build_tries
        self.graph = self._build_graph()

    def parse_intent(self, query: str) -> ResearchIntent:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        state = self._invoke({"operation": "parse_intent", "query": query})
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
            "explorer_datasets": [],
            **initial_state,
        }
        return self.graph.invoke(state)

    def _build_graph(self):
        graph = StateGraph(ResearchAgentState)
        graph.add_node("start", self._start_node)
        graph.add_node("parse_intent", self._parse_intent_node)
        graph.add_node("refine_intent", self._refine_intent_node)
        graph.add_node("validate_intent", self._validate_intent_node)
        graph.add_node("retrieve_datasets", self._retrieve_datasets_node)
        graph.add_node("design_research", self._design_research_node)

        graph.add_edge(START, "start")
        graph.add_conditional_edges(
            "start",
            self._route_start,
            {
                "parse_intent": "parse_intent",
                "refine_intent": "refine_intent",
                "validate_intent": "validate_intent",
            },
        )
        graph.add_conditional_edges(
            "parse_intent",
            self._route_after_parse,
            {
                "validate_intent": "validate_intent",
                END: END,
            },
        )
        graph.add_edge("refine_intent", END)
        graph.add_conditional_edges(
            "validate_intent",
            self._route_after_validate,
            {
                "retrieve_datasets": "retrieve_datasets",
                END: END,
            },
        )
        graph.add_conditional_edges(
            "retrieve_datasets",
            self._route_after_retrieval,
            {
                "design_research": "design_research",
                END: END,
            },
        )
        graph.add_edge("design_research", END)
        return graph.compile()

    def _start_node(self, state: ResearchAgentState) -> ResearchAgentState:
        return {}

    def _route_start(self, state: ResearchAgentState) -> str:
        operation = state.get("operation", "run")
        if operation == "refine_intent":
            return "refine_intent"
        if operation == "continue":
            return "validate_intent"
        return "parse_intent"

    def _route_after_parse(self, state: ResearchAgentState) -> str:
        if state.get("operation") == "parse_intent":
            return END
        return "validate_intent"

    def _route_after_validate(self, state: ResearchAgentState) -> str:
        if state["readiness"].status == OrchestrationStatus.READY_FOR_DESIGN:
            return "retrieve_datasets"
        return END

    def _route_after_retrieval(self, state: ResearchAgentState) -> str:
        if state.get("explorer_datasets"):
            return "design_research"
        return END

    def _parse_intent_node(self, state: ResearchAgentState) -> ResearchAgentState:
        query = state["query"]
        content = _create_json_completion(
            state["settings"],
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        return {
            "intent": _parse_intent_json(content, original_query=query),
        }

    def _refine_intent_node(self, state: ResearchAgentState) -> ResearchAgentState:
        content = _create_json_completion(
            state["settings"],
            build_refine_intent_messages(
                state["intent"],
                state.get("clarification_answers", []),
            ),
        )
        return {
            "intent": _parse_intent_json(
                content,
                original_query=state["intent"].original_query,
            ),
        }

    def _validate_intent_node(self, state: ResearchAgentState) -> ResearchAgentState:
        readiness = prepare_intent_for_design(
            state["intent"],
            use_defaults=state.get("use_defaults", False),
        )
        payload: ResearchAgentState = {"readiness": readiness}
        if readiness.status != OrchestrationStatus.READY_FOR_DESIGN:
            payload["result"] = readiness
        return payload

    def _retrieve_datasets_node(self, state: ResearchAgentState) -> ResearchAgentState:
        dataset_rerank = retrieve_and_rerank_datasets(
            state["intent"],
            settings=state["settings"],
        )
        explorer_datasets = build_explorer_handoff(dataset_rerank, include_source=True)
        payload: ResearchAgentState = {
            "dataset_rerank": dataset_rerank,
            "explorer_datasets": explorer_datasets,
        }
        if not explorer_datasets:
            payload["result"] = OrchestrationResult(
                status=OrchestrationStatus.NO_DATA,
                intent=state["intent"],
                clarification_requests=_result_clarifications(state),
                dataset_rerank=dataset_rerank,
                explorer_datasets=[],
                message=_rag_no_handoff_message(dataset_rerank),
            )
        return payload

    def _design_research_node(self, state: ResearchAgentState) -> ResearchAgentState:
        design = design_research(state["intent"], settings=state["settings"])
        dataset_structure = build_target_dataset_structure(state["intent"], design)
        build_run = None
        if self.run_build_script:
            build_script, build_run = generate_and_run_build_script(
                intent=state["intent"],
                design=design,
                structure=dataset_structure,
                dataset_rerank=state.get("dataset_rerank"),
                settings=state["settings"],
                output_dir=self.build_output_dir,
                max_tries=self.max_build_tries,
            )
        else:
            build_script = generate_build_script(
                intent=state["intent"],
                design=design,
                structure=dataset_structure,
                dataset_rerank=state.get("dataset_rerank"),
                settings=state["settings"],
            )
        status = OrchestrationStatus.DESIGN_READY
        message = "Сформированы дизайн исследования, структура датасета и скрипт сборки."
        if build_run is not None and build_run.status == "failed":
            status = OrchestrationStatus.BUILD_FAILED
            message = (
                f"Скрипт сборки не удалось довести до успешного запуска за {len(build_run.attempts)} "
                f"попыток. Последняя ошибка: {build_run.final_error}"
            )
        payload: ResearchAgentState = {
            "research_design": design,
            "dataset_structure": dataset_structure,
            "build_script": build_script,
            "build_run": build_run,
            "result": OrchestrationResult(
                status=status,
                intent=state["intent"],
                clarification_requests=_result_clarifications(state),
                dataset_rerank=state.get("dataset_rerank"),
                explorer_datasets=state.get("explorer_datasets", []),
                research_design=design,
                dataset_structure=dataset_structure,
                build_script=build_script,
                build_run=build_run,
                message=message,
            ),
        }
        return payload


def run_research_flow(
    query: str,
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
    use_defaults: bool = False,
    run_build_script: bool = True,
    build_output_dir: str = "artifacts/latest_run/generated_dataset",
    max_build_tries: int = 3,
) -> OrchestrationResult:
    return LangGraphResearchAgent(
        settings=settings,
        provider=provider,
        model=model,
        run_build_script=run_build_script,
        build_output_dir=build_output_dir,
        max_build_tries=max_build_tries,
    ).run(query, use_defaults=use_defaults)


def continue_research_flow(
    intent: ResearchIntent,
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
    use_defaults: bool = False,
    run_build_script: bool = True,
    build_output_dir: str = "artifacts/latest_run/generated_dataset",
    max_build_tries: int = 3,
) -> OrchestrationResult:
    return LangGraphResearchAgent(
        settings=settings,
        provider=provider,
        model=model,
        run_build_script=run_build_script,
        build_output_dir=build_output_dir,
        max_build_tries=max_build_tries,
    ).continue_from_intent(intent, use_defaults=use_defaults)


def refine_intent_with_clarifications(
    intent: ResearchIntent,
    answers: list[ClarificationAnswer],
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> ResearchIntent:
    return LangGraphResearchAgent(
        settings=settings,
        provider=provider,
        model=model,
    ).refine_intent(intent, answers)


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
                message="Нужно уточнить поля, для которых нет безопасного default assumption.",
            )

    return OrchestrationResult(
        status=OrchestrationStatus.READY_FOR_DESIGN,
        intent=intent,
        clarification_requests=[] if use_defaults else clarification_requests,
    )


def validate_intent_for_design(intent: ResearchIntent) -> list[ClarificationRequest]:
    if intent.next_action != NextAction.ASK_CLARIFICATION:
        return []
    return _requests_from_model_questions(intent)


def _requests_from_model_questions(intent: ResearchIntent) -> list[ClarificationRequest]:
    requests: list[ClarificationRequest] = []
    ambiguities = intent.ambiguities or []

    for index, question in enumerate(intent.clarifying_questions):
        requests.append(
            ClarificationRequest(
                field=f"clarification_{index + 1}",
                reason=ambiguities[index] if index < len(ambiguities) else "Запрос требует уточнения.",
                question=question,
                default_assumption=_default_at(intent, index),
            )
        )

    return requests


def _result_clarifications(state: ResearchAgentState) -> list[ClarificationRequest]:
    readiness = state.get("readiness")
    if not readiness or state.get("use_defaults"):
        return []
    return readiness.clarification_requests


def _rag_no_handoff_message(rerank_response: DatasetRerankResponse | None) -> str:
    if rerank_response and rerank_response.no_results_reason:
        return rerank_response.no_results_reason
    if rerank_response and rerank_response.results:
        return "RAG нашел candidates, но у выбранных датасетов нет локального data_path для explorer."
    return "RAG не нашел релевантные датасеты для передачи в explorer."


def _answers_to_json(answers: list[ClarificationAnswer]) -> str:
    return "[\n" + ",\n".join(
        answer.model_dump_json(indent=2, ensure_ascii=False) for answer in answers
    ) + "\n]"


def _default_at(intent: ResearchIntent, index: int) -> str | None:
    if index < len(intent.assumptions_if_no_answer):
        return intent.assumptions_if_no_answer[index]
    return None
