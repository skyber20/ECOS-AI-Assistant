from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intent_parser import (
    IntentType,
    IntentParserError,
    LLMSettings,
    NextAction,
    ResearchIntent,
    _create_json_completion,
    _parse_intent_json,
    create_llm_settings,
    parse_research_intent,
)
from research_designer import ResearchStudyDesign, design_research


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
    message: str | None = None

    @field_validator("clarification_requests", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return [] if value is None else value


REFINE_INTENT_PROMPT = """Ты обновляешь JSON первого этапа ResearchIntent после уточнений пользователя.

На входе:
1. предыдущий ResearchIntent;
2. вопросы системы и ответы пользователя.

Правила:
- Верни только полный валидный JSON ResearchIntent без Markdown.
- Не меняй смысл запроса без необходимости.
- Обнови поля, к которым относятся ответы: english_query, keyword_synonyms, topic, objects, geography, time_range, frequency, indicators, indicator_specs, entities, granularity, dataset_spec, research_questions, derived_metrics.
- Пересобери english_query и keyword_synonyms, если уточнения пользователя изменили смысл запроса.
- Удали закрытые ambiguities и clarifying_questions.
- Если все блокирующие уточнения закрыты, next_action = "proceed_with_assumptions".
- Если что-то все еще неясно, оставь next_action = "ask_clarification" и добавь новые clarifying_questions.
- assumptions_if_no_answer оставь только для оставшихся неуточненных полей.
- Не запускай дизайн исследования и не придумывай числовые значения данных.
"""


def run_research_flow(
    query: str,
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
    use_defaults: bool = False,
) -> OrchestrationResult:
    intent = parse_research_intent(
        query,
        provider=provider,
        model=model,
    )
    return continue_research_flow(
        intent,
        settings=settings,
        provider=provider,
        model=model,
        use_defaults=use_defaults,
    )


def continue_research_flow(
    intent: ResearchIntent,
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
    use_defaults: bool = False,
) -> OrchestrationResult:
    readiness = prepare_intent_for_design(intent, use_defaults=use_defaults)
    if readiness.status != OrchestrationStatus.READY_FOR_DESIGN:
        return readiness

    design = design_research(
        intent,
        settings=settings,
        provider=provider,
        model=model,
    )
    return OrchestrationResult(
        status=OrchestrationStatus.DESIGN_READY,
        intent=intent,
        clarification_requests=[] if use_defaults else readiness.clarification_requests,
        research_design=design,
    )


def refine_intent_with_clarifications(
    intent: ResearchIntent,
    answers: list[ClarificationAnswer],
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> ResearchIntent:
    if not answers:
        return intent

    llm_settings = settings or create_llm_settings(provider=provider, model=model)
    messages = build_refine_intent_messages(intent, answers)
    content = _create_json_completion(llm_settings, messages)

    try:
        return _parse_intent_json(content, original_query=intent.original_query)
    except IntentParserError as exc:
        raise RuntimeError(f"Failed to refine ResearchIntent after clarification: {exc}") from exc


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
