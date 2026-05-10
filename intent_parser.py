import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


FREQUENCY_ALIASES = {
    "annual": "годовая",
    "annually": "годовая",
    "yearly": "годовая",
    "year": "годовая",
    "monthly": "месячная",
    "month": "месячная",
    "quarterly": "квартальная",
    "quarter": "квартальная",
}


def normalize_frequency(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    normalized = value.strip()
    return FREQUENCY_ALIASES.get(normalized.lower(), normalized)


def none_to_empty_list(value: Any) -> Any:
    if value is None:
        return []
    return value


class NextAction(str, Enum):
    ASK_CLARIFICATION = "ask_clarification"
    PROCEED_WITH_ASSUMPTIONS = "proceed_with_assumptions"
    REPORT_NO_DATA = "report_no_data"
    UNSUPPORTED = "unsupported"


class IntentType(str, Enum):
    SIMPLE_DATA = "simple_data"
    COMPARATIVE = "comparative"
    RESEARCH = "research"
    DERIVED = "derived"
    AMBIGUOUS = "ambiguous"
    NO_DATA = "no_data"
    UNSUPPORTED = "unsupported"


class Complexity(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    COMPLEX = "complex"


class DataAvailabilityStatus(str, Enum):
    LIKELY_AVAILABLE = "likely_available"
    NEEDS_SOURCE_CHECK = "needs_source_check"
    LIKELY_UNAVAILABLE = "likely_unavailable"
    UNKNOWN = "unknown"


class LLMMode(str, Enum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES_PROMPT = "responses_prompt"


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    client: OpenAI
    mode: LLMMode
    model: str | None = None
    prompt_id: str | None = None


class TimeRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str | None = Field(
        default=None,
        description="Time period exactly as stated or inferred from the query.",
    )
    start_year: int | None = Field(default=None, description="Start year, if known.")
    end_year: int | None = Field(default=None, description="End year, if known.")
    is_explicit: bool = Field(
        default=False,
        description="True when the user explicitly specified the period.",
    )


class IndicatorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    definition: str | None = Field(
        default=None,
        description="Exact methodological definition when it matters.",
    )
    unit: str | None = None
    role: str | None = Field(
        default=None,
        description="For example primary, control, input, denominator, grouping.",
    )


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_grain: str | None = Field(
        default=None,
        description="Expected row grain: year, country-year, region-year, product-country-year.",
    )
    columns: list[str] = Field(default_factory=list)
    rows_approx: str | None = Field(
        default=None,
        description="Approximate expected row count or range as text.",
    )
    frequency: str | None = Field(default=None, description="Annual, monthly, quarterly, etc.")

    @field_validator("columns", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)

    @field_validator("frequency", mode="before")
    @classmethod
    def _normalize_frequency(cls, value: Any) -> Any:
        return normalize_frequency(value)


class SourceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dataset_or_indicator: str | None = None
    role: str | None = None
    notes: str | None = None


class ResearchDesign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypotheses: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    grouping: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    expected_visuals: list[str] = Field(default_factory=list)

    @field_validator(
        "hypotheses",
        "methods",
        "grouping",
        "controls",
        "expected_visuals",
        mode="before",
    )
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class DerivedMetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    inputs: list[str] = Field(default_factory=list)
    formula: str | None = None
    base_year: int | None = None
    normalization: str | None = None
    aggregation_rules: list[str] = Field(default_factory=list)

    @field_validator("inputs", "aggregation_rules", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class DataAvailabilityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DataAvailabilityStatus = DataAvailabilityStatus.UNKNOWN
    verdict: str | None = None
    reasons: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)

    @field_validator("reasons", "alternatives", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class ResearchIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.1"
    original_query: str
    intent_type: IntentType
    complexity: Complexity
    topic: str | None = None
    objects: list[str] = Field(
        default_factory=list,
        description="Main objects of observation: countries, regions, sectors, firms, etc.",
    )
    geography: list[str] = Field(default_factory=list)
    time_range: TimeRange | None = None
    frequency: str | None = Field(default=None, description="Annual, monthly, quarterly, etc.")

    @field_validator("frequency", mode="before")
    @classmethod
    def _normalize_frequency(cls, value: Any) -> Any:
        return normalize_frequency(value)

    disciplinary_perspective: str | None = Field(
        default=None,
        description="Economic, demographic, trade, labor market, or another research lens.",
    )
    indicators: list[str] = Field(default_factory=list)
    indicator_specs: list[IndicatorSpec] = Field(default_factory=list)
    entities: list[str] = Field(
        default_factory=list,
        description="Countries, regions, industries, products, organizations, or other named entities.",
    )
    granularity: str | None = Field(
        default=None,
        description="Expected row grain, for example country-year, region-year, product-country-year.",
    )
    research_questions: list[str] = Field(default_factory=list)
    research_design: ResearchDesign | None = None
    derived_metrics: list[DerivedMetricSpec] = Field(default_factory=list)
    dataset_spec: DatasetSpec | None = None
    source_candidates: list[SourceCandidate] = Field(default_factory=list)
    data_availability: DataAvailabilityAssessment = Field(
        default_factory=DataAvailabilityAssessment,
    )
    ambiguities: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    assumptions_if_no_answer: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    next_action: NextAction

    @field_validator(
        "objects",
        "geography",
        "indicators",
        "indicator_specs",
        "entities",
        "research_questions",
        "derived_metrics",
        "source_candidates",
        "ambiguities",
        "clarifying_questions",
        "assumptions_if_no_answer",
        mode="before",
    )
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


SYSTEM_PROMPT = """Ты парсер исследовательского намерения для системы сборки социально-экономических датасетов.

Твоя задача: преобразовать пользовательский запрос на естественном языке в один JSON-объект ResearchIntent.

Правила:
- Отвечай только валидным JSON без Markdown и пояснений.
- Возвращай только поля, описанные ниже. Не добавляй лишние ключи.
- Не выдумывай точные значения, которых нет в запросе. Если поле неизвестно, используй null или пустой список.
- Не подгоняй ответ под известные примеры. Классифицируй по общим признакам задачи.
- Если запрос неоднозначен, добавь 2-3 самых важных уточняющих вопроса.
- Даже если нужны уточнения, добавь assumptions_if_no_answer: разумные допущения, с которыми система сможет продолжить, если пользователь не ответит.
- Если запрос похож на данные, которых нет в открытых верифицированных источниках, не придумывай цифры: выставь intent_type=no_data, data_availability.status=likely_unavailable и next_action=report_no_data.
- Если пользователь просит связь, влияние, зависимость, факторы или объяснение явления, это research: нужны гипотезы, метод, контрольные переменные и структура датасета.
- Если пользователь просит индекс, реальные значения, поправку на инфляцию, нормализацию, базу = 100 или расчет по формуле, это derived: нужна derived_metrics с формулой.
- Если пользователь сравнивает несколько объектов по одному показателю, это comparative.
- Если пользователь просит один понятный показатель по одному объекту во времени, это simple_data.
- Если критически не хватает страны/региона, периода, частоты или методики показателя, это ambiguous и next_action=ask_clarification.
- next_action=ask_clarification используй только для блокирующих неоднозначностей. Если есть разумный статистический default, продолжай с assumptions_if_no_answer.
- Для конкретных запросов с понятной географией, периодом и показателем next_action обычно proceed_with_assumptions, а clarifying_questions оставляй пустым.
- Для ИПЦ/инфляции с годовой частотой, если методика не уточнена, используй default "ИПЦ декабрь к декабрю предыдущего года" и явно запиши это в definition или assumptions_if_no_answer.
- Если частота не указана для официальных годовых социально-экономических рядов или индексов с базовым годом, используй default "годовая", а не месячная.
- Для индекса с базой N=100 без явного периода: start_year = N, end_year = null, frequency = "годовая", rows_approx = null или "с базового года до последнего доступного года".
- Формулы производных метрик записывай явно через переменные с индексами t и base. Если база = 100, формула должна гарантировать значение 100 в базовом году.
- Используй русские значения частоты: "годовая", "квартальная", "месячная", "точечное последнее доступное наблюдение", "панельные данные".
- complexity=easy: простой ряд одного показателя по одному объекту; complexity=medium: сравнение нескольких объектов или производная метрика; complexity=complex: исследование связи/факторов, панельные данные, много показателей, контрольные переменные или сложная доступность данных.
- В indicator_specs фиксируй точное определение показателя, если оно влияет на результат. Например: ИПЦ декабрь к декабрю, СКР против ОКР, доля НИОКР в ВВП по методике Frascati.
- В dataset_spec.columns для запросов на данные по возможности включай столбцы источника и даты выгрузки.
- В source_candidates предлагай вероятные источники и коды показателей, если они широко известны. Не выдавай источник как проверенный факт сбора данных; это кандидаты для следующего шага. Для российских официальных показателей часто уместны Росстат и ЕМИСС.
- Для no_data не добавляй нерелевантные источники "для вида"; лучше укажи крупные базы/организации, где такие данные обычно проверяются (например World Bank, IMF, ILO, национальная статистика), и объясни отсутствие структурированных данных.
- Используй русский язык в текстовых полях.

JSON-схема верхнего уровня:
{
  "schema_version": "1.1",
  "original_query": "string",
  "intent_type": "simple_data|comparative|research|derived|ambiguous|no_data|unsupported",
  "complexity": "easy|medium|complex",
  "topic": "string|null",
  "objects": ["string"],
  "geography": ["string"],
  "time_range": {
    "raw": "string|null",
    "start_year": 2015,
    "end_year": 2024,
    "is_explicit": true
  } | null,
  "frequency": "string|null",
  "disciplinary_perspective": "string|null",
  "indicators": ["string"],
  "indicator_specs": [
    {"name": "string", "definition": "string|null", "unit": "string|null", "role": "string|null"}
  ],
  "entities": ["string"],
  "granularity": "string|null",
  "research_questions": ["string"],
  "research_design": {
    "hypotheses": ["string"],
    "methods": ["string"],
    "grouping": ["string"],
    "controls": ["string"],
    "expected_visuals": ["string"]
  } | null,
  "derived_metrics": [
    {
      "name": "string",
      "inputs": ["string"],
      "formula": "string|null",
      "base_year": 2014,
      "normalization": "string|null",
      "aggregation_rules": ["string"]
    }
  ],
  "dataset_spec": {
    "row_grain": "string|null",
    "columns": ["string"],
    "rows_approx": "string|null",
    "frequency": "string|null"
  } | null,
  "source_candidates": [
    {"name": "string", "dataset_or_indicator": "string|null", "role": "string|null", "notes": "string|null"}
  ],
  "data_availability": {
    "status": "likely_available|needs_source_check|likely_unavailable|unknown",
    "verdict": "string|null",
    "reasons": ["string"],
    "alternatives": ["string"]
  },
  "ambiguities": ["string"],
  "clarifying_questions": ["string"],
  "assumptions_if_no_answer": ["string"],
  "confidence": 0.0,
  "next_action": "ask_clarification|proceed_with_assumptions|report_no_data|unsupported"
}
"""


class IntentParserError(RuntimeError):
    """Raised when the model response cannot be converted to ResearchIntent."""


def create_llm_settings(
    provider: str | None = None,
    model: str | None = None,
) -> LLMSettings:
    load_dotenv()

    selected_provider = (provider or _get_env("LLM_PROVIDER") or "qwen").lower()
    if selected_provider == "qwen":
        return _create_qwen_settings(model=model)
    if selected_provider == "yandex":
        return _create_yandex_settings(model=model)

    raise RuntimeError(
        "Unsupported LLM_PROVIDER. Use one of: qwen, yandex."
    )


def create_llm_client(provider: str | None = None) -> OpenAI:
    return create_llm_settings(provider=provider).client


def _create_qwen_settings(model: str | None = None) -> LLMSettings:
    api_key = _get_env("QWEN_API_KEY")
    base_url = _get_env("QWEN_BASE_URL")
    project = _get_env("QWEN_PROJECT")
    llm_model = model or _get_env("QWEN_MODEL") or "qwen3.5-122b"

    if not api_key:
        raise RuntimeError(
            "Qwen API key is not set. Add QWEN_API_KEY to your .env file."
        )

    client_kwargs: dict[str, str] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    if project:
        client_kwargs["project"] = project

    return LLMSettings(
        provider="qwen",
        client=OpenAI(**client_kwargs),
        mode=LLMMode.CHAT_COMPLETIONS,
        model=llm_model,
    )


def _create_yandex_settings(model: str | None = None) -> LLMSettings:
    api_key = _get_env("YANDEX_API_KEY")
    base_url = _get_env("YANDEX_BASE_URL") or "https://ai.api.cloud.yandex.net/v1"
    project = _get_env("YANDEX_PROJECT")
    prompt_id = _get_env("YANDEX_PROMPT_ID")
    llm_model = model or _get_env("YANDEX_MODEL")
    api_mode = (_get_env("YANDEX_API_MODE") or "").lower()

    if not api_key:
        raise RuntimeError("YANDEX_API_KEY is not set. Add it to your .env file.")

    client_kwargs: dict[str, str] = {"api_key": api_key, "base_url": base_url}
    if project:
        client_kwargs["project"] = project

    if api_mode in {"responses", "prompt", "responses_prompt"}:
        mode = LLMMode.RESPONSES_PROMPT
    elif api_mode in {"chat", "chat_completions"}:
        mode = LLMMode.CHAT_COMPLETIONS
    elif llm_model:
        mode = LLMMode.CHAT_COMPLETIONS
    else:
        mode = LLMMode.RESPONSES_PROMPT

    if mode == LLMMode.CHAT_COMPLETIONS and not llm_model:
        raise RuntimeError(
            "YANDEX_MODEL is required when YANDEX_API_MODE=chat."
        )
    if mode == LLMMode.RESPONSES_PROMPT and not prompt_id:
        raise RuntimeError(
            "YANDEX_PROMPT_ID is required when using Yandex responses_prompt mode."
        )

    return LLMSettings(
        provider="yandex",
        client=OpenAI(**client_kwargs),
        mode=mode,
        model=llm_model,
        prompt_id=prompt_id,
    )


def _get_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def parse_research_intent(
    query: str,
    client: OpenAI | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> ResearchIntent:
    if not query.strip():
        raise ValueError("Query must not be empty.")

    load_dotenv()

    settings = (
        LLMSettings(
            provider=provider or "custom",
            client=client,
            mode=LLMMode.CHAT_COMPLETIONS,
            model=model or _get_env("QWEN_MODEL") or "qwen3.5-122b",
        )
        if client
        else create_llm_settings(provider=provider, model=model)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    content = _create_json_completion(settings, messages)
    return _parse_intent_json(content, original_query=query)


def _create_json_completion(
    settings: LLMSettings,
    messages: list[dict[str, str]],
) -> str:
    if settings.mode == LLMMode.RESPONSES_PROMPT:
        return _create_responses_prompt_completion(settings, messages)

    if not settings.model:
        raise IntentParserError("LLM model is not configured.")

    request: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    try:
        response = settings.client.chat.completions.create(**request)
    except BadRequestError:
        request.pop("response_format", None)
        response = settings.client.chat.completions.create(**request)

    content = response.choices[0].message.content
    if not content:
        raise IntentParserError("LLM returned an empty response.")
    return content


def _create_responses_prompt_completion(
    settings: LLMSettings,
    messages: list[dict[str, str]],
) -> str:
    if not settings.prompt_id:
        raise IntentParserError("Yandex prompt id is not configured.")

    response = settings.client.responses.create(
        prompt={"id": settings.prompt_id},
        input=_messages_to_prompt_input(messages),
    )

    content = getattr(response, "output_text", None)
    if not content:
        raise IntentParserError("LLM returned an empty response.")
    return content


def _messages_to_prompt_input(messages: list[dict[str, str]]) -> str:
    system_message = next(
        (message["content"] for message in messages if message["role"] == "system"),
        "",
    )
    user_message = next(
        (message["content"] for message in messages if message["role"] == "user"),
        "",
    )

    return (
        f"{system_message}\n\n"
        f"Пользовательский запрос:\n{user_message}\n\n"
        "Верни только один JSON-объект по схеме выше."
    )


def _parse_intent_json(content: str, original_query: str) -> ResearchIntent:
    try:
        data = _load_json_object(content)
    except json.JSONDecodeError as exc:
        raise IntentParserError(f"LLM returned invalid JSON: {exc}") from exc

    if isinstance(data, dict) and not data.get("original_query"):
        data["original_query"] = original_query

    try:
        intent = ResearchIntent.model_validate(data)
    except ValidationError as exc:
        raise IntentParserError(f"LLM JSON does not match ResearchIntent schema: {exc}") from exc

    return _apply_deterministic_corrections(intent)


def _load_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise

        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(text[start:])

    if not isinstance(data, dict):
        raise IntentParserError("LLM returned JSON, but the root value is not an object.")

    return data


def _apply_deterministic_corrections(intent: ResearchIntent) -> ResearchIntent:
    has_blocking_ambiguity = bool(intent.ambiguities or intent.clarifying_questions)
    misses_core_scope = not intent.geography or (
        intent.time_range is None or not intent.time_range.is_explicit
    )

    if (
        intent.intent_type == IntentType.SIMPLE_DATA
        and has_blocking_ambiguity
        and misses_core_scope
    ):
        intent.intent_type = IntentType.AMBIGUOUS
        intent.complexity = Complexity.MEDIUM
        intent.next_action = NextAction.ASK_CLARIFICATION

    return intent
