import json
from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from catalog_builder import ROOT
from intent_parser import (
    IntentParserError,
    LLMSettings,
    ResearchIntent,
    _create_json_completion,
    _load_json_object,
    create_llm_settings,
    none_to_empty_list,
)


CATALOG_RECORDS_PATH = ROOT / "data" / "catalog_records.jsonl"
LLM_CANDIDATE_LIMIT = 30
MIN_TOP_RESULTS = 3
MAX_TOP_RESULTS = 7

RERANK_CONTEXT_FIELDS = (
    "record_id",
    "dataset_id",
    "source",
    "title",
    "description",
    "long_description",
    "methodology",
    "limitations",
    "tags",
    "dimensions",
    "unit",
    "frequency",
    "source_name",
    "data_path",
    "source_url",
    "is_invalid",
)

OUTPUT_METADATA_FIELDS = (
    "record_id",
    "dataset_id",
    "title",
    "source",
    "description",
    "tags",
    "unit",
    "frequency",
    "data_path",
    "source_url",
)


class DatasetRerankerError(RuntimeError):
    pass


class RerankLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DatasetRerankResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    dataset_id: str | None = None
    title: str | None = None
    source: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    unit: str | None = None
    frequency: str | None = None
    data_path: str | None = None
    source_url: str | None = None
    relevance: RerankLevel
    usefulness_confidence: RerankLevel
    why_matched: str
    possible_limitations: list[str] = Field(default_factory=list)

    @field_validator("tags", "possible_limitations", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)

    @field_validator("relevance", "usefulness_confidence", mode="before")
    @classmethod
    def _normalize_level(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class RejectedSimilarCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    dataset_id: str | None = None
    title: str | None = None
    reason: str


class DatasetRerankResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[DatasetRerankResult] = Field(default_factory=list, max_length=MAX_TOP_RESULTS)
    rejected_similar_candidates: list[RejectedSimilarCandidate] = Field(default_factory=list)
    no_results_reason: str | None = None


class ExplorerDatasetHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    dataset_id: str | None = None
    title: str | None = None
    data_path: str | None = None
    source: str | None = None


RERANKER_PROMPT = """Ты LLM reranker для поиска датасетов по metadata.

На входе:
1. исходный user_query;
2. candidate datasets из retrieval;
3. metadata каждого candidate из catalog_records.jsonl.

Задача:
- Выбери только лучшие датасеты из переданного списка candidates.
- Верни top 3-7 datasets, максимум top_n.
- Если релевантных кандидатов меньше трех, верни только релевантных.
- Если среди candidates нет датасетов, которые по metadata подходят для ответа на user_query, верни пустой results и кратко объясни причину в no_results_reason.
- Не выбирай лучший из нерелевантных или слабых candidates только ради заполнения results.
- Возвращай только датасеты, которые напрямую отвечают user_query или являются явно необходимым входом для расчета запрошенного показателя.
- Не включай похожие, фоновые, сравнительные или противоположные показатели, если пользователь прямо не просил их сравнивать.
- Если есть 1-2 сильных кандидата, не добавляй слабые кандидаты для количества.
- Если среди candidates есть прямые совпадения, не добавляй частичные input-датасеты для ручного расчета.
- Не предлагай ручные пересчеты и формулы, если user_query прямо не просит derived/calculation.
- Не пиши, что показатель можно пересчитать, умножить, разделить, масштабировать или восстановить из другого показателя, если такой расчет не указан в metadata.
- Перед выбором каждого candidate проверь по metadata:
  1. это тот же показатель, а не похожий показатель;
  2. это нужный тип значения: absolute level, per capita, rate, percent, index, current prices, constant prices;
  3. unit не противоречит запросу;
  4. frequency не противоречит запросу;
  5. dimensions подходят под объект, географию и временную структуру запроса.
- Если user_query просит общий показатель, per capita, percent, rate и index должны быть rejected_similar_candidates, а не results.
- Если user_query просит per capita, percent, rate или index, абсолютный level должен быть rejected_similar_candidates, а не results.
- Если user_query просит current prices или constant prices, проверяй это по title, description, methodology, tags, unit и dimensions.
- Если candidate только помогает рассчитать нужный показатель, но сам им не является, не выбирай его, если user_query не просит расчетный показатель.
- Если прямого candidate нет, верни пустой results и объясни в no_results_reason, какие близкие candidates были отклонены.
- Используй только переданную metadata. Не придумывай поля, coverage, значения, колонки, периоды или географию.
- Не читай и не предполагай raw rows, parquet, clean_jsonl или observation dumps.
- Metadata-поля record_id, dataset_id, title, source, description, tags, unit, frequency, data_path, source_url сохраняй как в кандидате.
- Если metadata не хватает, пиши это в possible_limitations, но не используй ограничения как оправдание для нерелевантного выбора.
- why_matched должен кратко объяснять, какие metadata-поля совпали с запросом и почему датасет помогает ответить пользователю.
- possible_limitations пиши по-русски, кратко и предметно.
- rejected_similar_candidates заполняй только для близких, но отклоненных candidates: например GDP per capita вместо GDP, rate вместо level, rural/urban вместо total, index вместо absolute value.
- В rejected_similar_candidates.reason кратко укажи конкретное metadata-расхождение.
- relevance: "high", "medium" или "low", насколько metadata напрямую соответствует запросу.
- usefulness_confidence: "high", "medium" или "low", насколько уверенно по metadata можно использовать датасет для user_query.
- Верни только валидный JSON без Markdown и лишних ключей.
"""


def retrieve_and_rerank_datasets(
    user_query: str | ResearchIntent,
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
    top_n: int = MAX_TOP_RESULTS,
    candidate_limit: int = LLM_CANDIDATE_LIMIT,
) -> DatasetRerankResponse:
    from hybrid_candidate_retriever import retrieve_candidate_datasets

    candidates = retrieve_candidate_datasets(
        user_query,
        candidate_top_k=candidate_limit,
    )
    query = (
        user_query.original_query
        if isinstance(user_query, ResearchIntent)
        else user_query
    )
    return rerank_dataset_candidates(
        user_query=query,
        candidates=candidates,
        settings=settings,
        provider=provider,
        model=model,
        top_n=top_n,
        candidate_limit=candidate_limit,
    )


def rerank_dataset_candidates(
    user_query: str,
    candidates: list[dict[str, Any]],
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
    top_n: int = MAX_TOP_RESULTS,
    candidate_limit: int = LLM_CANDIDATE_LIMIT,
) -> DatasetRerankResponse:
    query = _clean_query(user_query)
    _validate_limits(top_n, candidate_limit)
    candidate_contexts = prepare_rerank_candidates(candidates, candidate_limit=candidate_limit)
    if not candidate_contexts:
        return DatasetRerankResponse(
            results=[],
            no_results_reason="Retrieval не вернул candidates с metadata из catalog_records.jsonl.",
        )

    llm_settings = settings or create_llm_settings(provider=provider, model=model)
    messages = build_dataset_rerank_messages(query, candidate_contexts, top_n=top_n)
    content = _create_json_completion(llm_settings, messages)
    response = _parse_rerank_json(content)
    records_by_id = {candidate["record_id"]: candidate for candidate in candidate_contexts}
    return _hydrate_response(response, records_by_id, top_n=top_n)


def build_explorer_handoff(
    rerank_response: DatasetRerankResponse | list[DatasetRerankResult],
    include_source: bool = False,
) -> list[dict[str, str | None]]:
    results = (
        rerank_response.results
        if isinstance(rerank_response, DatasetRerankResponse)
        else rerank_response
    )
    handoff: list[dict[str, str | None]] = []

    for result in results:
        if not result.data_path:
            continue
        item = ExplorerDatasetHandoff(
            record_id=result.record_id,
            dataset_id=result.dataset_id,
            title=result.title,
            data_path=result.data_path,
            source=result.source if include_source else None,
        )
        handoff.append(item.model_dump(mode="json", exclude_none=True))

    return handoff


def prepare_rerank_candidates(
    candidates: list[dict[str, Any]],
    candidate_limit: int = LLM_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    records = _catalog_records()
    contexts: list[dict[str, Any]] = []
    seen: set[str] = set()

    for candidate in candidates:
        record_id = _clean_optional_text(candidate.get("record_id"))
        if not record_id or record_id in seen:
            continue
        record = records.get(record_id)
        if not record:
            continue
        contexts.append(_record_context(record))
        seen.add(record_id)
        if len(contexts) >= candidate_limit:
            break

    return contexts


def build_dataset_rerank_messages(
    user_query: str,
    candidate_contexts: list[dict[str, Any]],
    top_n: int = MAX_TOP_RESULTS,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"{RERANKER_PROMPT}\n\n"
                "JSON Schema:\n"
                f"{json.dumps(DatasetRerankResponse.model_json_schema(), ensure_ascii=False, indent=2)}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"top_n: {top_n}\n\n"
                "=== USER QUERY ===\n"
                f"{user_query}\n\n"
                "=== CANDIDATE DATASETS ===\n"
                f"{json.dumps(candidate_contexts, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def _parse_rerank_json(content: str) -> DatasetRerankResponse:
    try:
        data = _load_json_object(content)
    except (IntentParserError, json.JSONDecodeError) as exc:
        raise DatasetRerankerError(f"LLM returned invalid rerank JSON: {exc}") from exc

    try:
        return DatasetRerankResponse.model_validate(data, extra="ignore")
    except ValidationError as exc:
        raise DatasetRerankerError(f"LLM JSON does not match DatasetRerankResponse schema: {exc}") from exc


def _hydrate_response(
    response: DatasetRerankResponse,
    records_by_id: dict[str, dict[str, Any]],
    top_n: int,
) -> DatasetRerankResponse:
    results: list[DatasetRerankResult] = []
    rejected = _hydrate_rejected_candidates(response.rejected_similar_candidates, records_by_id)
    seen: set[str] = set()

    for result in response.results:
        if result.record_id in seen:
            continue
        record = records_by_id.get(result.record_id)
        if not record or record.get("is_invalid"):
            continue
        payload = result.model_dump(mode="json")
        for field in OUTPUT_METADATA_FIELDS:
            payload[field] = _output_metadata_value(record.get(field), field)
        payload["possible_limitations"] = _merge_limitations(
            payload.get("possible_limitations"),
            _metadata_limitations(record),
        )
        results.append(DatasetRerankResult.model_validate(payload))
        seen.add(result.record_id)
        if len(results) >= top_n:
            break

    if not results:
        return DatasetRerankResponse(
            results=[],
            rejected_similar_candidates=rejected,
            no_results_reason=response.no_results_reason
            or "Среди переданных retrieval-candidates не найдено релевантных датасетов.",
        )
    return DatasetRerankResponse(
        results=results,
        rejected_similar_candidates=rejected,
        no_results_reason=None,
    )


def _hydrate_rejected_candidates(
    rejected: list[RejectedSimilarCandidate],
    records_by_id: dict[str, dict[str, Any]],
) -> list[RejectedSimilarCandidate]:
    result: list[RejectedSimilarCandidate] = []
    seen: set[str] = set()

    for item in rejected[:10]:
        if item.record_id in seen:
            continue
        record = records_by_id.get(item.record_id)
        if not record:
            continue
        payload = item.model_dump(mode="json")
        payload["dataset_id"] = _empty_to_none(record.get("dataset_id"))
        payload["title"] = _empty_to_none(record.get("title"))
        result.append(RejectedSimilarCandidate.model_validate(payload))
        seen.add(item.record_id)

    return result


def _record_context(record: dict[str, Any]) -> dict[str, Any]:
    return {
        field: _catalog_value(record.get(field), field)
        for field in RERANK_CONTEXT_FIELDS
    }


def _catalog_value(value: Any, field: str) -> Any:
    if field in {"tags", "dimensions"}:
        return value if isinstance(value, list) else []
    return _empty_to_none(value)


def _output_metadata_value(value: Any, field: str) -> Any:
    if field == "tags":
        return value if isinstance(value, list) else []
    return _empty_to_none(value)


def _clean_query(user_query: str) -> str:
    query = " ".join(user_query.split())
    if not query:
        raise ValueError("user_query is empty")
    return query


def _validate_limits(top_n: int, candidate_limit: int) -> None:
    if not MIN_TOP_RESULTS <= top_n <= MAX_TOP_RESULTS:
        raise ValueError(f"top_n must be between {MIN_TOP_RESULTS} and {MAX_TOP_RESULTS}.")
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive.")


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _empty_to_none(value: Any) -> Any:
    return None if value == "" else value


def _merge_limitations(current: Any, additions: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    values = current if isinstance(current, list) else []
    for value in [*values, *additions]:
        text = _clean_optional_text(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _metadata_limitations(record: dict[str, Any]) -> list[str]:
    limitations: list[str] = []
    if not _clean_optional_text(record.get("data_path")):
        limitations.append("В metadata нет локального data_path.")
    if not _clean_optional_text(record.get("source_url")):
        limitations.append("В metadata нет source_url.")
    return limitations


@lru_cache(maxsize=1)
def _catalog_records() -> dict[str, dict[str, Any]]:
    with CATALOG_RECORDS_PATH.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    return {record["record_id"]: record for record in records}
