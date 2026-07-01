import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from dataset_reranker import DatasetRerankResponse, DatasetRerankResult, RerankLevel
from intent_parser import ResearchIntent
from mcp_client import connect_and_run


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def parse_intent_via_mcp(query: str) -> ResearchIntent:
    result = call_mcp_tool("parse_intent", {"query": query})
    intent = result.get("intent")

    if not isinstance(intent, dict):
        raise RuntimeError(f"MCP parse_intent returned invalid payload: {result}")

    return ResearchIntent.model_validate(intent)


def retrieve_datasets_via_mcp(intent: ResearchIntent) -> DatasetRerankResponse:
    search_limit = int(os.getenv("MCP_SEARCH_LIMIT", "5"))
    search_threshold = float(os.getenv("MCP_SEARCH_THRESHOLD", "0.35"))

    search_result = call_mcp_tool(
        "search_catalog",
        {
            "intent": intent_for_mcp_search(intent),
            "limit": search_limit,
            "threshold": search_threshold,
        },
    )

    return search_result_to_rerank(search_result)


def call_mcp_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    mcp_url = os.getenv("MCP_URL")
    mcp_token = os.getenv("MCP_TOKEN")

    if not mcp_url:
        raise RuntimeError("MCP_URL is not set")
    if not mcp_token:
        raise RuntimeError("MCP_TOKEN is not set")

    return asyncio.run(
        connect_and_run(
            mcp_url,
            mcp_token,
            lambda client: client.call_tool(name, args),
        )
    )


def intent_for_mcp_search(intent: ResearchIntent) -> dict[str, Any]:
    payload = intent.model_dump(mode="json")
    payload["schema_version"] = "1.1"
    payload.pop("english_query", None)
    payload.pop("keyword_synonyms", None)
    return payload


def search_result_to_rerank(search_result: dict[str, Any]) -> DatasetRerankResponse:
    results = search_result.get("results") or []

    if not isinstance(results, list):
        return DatasetRerankResponse(
            results=[],
            no_results_reason=f"MCP search_catalog returned invalid payload: {search_result}",
        )

    rerank_results = [
        hit_to_rerank_result(hit)
        for hit in results
        if isinstance(hit, dict) and hit.get("code_or_id")
    ]

    if rerank_results:
        return DatasetRerankResponse(results=rerank_results)

    reason = search_result.get("error") or "MCP search_catalog не вернул подходящие датасеты."
    return DatasetRerankResponse(results=[], no_results_reason=str(reason))


def hit_to_rerank_result(hit: dict[str, Any]) -> DatasetRerankResult:
    source = str(hit.get("source") or "mcp").strip()
    code = str(hit.get("code_or_id") or "unknown").strip()
    name = str(hit.get("name") or "").strip() or None
    text_preview = str(hit.get("text_preview") or "").strip() or None
    unit = str(hit.get("unit") or "").strip() or None
    source_url = str(hit.get("url") or "").strip() or None
    parquet_path = str(hit.get("parquet_path") or "").strip()
    data_path = local_data_path(source, parquet_path)

    try:
        score = float(hit.get("score"))
    except (TypeError, ValueError):
        score = None

    relevance = RerankLevel.MEDIUM
    if score is not None and score >= 0.85:
        relevance = RerankLevel.HIGH
    elif score is not None and score < 0.70:
        relevance = RerankLevel.LOW

    limitations = []
    if not parquet_path:
        limitations.append("MCP search_catalog не вернул parquet_path для локального SQL executor.")
    elif data_path and not Path(data_path).exists():
        limitations.append(f"Локальный parquet пока не найден: {data_path}")

    why_matched = "Датасет выбран через MCP search_catalog."
    if score is not None:
        why_matched = f"Датасет выбран через MCP search_catalog, score={score:.3f}."

    return DatasetRerankResult(
        record_id=f"{source}:{code}",
        dataset_id=code,
        title=name,
        source=source,
        description=text_preview,
        tags=[],
        unit=unit,
        frequency=frequency_from_topics(hit.get("topics")),
        data_path=data_path,
        source_url=source_url,
        relevance=relevance,
        usefulness_confidence=RerankLevel.MEDIUM,
        why_matched=why_matched,
        possible_limitations=limitations,
    )


def local_data_path(source: str, parquet_path: str) -> str | None:
    if not parquet_path:
        return None

    filename = Path(parquet_path).name
    source_key = source.lower()
    dumps_dir = ROOT / "data" / "dumps"

    if source_key in {"fedstatru", "fedstat"}:
        return str(dumps_dir / "fedstatru" / "fedstatru" / "data" / "parquet" / filename)
    if source_key in {"wb", "world_bank", "worldbank"}:
        return str(dumps_dir / "wb" / "wb" / "parquet" / filename)

    return str(dumps_dir / source_key / "parquet" / filename)


def frequency_from_topics(topics: Any) -> str | None:
    text = str(topics or "").lower()

    if "годовая" in text:
        return "годовая"
    if "квартальная" in text:
        return "квартальная"
    if "месячная" in text:
        return "месячная"
    return None
