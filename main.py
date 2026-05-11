import argparse
import json
import sys
from typing import Any

from openai import OpenAIError

from dataset_reranker import (
    DatasetRerankerError,
    build_explorer_handoff,
    rerank_dataset_candidates,
)
from hybrid_candidate_retriever import retrieve_candidate_datasets
from intent_parser import IntentParserError, ResearchIntent, parse_research_intent


DEFAULT_QUERY = "Нужен годовой ВВП России в текущих и постоянных ценах"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporary end-to-end RAG pipeline.")
    parser.add_argument("query", nargs="*", help="Natural-language research request.")
    parser.add_argument("--provider", choices=["qwen", "yandex"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--candidate-limit", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = " ".join(args.query).strip() or DEFAULT_QUERY

    try:
        result = run_pipeline(
            query=query,
            provider=args.provider,
            model=args.model,
            candidate_limit=args.candidate_limit,
            top_n=args.top_n,
        )
    except (
        RuntimeError,
        IntentParserError,
        DatasetRerankerError,
        ValueError,
        OpenAIError,
    ) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_pipeline(
    query: str,
    provider: str | None = None,
    model: str | None = None,
    candidate_limit: int = 30,
    top_n: int = 5,
) -> dict[str, Any]:
    print("этап 1: intent parsing", file=sys.stderr, flush=True)
    intent = parse_research_intent(query, provider=provider, model=model)

    print("этап 2: Chroma + BM25 retrieval", file=sys.stderr, flush=True)
    candidates = retrieve_candidate_datasets(
        intent,
        candidate_top_k=candidate_limit,
    )

    print("этап 3: LLM reranker", file=sys.stderr, flush=True)
    rerank_response = rerank_dataset_candidates(
        user_query=intent.original_query,
        candidates=candidates,
        provider=provider,
        model=model,
        top_n=top_n,
        candidate_limit=candidate_limit,
    )

    handoff = build_explorer_handoff(rerank_response, include_source=True)

    return {
        "query": query,
        "intent": _intent_payload(intent),
        "retrieval": {
            "candidate_count": len(candidates),
            "top_candidates": [_candidate_payload(candidate) for candidate in candidates[:10]],
        },
        "reranker": rerank_response.model_dump(mode="json"),
        "explorer_handoff": handoff,
        "evaluation": _evaluate_pipeline(candidates, handoff, rerank_response.no_results_reason),
    }


def _intent_payload(intent: ResearchIntent) -> dict[str, Any]:
    return {
        "schema_version": intent.schema_version,
        "original_query": intent.original_query,
        "english_query": intent.english_query,
        "keyword_synonyms": [
            item.model_dump(mode="json") for item in intent.keyword_synonyms
        ],
        "intent_type": intent.intent_type.value,
        "complexity": intent.complexity.value,
        "topic": intent.topic,
        "geography": intent.geography,
        "time_range": intent.time_range.model_dump(mode="json") if intent.time_range else None,
        "frequency": intent.frequency,
        "indicators": intent.indicators,
        "next_action": intent.next_action.value,
        "confidence": intent.confidence,
    }


def _candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": candidate.get("record_id"),
        "dataset_id": candidate.get("dataset_id"),
        "title": candidate.get("title"),
        "source": candidate.get("source"),
        "matched_by": candidate.get("matched_by", []),
        "data_path": candidate.get("data_path"),
    }


def _evaluate_pipeline(
    candidates: list[dict[str, Any]],
    handoff: list[dict[str, str | None]],
    no_results_reason: str | None,
) -> dict[str, Any]:
    return {
        "retrieval_returned_candidates": bool(candidates),
        "reranker_returned_results": bool(handoff),
        "ready_for_explorer": bool(handoff),
        "no_results_reason": no_results_reason,
    }


if __name__ == "__main__":
    main()
