import json
import os
from functools import lru_cache
from typing import Any, Iterable

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from catalog_bm25_indexer import DEFAULT_INDEX_PATH, search_bm25
from catalog_builder import ROOT
from catalog_chroma_indexer import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_MODEL,
    QUERY_INSTRUCTION,
)
from intent_parser import ResearchIntent


VECTOR_TOP_K = 15
BM25_TOP_K = 15
CANDIDATE_TOP_K = 30
CATALOG_RECORDS_PATH = ROOT / "data" / "catalog_records.jsonl"
CANDIDATE_FIELDS = (
    "record_id",
    "dataset_id",
    "title",
    "source",
    "data_path",
    "source_url",
    "description",
    "long_description",
    "methodology",
    "limitations",
    "tags",
    "dimensions",
    "unit",
    "frequency",
    "source_name",
    "is_invalid",
)


def retrieve_candidate_datasets(
    user_query: str | ResearchIntent,
    vector_top_k: int = VECTOR_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    candidate_top_k: int = CANDIDATE_TOP_K,
) -> list[dict[str, Any]]:
    vector_queries, bm25_queries = _retrieval_queries(user_query)
    if not CATALOG_RECORDS_PATH.exists():
        return []
    vector_results = _search_vector_many(vector_queries, vector_top_k)
    bm25_results = _search_bm25_many(bm25_queries, bm25_top_k)
    return _merge_candidates(vector_results, bm25_results)[:candidate_top_k]


def _retrieval_queries(user_query: str | ResearchIntent) -> tuple[list[str], list[str]]:
    if isinstance(user_query, ResearchIntent):
        original_query = _clean_query(user_query.original_query)
        vector_queries = _dedupe_queries([original_query, user_query.english_query])
        bm25_queries = _dedupe_queries(
            [
                original_query,
                user_query.english_query,
                *_keyword_synonym_queries(user_query),
            ]
        )
        return vector_queries, bm25_queries

    query = _clean_query(user_query)
    return [query], [query]


def _keyword_synonym_queries(intent: ResearchIntent) -> list[str]:
    queries: list[str] = []
    for item in intent.keyword_synonyms:
        terms = _dedupe_queries([item.english_keyword, *item.synonyms])
        if terms:
            queries.append(" ".join(terms))
    return queries


def _search_vector_many(queries: list[str], top_k: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for query in queries:
        results.extend(_search_vector(query, top_k))
    return results


def _search_vector(query: str, top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0 or not CHROMA_DIR.exists():
        return []

    try:
        import chromadb

        embedding = _embedding_model().encode(
            [f"{QUERY_INSTRUCTION}{query}"],
            normalize_embeddings=True,
        )[0].tolist()
        collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(
            COLLECTION_NAME,
        )
        result = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
    except Exception:
        return []
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        {
            **dict(metadata or {}),
            "vector_score": distance,
        }
        for metadata, distance in zip(metadatas, distances)
    ]


def _search_bm25_many(queries: list[str], top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0 or not DEFAULT_INDEX_PATH.exists():
        return []

    results: list[dict[str, Any]] = []
    for query in queries:
        results.extend(search_bm25(query, top_k=top_k))
    return results


def _merge_candidates(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}

    for item in vector_results:
        _upsert_candidate(candidates, item, "vector")
    for item in bm25_results:
        _upsert_candidate(candidates, item, "bm25")

    return list(candidates.values())


def _upsert_candidate(
    candidates: dict[str, dict[str, Any]],
    item: dict[str, Any],
    source: str,
) -> None:
    record_id = item.get("record_id")
    if not record_id:
        return

    candidate = candidates.setdefault(record_id, _candidate_from(item))
    if source not in candidate["matched_by"]:
        candidate["matched_by"].append(source)
    score_field = f"{source}_score"
    if score_field in item:
        candidate[score_field] = item[score_field]


def _candidate_from(item: dict[str, Any]) -> dict[str, Any]:
    record = _catalog_records().get(item["record_id"], item)
    candidate = {field: _empty_to_none(record.get(field)) for field in CANDIDATE_FIELDS}
    candidate["matched_by"] = []
    return candidate


def _clean_query(user_query: str) -> str:
    query = _clean_optional_query(user_query)
    if not query:
        raise ValueError("user_query is empty")
    return query


def _dedupe_queries(queries: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = _clean_optional_query(query)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        result.append(cleaned)
        seen.add(key)
    return result


def _clean_optional_query(value: Any) -> str | None:
    if value is None:
        return None
    query = " ".join(str(value).split())
    return query or None


def _empty_to_none(value: Any) -> Any:
    return None if value == "" else value


@lru_cache(maxsize=1)
def _catalog_records() -> dict[str, dict[str, Any]]:
    with CATALOG_RECORDS_PATH.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    return {record["record_id"]: record for record in records}


@lru_cache(maxsize=1)
def _embedding_model() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        EMBEDDING_MODEL,
        cache_folder=str(EMBEDDING_CACHE_DIR),
        local_files_only=True,
    )
