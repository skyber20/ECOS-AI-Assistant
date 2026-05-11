import json
import os
from functools import lru_cache
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import chromadb
from sentence_transformers import SentenceTransformer

from catalog_bm25_indexer import search_bm25
from catalog_builder import ROOT
from catalog_chroma_indexer import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_MODEL,
    QUERY_INSTRUCTION,
)


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
    user_query: str,
    vector_top_k: int = VECTOR_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    candidate_top_k: int = CANDIDATE_TOP_K,
) -> list[dict[str, Any]]:
    query = _clean_query(user_query)
    vector_results = _search_vector(query, vector_top_k)
    bm25_results = search_bm25(query, top_k=bm25_top_k)
    return _merge_candidates(vector_results, bm25_results)[:candidate_top_k]


def _search_vector(query: str, top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []

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
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        {
            **dict(metadata or {}),
            "vector_score": distance,
        }
        for metadata, distance in zip(metadatas, distances)
    ]


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
    query = " ".join(user_query.split())
    if not query:
        raise ValueError("user_query is empty")
    return query


def _empty_to_none(value: Any) -> Any:
    return None if value == "" else value


@lru_cache(maxsize=1)
def _catalog_records() -> dict[str, dict[str, Any]]:
    with CATALOG_RECORDS_PATH.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    return {record["record_id"]: record for record in records}


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    return SentenceTransformer(
        EMBEDDING_MODEL,
        cache_folder=str(EMBEDDING_CACHE_DIR),
        local_files_only=True,
    )
