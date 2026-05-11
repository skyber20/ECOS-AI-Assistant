import json
import os
from functools import lru_cache
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from catalog_bm25_indexer import search_bm25
from catalog_builder import CATALOG_RECORDS_PATH, ROOT
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

WORLD_BANK_ALIASES = {
    "gdp": (
        "wb:NY.GDP.MKTP.CD",
        "wb:NY.GDP.MKTP.KD",
        "wb:NY.GDP.MKTP.PP.CD",
        "wb:NY.GDP.MKTP.ZG",
    ),
    "inflation": (
        "wb:FP.CPI.TOTL.ZG",
        "wb:FP.CPI.TOTL",
    ),
}

GDP_MARKERS = {
    "ввп",
    "gdp",
    "валовой внутренний продукт",
    "gross domestic product",
}
INFLATION_MARKERS = {
    "инфляция",
    "инфляции",
    "инфляцию",
    "ипц",
    "cpi",
    "consumer price",
    "consumer prices",
}
INTERNATIONAL_MARKERS = {
    "сша",
    "usa",
    "us",
    "united states",
    "казахстан",
    "страны",
    "странам",
    "country",
    "countries",
    "мир",
    "world",
    "брики",
    "брикс",
    "ес",
    "eu",
}


def retrieve_candidate_datasets(
    user_query: str,
    vector_top_k: int = VECTOR_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    candidate_top_k: int = CANDIDATE_TOP_K,
) -> list[dict[str, Any]]:
    query = _clean_query(user_query)
    if not CATALOG_RECORDS_PATH.exists():
        return []
    alias_results = _search_alias_records(query)
    vector_results = _search_vector(query, vector_top_k)
    bm25_results = search_bm25(query, top_k=bm25_top_k)
    return _merge_candidates(query, alias_results, vector_results, bm25_results)[:candidate_top_k]


def _search_alias_records(query: str) -> list[dict[str, Any]]:
    records = _catalog_records()
    aliases: list[dict[str, Any]] = []
    for alias_group in _query_alias_groups(query):
        for rank, record_id in enumerate(WORLD_BANK_ALIASES[alias_group], start=1):
            record = records.get(record_id)
            if not record:
                continue
            aliases.append(
                {
                    **record,
                    "alias_rank": rank,
                    "alias_group": alias_group,
                }
            )
    return aliases


def _search_vector(query: str, top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    if not CHROMA_DIR.exists():
        return []

    try:
        import chromadb

        embedding = _query_embedding(query)
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
            "vector_rank": rank,
        }
        for rank, (metadata, distance) in enumerate(zip(metadatas, distances), start=1)
    ]


def _merge_candidates(
    query: str,
    alias_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}

    for item in alias_results:
        _upsert_candidate(candidates, item, "alias")
    for item in vector_results:
        _upsert_candidate(candidates, item, "vector")
    for item in bm25_results:
        _upsert_candidate(candidates, item, "bm25")

    return sorted(
        candidates.values(),
        key=lambda candidate: _candidate_rank_score(query, candidate),
        reverse=True,
    )


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
    rank_field = f"{source}_rank"
    if rank_field in item:
        candidate[rank_field] = item[rank_field]
    if source == "alias" and item.get("alias_group"):
        candidate["alias_group"] = item["alias_group"]


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


def _query_alias_groups(query: str) -> list[str]:
    normalized = _normalized_query(query)
    groups: list[str] = []
    if _contains_any(normalized, GDP_MARKERS):
        groups.append("gdp")
    if _contains_any(normalized, INFLATION_MARKERS):
        groups.append("inflation")
    return groups


def _candidate_rank_score(query: str, candidate: dict[str, Any]) -> float:
    normalized_query = _normalized_query(query)
    normalized_title = _normalized_query(str(candidate.get("title") or ""))
    score = 0.0

    alias_rank = candidate.get("alias_rank")
    if isinstance(alias_rank, int):
        score += 100.0 - alias_rank

    bm25_rank = candidate.get("bm25_rank")
    if isinstance(bm25_rank, int):
        score += 60.0 / (bm25_rank + 1)

    vector_rank = candidate.get("vector_rank")
    if isinstance(vector_rank, int):
        score += 8.0 / (vector_rank + 1)

    if _is_international_query(normalized_query) and candidate.get("source") == "world_bank":
        score += 8.0

    if _contains_any(normalized_query, GDP_MARKERS):
        if _contains_any(normalized_title, {"gdp", "gross domestic product"}):
            score += 12.0
        if "% of gdp" in normalized_title or "percent of gdp" in normalized_title:
            score -= 8.0
        if "per capita" in normalized_title or "на душу" in normalized_title:
            score -= 4.0

    if _contains_any(normalized_query, INFLATION_MARKERS):
        if _contains_any(normalized_title, {"inflation", "consumer price", "consumer prices", "cpi"}):
            score += 12.0
        if "base year" in normalized_title:
            score -= 6.0

    return score


def _is_international_query(normalized_query: str) -> bool:
    matched = sum(1 for marker in INTERNATIONAL_MARKERS if marker in normalized_query)
    return matched >= 1 and not _is_russia_only_query(normalized_query)


def _is_russia_only_query(normalized_query: str) -> bool:
    has_russia = "россия" in normalized_query or "россии" in normalized_query or "russia" in normalized_query
    has_other_country = any(
        marker in normalized_query
        for marker in INTERNATIONAL_MARKERS
        if marker not in {"мир", "world"}
    )
    return has_russia and not has_other_country


def _contains_any(text: str, markers: set[str]) -> bool:
    return any(marker in text for marker in markers)


def _normalized_query(text: str) -> str:
    return " ".join(text.lower().replace("ё", "е").split())


@lru_cache(maxsize=1)
def _catalog_records() -> dict[str, dict[str, Any]]:
    with CATALOG_RECORDS_PATH.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    return {record["record_id"]: record for record in records}


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        EMBEDDING_MODEL,
        cache_folder=str(EMBEDDING_CACHE_DIR),
        local_files_only=True,
    )


def _query_embedding(query: str) -> list[float]:
    expanded_query = _expanded_query_for_embedding(query)
    if EMBEDDING_MODEL == "local-hash":
        from catalog_local_embeddings import embed_text

        return embed_text(f"{QUERY_INSTRUCTION}{expanded_query}")

    return _embedding_model().encode(
        [f"{QUERY_INSTRUCTION}{expanded_query}"],
        normalize_embeddings=True,
    )[0].tolist()


def _expanded_query_for_embedding(query: str) -> str:
    try:
        from catalog_bm25_indexer import tokenize_query

        tokens = tokenize_query(query)
    except Exception:
        tokens = []
    return " ".join([query, *tokens])
