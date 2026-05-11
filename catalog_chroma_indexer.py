import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from catalog_builder import (
    CATALOG_DOCUMENTS_PATH,
    CHROMA_DIR,
    EMBEDDING_CACHE_DIR,
    ROOT,
    clean_text,
    relative,
)


COLLECTION_NAME = "catalog_documents"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
TEXT_INSTRUCTION = "passage: "
QUERY_INSTRUCTION = "query: "
BATCH_SIZE = 512

METADATA_FIELDS = (
    "record_id",
    "dataset_id",
    "source",
    "title",
    "tags",
    "dimensions",
    "unit",
    "frequency",
    "source_name",
    "owner",
    "data_path",
    "source_url",
    "language",
    "is_invalid",
)


@dataclass(frozen=True)
class CatalogIndexDocument:
    doc_id: str
    text: str
    metadata: dict[str, str | bool]


@dataclass(frozen=True)
class IndexStats:
    records_processed: int
    documents_indexed: int
    skipped_empty_search_text: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma index from catalog search documents.")
    parser.add_argument("--input", default=str(CATALOG_DOCUMENTS_PATH))
    parser.add_argument("--persist-dir", default=str(CHROMA_DIR))
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def normalize_text(value: Any) -> str:
    text = clean_text(value)
    return " ".join(text.split()) if text else ""


def normalize_list(values: Iterable[Any]) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_text(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return "; ".join(result)


def metadata_value(value: Any) -> str | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, list):
        return normalize_list(value)
    return normalize_text(value)


def build_metadata(record: dict[str, Any]) -> dict[str, str | bool]:
    return {field: metadata_value(record.get(field)) for field in METADATA_FIELDS}


def build_documents(records: Iterable[dict[str, Any]]) -> tuple[list[CatalogIndexDocument], IndexStats]:
    documents: list[CatalogIndexDocument] = []
    records_processed = 0
    skipped_empty_search_text = 0

    for record in records:
        records_processed += 1
        record_id = normalize_text(record.get("record_id"))
        search_text = normalize_text(record.get("search_text"))
        if not record_id:
            raise ValueError("catalog document without record_id")
        if not search_text:
            skipped_empty_search_text += 1
            continue
        documents.append(
            CatalogIndexDocument(
                doc_id=record_id,
                text=search_text,
                metadata=build_metadata(record),
            )
        )

    return documents, IndexStats(
        records_processed=records_processed,
        documents_indexed=len(documents),
        skipped_empty_search_text=skipped_empty_search_text,
    )


def reset_persist_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def index_documents(
    documents: list[CatalogIndexDocument],
    persist_dir: Path,
    collection_name: str,
    embedding_model_name: str,
    batch_size: int,
) -> int:
    import chromadb
    from llama_index.core import Settings, StorageContext, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.schema import TextNode
    from llama_index.vector_stores.chroma import ChromaVectorStore

    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    Settings.llm = None
    embed_model = HuggingFaceEmbedding(
        model_name=embedding_model_name,
        text_instruction=TEXT_INSTRUCTION,
        query_instruction=QUERY_INSTRUCTION,
        cache_folder=str(EMBEDDING_CACHE_DIR),
    )
    reset_persist_dir(persist_dir)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    metadata_keys = list(METADATA_FIELDS)
    nodes = [
        TextNode(
            text=document.text,
            id_=document.doc_id,
            metadata=document.metadata,
            excluded_embed_metadata_keys=metadata_keys,
            excluded_llm_metadata_keys=metadata_keys,
        )
        for document in documents
    ]
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[],
        insert_batch_size=batch_size,
        show_progress=True,
    )
    return collection.count()


def print_stats(stats: IndexStats, persist_dir: Path, collection_name: str, collection_count: int) -> None:
    print(f"Records обработано: {stats.records_processed}")
    print(f"Documents проиндексировано: {stats.documents_indexed}")
    print(f"Пустых search_text пропущено: {stats.skipped_empty_search_text}")
    print(f"Documents в Chroma collection: {collection_count}")
    print(f"Chroma index: {relative(persist_dir)}")
    print(f"Chroma collection: {collection_name}")


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    persist_dir = resolve_path(args.persist_dir)
    documents, stats = build_documents(iter_jsonl(input_path))
    collection_count = index_documents(
        documents=documents,
        persist_dir=persist_dir,
        collection_name=args.collection,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
    )
    print_stats(stats, persist_dir, args.collection, collection_count)


if __name__ == "__main__":
    main()
