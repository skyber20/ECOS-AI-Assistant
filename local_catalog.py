import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from catalog_bm25_indexer import build_bm25_index
from catalog_builder import (
    CATALOG_BM25_INDEX_PATH,
    CATALOG_DOCUMENTS_PATH,
    CATALOG_RECORDS_PATH,
    FEDSTAT_ARCHIVE_PATH,
    FEDSTAT_DIR,
    ROOT,
    WB_ARCHIVE_PATH,
    WB_DIR,
    build_catalog,
    build_stats,
    relative,
    write_jsonl,
)
from catalog_documents_builder import build_documents, iter_jsonl as iter_document_input, write_jsonl as write_documents_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract local dumps and build metadata catalog/index artifacts."
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild catalog artifacts even if they exist.")
    parser.add_argument("--skip-extract", action="store_true", help="Do not extract zip archives before indexing.")
    parser.add_argument("--build-chroma", action="store_true", help="Also build the optional Chroma vector index.")
    return parser.parse_args()


def prepare_local_catalog(
    rebuild: bool = False,
    extract: bool = True,
    build_chroma: bool = False,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "extracted": {},
        "catalog_records": None,
        "catalog_documents": None,
        "bm25_index": None,
        "chroma_index": None,
    }

    if extract:
        stats["extracted"] = extract_local_dumps()

    if rebuild or not CATALOG_RECORDS_PATH.exists():
        records = build_catalog()
        write_jsonl(records, CATALOG_RECORDS_PATH)
        stats["catalog_records"] = {
            "path": relative(CATALOG_RECORDS_PATH),
            **build_stats(records),
        }
    else:
        stats["catalog_records"] = {
            "path": relative(CATALOG_RECORDS_PATH),
            "status": "exists",
        }

    if rebuild or not CATALOG_DOCUMENTS_PATH.exists():
        documents = build_documents(iter_document_input(CATALOG_RECORDS_PATH))
        write_documents_jsonl(documents, CATALOG_DOCUMENTS_PATH)
        stats["catalog_documents"] = {
            "path": relative(CATALOG_DOCUMENTS_PATH),
            "documents": len(documents),
            "empty_search_text": sum(not document["search_text"] for document in documents),
        }
    else:
        stats["catalog_documents"] = {
            "path": relative(CATALOG_DOCUMENTS_PATH),
            "status": "exists",
        }

    if rebuild or not CATALOG_BM25_INDEX_PATH.exists():
        bm25_stats = build_bm25_index(
            input_path=CATALOG_DOCUMENTS_PATH,
            index_path=CATALOG_BM25_INDEX_PATH,
        )
        stats["bm25_index"] = {
            "path": relative(CATALOG_BM25_INDEX_PATH),
            **bm25_stats,
        }
    else:
        stats["bm25_index"] = {
            "path": relative(CATALOG_BM25_INDEX_PATH),
            "status": "exists",
        }

    if build_chroma:
        from catalog_chroma_indexer import build_documents as build_chroma_documents
        from catalog_chroma_indexer import index_documents, iter_jsonl
        from catalog_chroma_indexer import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, BATCH_SIZE

        documents, chroma_stats = build_chroma_documents(iter_jsonl(CATALOG_DOCUMENTS_PATH))
        collection_count = index_documents(
            documents=documents,
            persist_dir=CHROMA_DIR,
            collection_name=COLLECTION_NAME,
            embedding_model_name=EMBEDDING_MODEL,
            batch_size=BATCH_SIZE,
        )
        stats["chroma_index"] = {
            "path": relative(CHROMA_DIR),
            "records_processed": chroma_stats.records_processed,
            "documents_indexed": chroma_stats.documents_indexed,
            "collection_count": collection_count,
        }

    return stats


def extract_local_dumps() -> dict[str, str]:
    return {
        "world_bank": _extract_if_needed(WB_ARCHIVE_PATH, WB_DIR.parent, WB_DIR),
        "fedstat": _extract_if_needed(FEDSTAT_ARCHIVE_PATH, FEDSTAT_DIR.parents[1], FEDSTAT_DIR),
    }


def _extract_if_needed(archive_path: Path, output_dir: Path, marker_dir: Path) -> str:
    if marker_dir.exists():
        return "exists"
    if not archive_path.exists():
        return f"missing archive: {relative(archive_path)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.filename.startswith("__MACOSX/") or "/__MACOSX/" in member.filename:
                continue
            archive.extract(member, output_dir)
    return "extracted"


def main() -> None:
    args = parse_args()
    stats = prepare_local_catalog(
        rebuild=args.rebuild,
        extract=not args.skip_extract,
        build_chroma=args.build_chroma,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
