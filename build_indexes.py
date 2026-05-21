#!/usr/bin/env python3
"""сборка индексов"""
import zipfile
from pathlib import Path

from catalog_builder import build_catalog, write_jsonl
from catalog_documents_builder import build_documents, write_jsonl as write_docs
from catalog_bm25_indexer import build_bm25_index
from catalog_chroma_indexer import index_documents, build_documents as build_chroma_docs, reset_persist_dir

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

def main():
    print("1. Сборка catalog_records.jsonl...")
    records = build_catalog()
    write_jsonl(records, DATA_DIR / "catalog_records.jsonl")
    print(f"   Создано {len(records)} записей")

    print("2. Сборка catalog_documents.jsonl...")
    documents = build_documents(records)
    write_docs(documents, DATA_DIR / "catalog_documents.jsonl")
    print(f"   Создано {len(documents)} документов")

    print("3. Построение BM25 индекса...")
    build_bm25_index(DATA_DIR / "catalog_documents.jsonl", DATA_DIR / "catalog_bm25.sqlite")

    print("4. Построение Chroma векторного индекса...")
    from catalog_chroma_indexer import CHROMA_DIR, EMBEDDING_CACHE_DIR, COLLECTION_NAME, EMBEDDING_MODEL, BATCH_SIZE
    
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    from catalog_chroma_indexer import iter_jsonl, CatalogIndexDocument
    docs_list, stats = build_chroma_docs(iter_jsonl(DATA_DIR / "catalog_documents.jsonl"))
    collection_count = index_documents(
        documents=docs_list,
        persist_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_model_name=EMBEDDING_MODEL,
        batch_size=BATCH_SIZE,
    )
    print(f"   Проиндексировано {collection_count} документов")

    print("5. Архивирование...")
    with zipfile.ZipFile(DATA_DIR / "chroma.zip", "w") as zf:
        for f in CHROMA_DIR.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(CHROMA_DIR.parent))
    
    with zipfile.ZipFile(DATA_DIR / "embedding_cache.zip", "w") as zf:
        for f in EMBEDDING_CACHE_DIR.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(EMBEDDING_CACHE_DIR.parent))
    
    print("✅ Готово! Файлы созданы:")
    print(f"   - {DATA_DIR}/catalog_documents.jsonl")
    print(f"   - {DATA_DIR}/chroma.zip")
    print(f"   - {DATA_DIR}/embedding_cache.zip")

if __name__ == "__main__":
    main()