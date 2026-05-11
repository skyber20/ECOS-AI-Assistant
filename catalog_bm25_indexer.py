import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from catalog_builder import CATALOG_BM25_INDEX_PATH, CATALOG_DOCUMENTS_PATH, ROOT, clean_text, relative


DEFAULT_DOCUMENTS_PATH = CATALOG_DOCUMENTS_PATH
DEFAULT_INDEX_PATH = CATALOG_BM25_INDEX_PATH

LEXICAL_FIELDS = (
    "record_id",
    "dataset_id",
    "title",
    "search_text",
    "tags",
    "dimensions",
    "source_name",
    "owner",
    "unit",
    "frequency",
)

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
QUERY_SYNONYMS = {
    "врп": ["валовой", "региональный", "продукт"],
    "ввп": ["валовой", "внутренний", "продукт", "gdp", "gross", "domestic", "product"],
    "инфляция": ["inflation", "consumer", "prices", "annual", "cpi", "индекс", "потребительских", "цен"],
    "инфляции": ["inflation", "consumer", "prices", "annual", "cpi", "индекс", "потребительских", "цен"],
    "инфляцию": ["inflation", "consumer", "prices", "annual", "cpi", "индекс", "потребительских", "цен"],
    "ниокр": ["исследования", "разработки"],
    "ипц": ["индекс", "потребительских", "цен", "inflation", "consumer", "prices", "cpi"],
    "торговля": ["внешняя", "экспорт", "импорт", "товаров", "услуг"],
    "товарооборот": ["торговля", "экспорт", "импорт"],
    "всемирный": ["world", "bank", "world_bank", "wb_wdi"],
    "мировой": ["world", "bank", "world_bank", "wb_wdi"],
    "ворлд": ["world", "bank", "world_bank", "wb_wdi"],
    "world": ["bank", "world_bank", "wb_wdi"],
    "wb": ["world", "bank", "world_bank", "wb_wdi"],
    "wdi": ["world", "bank", "world_bank", "wb_wdi"],
    "росстат": ["fedstat"],
    "емисс": ["fedstat"],
}
QUERY_STOP_TOKENS = {
    "россия",
    "россии",
    "российская",
    "российской",
    "федерация",
    "федерации",
    "казахстан",
    "казахстана",
    "сша",
    "мир",
    "мира",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SQLite FTS5/BM25 index from catalog metadata documents."
    )
    parser.add_argument("--input", default=str(DEFAULT_DOCUMENTS_PATH))
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
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


def normalize_field(value: Any) -> str:
    if isinstance(value, list):
        return normalize_list(value)
    return normalize_text(value)


def lexical_fields(document: dict[str, Any]) -> dict[str, str]:
    return {field: normalize_field(document.get(field)) for field in LEXICAL_FIELDS}


def create_schema(connection: sqlite3.Connection) -> None:
    columns = ", ".join(LEXICAL_FIELDS)
    connection.executescript(
        f"""
        DROP TABLE IF EXISTS bm25_documents;
        DROP TABLE IF EXISTS bm25_index;

        CREATE TABLE bm25_documents (
            doc_id INTEGER PRIMARY KEY,
            record_id TEXT NOT NULL UNIQUE,
            document_json TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE bm25_index USING fts5(
            {columns},
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )


def insert_document(
    connection: sqlite3.Connection,
    document: dict[str, Any],
    fields: dict[str, str],
) -> None:
    cursor = connection.execute(
        "INSERT INTO bm25_documents (record_id, document_json) VALUES (?, ?)",
        (
            fields["record_id"],
            json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        ),
    )
    placeholders = ", ".join("?" for _ in LEXICAL_FIELDS)
    connection.execute(
        f"INSERT INTO bm25_index (rowid, {', '.join(LEXICAL_FIELDS)}) VALUES (?, {placeholders})",
        [cursor.lastrowid, *[fields[field] for field in LEXICAL_FIELDS]],
    )


def build_bm25_index(
    input_path: str | Path = DEFAULT_DOCUMENTS_PATH,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> dict[str, int]:
    input_path = resolve_path(input_path)
    index_path = resolve_path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index_path.unlink()

    stats = {
        "records_processed": 0,
        "documents_indexed": 0,
    }

    with sqlite3.connect(index_path) as connection:
        create_schema(connection)

        for document in iter_jsonl(input_path):
            stats["records_processed"] += 1
            fields = lexical_fields(document)
            record_id = fields["record_id"]

            if not record_id:
                raise ValueError("catalog document without record_id")
            try:
                insert_document(connection, document, fields)
            except sqlite3.IntegrityError as error:
                raise ValueError(f"duplicate record_id in catalog documents: {record_id}") from error
            stats["documents_indexed"] += 1

    return stats


def tokenize_query(query: str) -> list[str]:
    tokens = TOKEN_RE.findall(normalize_text(query).lower())
    expanded: list[str] = []
    for token in tokens:
        if token.isdigit() or token in QUERY_STOP_TOKENS:
            continue
        expanded.append(token)
        expanded.extend(QUERY_SYNONYMS.get(token, []))
    return list(dict.fromkeys(expanded))


def bm25_query(query: str) -> str:
    tokens = tokenize_query(query)
    return " OR ".join(f'"{token}"' for token in tokens)


def search_bm25(
    query: str,
    top_k: int = 10,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> list[dict[str, Any]]:
    match_query = bm25_query(query)
    if not match_query:
        return []
    resolved_index_path = resolve_path(index_path)
    if not resolved_index_path.exists():
        return []

    with sqlite3.connect(resolved_index_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                d.document_json,
                bm25(bm25_index) AS score
            FROM bm25_index
            JOIN bm25_documents d ON d.doc_id = bm25_index.rowid
            WHERE bm25_index MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (match_query, top_k),
        ).fetchall()

    results: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        document = json.loads(row["document_json"])
        document["bm25_score"] = row["score"]
        document["bm25_rank"] = rank
        results.append(document)
    return results


def display_path(path: str | Path) -> str:
    resolved = resolve_path(path)
    try:
        return relative(resolved)
    except ValueError:
        return resolved.as_posix()


def print_stats(stats: dict[str, int], index_path: str | Path) -> None:
    print(f"Records обработано: {stats['records_processed']}")
    print(f"Documents в BM25 index: {stats['documents_indexed']}")
    print(f"BM25 index: {display_path(index_path)}")


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    index_path = resolve_path(args.index)

    stats = build_bm25_index(input_path=input_path, index_path=index_path)
    print_stats(stats, index_path)


if __name__ == "__main__":
    main()
