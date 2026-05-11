import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from catalog_builder import ROOT, clean_text, relative


SEARCH_TEXT_FIELDS = (
    "title",
    "description",
    "long_description",
    "methodology",
    "limitations",
    "tags",
    "dimensions",
    "source_name",
    "unit",
    "frequency",
)

DOCUMENT_SCHEMA = (
    "record_id",
    "dataset_id",
    "source",
    "title",
    "search_text",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build metadata-only catalog search documents.")
    parser.add_argument("--input", default="data/catalog_records.jsonl")
    parser.add_argument("--output", default="data/catalog_documents.jsonl")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def normalize_text(value: Any) -> str | None:
    text = clean_text(value)
    return " ".join(text.split()) if text else None


def normalize_list(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_text(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def normalize_list_field(value: Any) -> list[str]:
    if isinstance(value, list):
        return normalize_list(value)
    text = normalize_text(value)
    return [text] if text else []


def normalize_search_value(value: Any) -> str | None:
    if isinstance(value, list):
        values = normalize_list(value)
        return "; ".join(values) if values else None
    return normalize_text(value)


def build_search_text(record: dict[str, Any]) -> str:
    parts = [
        text
        for field in SEARCH_TEXT_FIELDS
        if (text := normalize_search_value(record.get(field)))
    ]
    return "\n".join(parts)


def build_document(record: dict[str, Any]) -> dict[str, Any]:
    document = {
        "record_id": normalize_text(record.get("record_id")),
        "dataset_id": normalize_text(record.get("dataset_id")),
        "source": normalize_text(record.get("source")),
        "title": normalize_text(record.get("title")),
        "search_text": build_search_text(record),
        "tags": normalize_list_field(record.get("tags")),
        "dimensions": normalize_list_field(record.get("dimensions")),
        "unit": normalize_text(record.get("unit")),
        "frequency": normalize_text(record.get("frequency")),
        "source_name": normalize_text(record.get("source_name")),
        "owner": normalize_text(record.get("owner")),
        "data_path": normalize_text(record.get("data_path")),
        "source_url": normalize_text(record.get("source_url")),
        "language": normalize_text(record.get("language")),
        "is_invalid": bool(record.get("is_invalid")),
    }
    return {field: document[field] for field in DOCUMENT_SCHEMA}


def build_documents(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_document(record) for record in records]


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_stats(documents: list[dict[str, Any]], records_processed: int) -> dict[str, int]:
    return {
        "records processed": records_processed,
        "documents created": len(documents),
        "documents with empty search_text": sum(not document["search_text"] for document in documents),
    }


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    documents = build_documents(iter_jsonl(input_path))
    write_jsonl(documents, output_path)

    print(f"Сохранено: {relative(output_path)}")
    for name, value in build_stats(documents, len(documents)).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
