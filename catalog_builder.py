import argparse
import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DUMPS_DIR = ROOT / "dumps"
DATA_DIR = ROOT / "data"
WB_ARCHIVE_PATH = DUMPS_DIR / "wb" / "data.zip"
FEDSTAT_ARCHIVE_PATH = DUMPS_DIR / "fedstatru" / "fedstatru.zip"
WB_DIR = DUMPS_DIR / "wb" / "wb"
FEDSTAT_DIR = DUMPS_DIR / "fedstatru" / "fedstatru" / "data"
CATALOG_RECORDS_PATH = DATA_DIR / "catalog_records.jsonl"
CATALOG_DOCUMENTS_PATH = DATA_DIR / "catalog_documents.jsonl"
CATALOG_BM25_INDEX_PATH = DATA_DIR / "catalog_bm25.sqlite"
CHROMA_DIR = DATA_DIR / "chroma"
EMBEDDING_CACHE_DIR = DATA_DIR / "embedding_cache"
EMPTY_TEXT = {"", "-", "null", "none", "nan"}

SCHEMA = (
    "record_id",
    "source",
    "dataset_id",
    "title",
    "description",
    "long_description",
    "methodology",
    "limitations",
    "tags",
    "dimensions",
    "unit",
    "frequency",
    "source_name",
    "owner",
    "row_count",
    "source_url",
    "data_path",
    "metadata_path",
    "language",
    "is_invalid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build metadata-only catalog records.")
    parser.add_argument("--output", default=str(CATALOG_RECORDS_PATH))
    return parser.parse_args()


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = html.unescape(str(value)).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text).strip()
    return None if text.lower() in EMPTY_TEXT else text


def clean_line(value: Any) -> str | None:
    text = clean_text(value)
    return " ".join(text.split()) if text else None


def first_text(*values: Any) -> str | None:
    return next((text for text in map(clean_text, values) if text), None)


def dedupe(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_line(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def split_marked(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    return dedupe(re.sub(r"^\s*[-*]\s*", "", line) for line in text.splitlines())


def clean_unit(value: Any) -> str | None:
    parts = split_marked(value)
    return "; ".join(parts) if parts else clean_line(value)


def int_or_none(value: Any) -> int | None:
    text = clean_line(value)
    return int(text) if text and text.isdigit() else None


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def existing_path(path: Path, is_invalid: bool = False) -> str | None:
    return None if is_invalid or not path.exists() else relative(path)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def merge_missing(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if clean_text(merged.get(key)) is None and clean_text(value) is not None:
            merged[key] = value
    return merged


def base_record(**values: Any) -> dict[str, Any]:
    record = {field: None for field in SCHEMA}
    record.update({"tags": [], "dimensions": [], "is_invalid": False})
    record.update(values)
    return record


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = set(record)
    expected = set(SCHEMA)
    if keys != expected:
        missing = ", ".join(sorted(expected - keys))
        extra = ", ".join(sorted(keys - expected))
        raise ValueError(f"Invalid catalog schema. Missing: {missing}. Extra: {extra}.")
    return {field: record[field] for field in SCHEMA}


def load_wb_metadata() -> tuple[dict[tuple[str, str | None], dict[str, Any]], dict[str, dict[str, Any]]]:
    by_pair: dict[tuple[str, str | None], dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(WB_DIR / "metadata.jsonl"):
        dataset_id = clean_line(row.get("id"))
        source_id = clean_line(row.get("source_id"))
        if not dataset_id:
            continue
        normalized = dict(row, id=dataset_id, source_id=source_id)
        pair_key = (dataset_id, source_id)
        by_pair[pair_key] = merge_missing(by_pair.get(pair_key, {}), normalized)
        by_id[dataset_id] = merge_missing(by_id.get(dataset_id, {}), normalized)
    return by_pair, by_id


def build_wb_record_id(dataset_id: str, source_id: str | None, id_counts: Counter[str]) -> str:
    return f"wb:{source_id}:{dataset_id}" if id_counts[dataset_id] > 1 and source_id else f"wb:{dataset_id}"


def build_wb_records() -> list[dict[str, Any]]:
    indicators = load_json(WB_DIR / "indicators.json")
    sources = {clean_line(row.get("id")): row for row in load_json(WB_DIR / "sources.json")}
    metadata_by_pair, metadata_by_id = load_wb_metadata()
    id_counts = Counter(clean_line(row.get("id")) for row in indicators if clean_line(row.get("id")))
    records: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()

    for indicator in indicators:
        dataset_id = clean_line(indicator.get("id"))
        if not dataset_id:
            continue
        source = indicator.get("source") or {}
        source_id = clean_line(source.get("id"))
        metadata = metadata_by_pair.get((dataset_id, source_id)) or metadata_by_id.get(dataset_id, {})
        source_info = sources.get(source_id, {})
        topics = [topic.get("value") for topic in indicator.get("topics") or []]
        tags = dedupe([*topics, metadata.get("Topic"), metadata.get("Dataset")])
        metadata_path = WB_DIR / ("metadata.jsonl" if metadata else "indicators.json")
        data_path = WB_DIR / "parquet" / f"{dataset_id}.parquet"

        record = validate_record(base_record(
            record_id=build_wb_record_id(dataset_id, source_id, id_counts),
            source="world_bank",
            dataset_id=dataset_id,
            title=clean_line(first_text(metadata.get("IndicatorName"), indicator.get("name"))),
            description=first_text(metadata.get("Shortdefinition"), indicator.get("sourceNote"), metadata.get("Longdefinition")),
            long_description=first_text(metadata.get("Longdefinition"), metadata.get("Developmentrelevance")),
            methodology=first_text(metadata.get("Statisticalconceptandmethodology"), metadata.get("Aggregationmethod")),
            limitations=first_text(metadata.get("Limitationsandexceptions")),
            tags=tags,
            dimensions=["country", "year"],
            unit=clean_unit(first_text(metadata.get("Unitofmeasure"), indicator.get("unit"))),
            frequency=clean_line(metadata.get("Periodicity")),
            source_name=clean_line(first_text(metadata.get("Dataset"), source_info.get("name"), source.get("value"))),
            owner=clean_line(first_text(indicator.get("sourceOrganization"), metadata.get("Source"), "World Bank")),
            row_count=None,
            source_url=f"https://data.worldbank.org/indicator/{dataset_id}",
            data_path=existing_path(data_path),
            metadata_path=relative(metadata_path),
            language="en",
            is_invalid=False,
        ))
        if record["record_id"] not in seen_record_ids:
            records.append(record)
            seen_record_ids.add(record["record_id"])

    return records


def load_fedstat_csv() -> dict[str, dict[str, Any]]:
    path = FEDSTAT_DIR / "metdata.csv"
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = {}
        for row in csv.DictReader(file):
            code = clean_line(row.get("code"))
            if code:
                rows[code] = row
        return rows


def load_fedstat_json_metadata() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((FEDSTAT_DIR / "metadata").glob("*.json")):
        payload = load_json(path)
        code = clean_line(payload.get("code")) or path.stem
        props = payload.get("props") or {}
        rows[code] = {"code": code, "name": payload.get("name"), **props}
    return rows


def load_invalid_fedstat_codes() -> set[str]:
    return set(load_json(FEDSTAT_DIR / "invalid_files.json"))


def fedstat_metadata_path(code: str, csv_rows: dict[str, dict[str, Any]], json_rows: dict[str, dict[str, Any]]) -> str:
    if code in json_rows:
        return relative(FEDSTAT_DIR / "metadata" / f"{code}.json")
    if code in csv_rows:
        return relative(FEDSTAT_DIR / "metdata.csv")
    return relative(FEDSTAT_DIR / "invalid_files.json")


def sort_codes(codes: Iterable[str]) -> list[str]:
    return sorted(codes, key=lambda code: (not code.isdigit(), int(code) if code.isdigit() else code))


def build_fedstat_records() -> list[dict[str, Any]]:
    csv_rows = load_fedstat_csv()
    json_rows = load_fedstat_json_metadata()
    invalid_codes = load_invalid_fedstat_codes()
    codes = sort_codes(set(csv_rows) | set(json_rows) | invalid_codes)
    records: list[dict[str, Any]] = []

    for code in codes:
        row = merge_missing(csv_rows.get(code, {}), json_rows.get(code, {}))
        is_invalid = code in invalid_codes
        dimensions = split_marked(row.get("Признаки (перечень на базе классификаторов и справочников)"))
        jsonl_path = FEDSTAT_DIR / "clean_jsonl" / f"{code}.jsonl.gz"
        parquet_path = FEDSTAT_DIR / "parquet" / f"{code}.parquet"
        data_path = jsonl_path if jsonl_path.exists() else parquet_path

        records.append(validate_record(base_record(
            record_id=f"fedstat:{code}",
            source="fedstat",
            dataset_id=code,
            title=clean_line(row.get("name")),
            description=first_text(row.get("Методологические пояснения")),
            long_description=first_text(row.get("Источники и способ формирования показателя")),
            methodology=first_text(row.get("Методологические пояснения")),
            limitations=first_text(row.get("Комментарий")),
            tags=dedupe([row.get("Размещение")]),
            dimensions=dimensions,
            unit=clean_unit(row.get("Единицы измерения")),
            frequency=clean_text(row.get("Периодичность и характеристика временного ряда")),
            source_name=clean_line(row.get("Источники и способ формирования показателя")),
            owner=clean_line(row.get("Ведомство (субъект статистического учета)")),
            row_count=int_or_none(row.get("rows")),
            source_url=clean_line(row.get("url")),
            data_path=existing_path(data_path, is_invalid=is_invalid),
            metadata_path=fedstat_metadata_path(code, csv_rows, json_rows),
            language="ru",
            is_invalid=is_invalid,
        )))

    return records


def build_catalog() -> list[dict[str, Any]]:
    return build_wb_records() + build_fedstat_records()


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_stats(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "WB records": sum(record["source"] == "world_bank" for record in records),
        "Fedstat records": sum(record["source"] == "fedstat" for record in records),
        "invalid records": sum(record["is_invalid"] for record in records),
        "records without description": sum(not record["description"] for record in records),
        "records without tags": sum(not record["tags"] for record in records),
        "records without data_path": sum(not record["data_path"] for record in records),
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    records = build_catalog()
    write_jsonl(records, output_path)

    print(f"Сохранено: {relative(output_path)}")
    for name, value in build_stats(records).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
