from __future__ import annotations

import csv
import json
import re
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

try:
    import requests
except ImportError:  # pragma: no cover - urllib fallback covers minimal environments.
    requests = None  # type: ignore[assignment]


DEFAULT_NSEDC_BASE_URL = "https://repository.nsedc.ru"
DEFAULT_STORAGE_DIR = Path(__file__).resolve().parents[1] / "data" / "nsedc"
DEFAULT_TABLE_FORMAT_PRIORITY = (
    "CSV",
    "TSV",
    "XLSX",
    "XLS",
    "PARQUET",
    "CSV_ZST",
    "JSON",
    "XML",
    "OTHER",
)
NON_DATA_RESOURCE_MARKERS = (
    "metadata",
    "codelist",
    "codelists",
    "conceptscheme",
    "dataflows",
    "indicator",
    "indicators",
    "schema",
)


class NsedcApiError(RuntimeError):
    """Raised when the NSEDC CKAN API cannot return a successful result."""


@dataclass(frozen=True)
class NsedcResource:
    id: str
    name: str
    format: str
    url: str
    description: str | None = None
    mimetype: str | None = None
    package_id: str | None = None
    position: int | None = None


@dataclass(frozen=True)
class TableProfile:
    format: str
    size_bytes: int
    columns: list[str]
    sample_rows: list[dict[str, str]]
    row_count: int | None
    notes: list[str]


@dataclass(frozen=True)
class DownloadedTable:
    dataset_name: str
    resource: NsedcResource
    local_path: str
    profile_path: str
    profile: TableProfile
    source_url: str
    downloaded_at: str

    def context_payload(self) -> dict[str, Any]:
        """Small payload that can be safely passed to an LLM prompt."""

        return {
            "dataset_name": self.dataset_name,
            "resource_id": self.resource.id,
            "resource_name": self.resource.name,
            "format": self.profile.format,
            "local_path": self.local_path,
            "profile_path": self.profile_path,
            "source_url": self.source_url,
            "size_bytes": self.profile.size_bytes,
            "columns": self.profile.columns,
            "sample_rows": self.profile.sample_rows,
            "row_count": self.profile.row_count,
            "notes": self.profile.notes,
        }


class NsedcClient:
    """Small CKAN client for the National Socio-Economic Data Center."""

    def __init__(
        self,
        base_url: str = DEFAULT_NSEDC_BASE_URL,
        storage_dir: str | Path = DEFAULT_STORAGE_DIR,
        timeout: int = 60,
        user_agent: str = "ECOS-AI-Assistant/0.1",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.storage_dir = Path(storage_dir)
        self.timeout = timeout
        self.user_agent = user_agent

    def action(self, action_name: str, params: dict[str, Any] | None = None) -> Any:
        """Call a CKAN action and return its `result` field."""

        url = f"{self.base_url}/api/3/action/{action_name}"
        query = urlencode(_drop_none(params or {}), doseq=True)
        if query:
            url = f"{url}?{query}"

        if requests is not None:
            try:
                response = requests.get(
                    url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": self.user_agent,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, json.JSONDecodeError) as exc:
                raise NsedcApiError(f"NSEDC API request failed for {url}: {exc}") from exc
            if not payload.get("success"):
                raise NsedcApiError(f"NSEDC API action {action_name!r} failed: {payload!r}")
            return payload.get("result")

        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise NsedcApiError(f"NSEDC API returned HTTP {exc.code} for {url}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise NsedcApiError(f"NSEDC API request failed for {url}: {exc}") from exc

        if not payload.get("success"):
            raise NsedcApiError(f"NSEDC API action {action_name!r} failed: {payload!r}")
        return payload.get("result")

    def search_datasets(
        self,
        query: str,
        rows: int = 10,
        start: int = 0,
        filter_query: str | None = None,
        sort: str | None = None,
    ) -> dict[str, Any]:
        return self.action(
            "package_search",
            {
                "q": query.strip() or "*:*",
                "rows": rows,
                "start": start,
                "fq": filter_query,
                "sort": sort,
            },
        )

    def get_dataset(self, dataset_name: str) -> dict[str, Any]:
        if not dataset_name.strip():
            raise ValueError("dataset_name must not be empty.")
        return self.action("package_show", {"id": dataset_name})

    def list_resources(self, dataset_name: str) -> list[NsedcResource]:
        dataset = self.get_dataset(dataset_name)
        return [resource_from_ckan(item) for item in dataset.get("resources", [])]

    def choose_table_resource(
        self,
        resources: list[NsedcResource] | list[dict[str, Any]],
        preferred_formats: tuple[str, ...] = DEFAULT_TABLE_FORMAT_PRIORITY,
    ) -> NsedcResource | None:
        typed_resources = [
            resource if isinstance(resource, NsedcResource) else resource_from_ckan(resource)
            for resource in resources
        ]
        downloadable = [resource for resource in typed_resources if resource.url]
        if not downloadable:
            return None

        priority = {item: index for index, item in enumerate(preferred_formats)}
        return min(
            downloadable,
            key=lambda resource: _resource_sort_key(resource, priority),
        )

    def download_best_table(
        self,
        dataset_name: str,
        preferred_formats: tuple[str, ...] = DEFAULT_TABLE_FORMAT_PRIORITY,
        overwrite: bool = False,
        sample_rows: int = 5,
        sample_columns: int = 30,
    ) -> DownloadedTable:
        dataset = self.get_dataset(dataset_name)
        resource = self.choose_table_resource(dataset.get("resources", []), preferred_formats)
        if resource is None:
            raise NsedcApiError(f"Dataset {dataset_name!r} has no downloadable resources.")
        return self.download_resource(
            dataset_name,
            resource,
            overwrite=overwrite,
            sample_rows=sample_rows,
            sample_columns=sample_columns,
        )

    def download_resource(
        self,
        dataset_name: str,
        resource: NsedcResource | dict[str, Any],
        overwrite: bool = False,
        sample_rows: int = 5,
        sample_columns: int = 30,
    ) -> DownloadedTable:
        resource = resource if isinstance(resource, NsedcResource) else resource_from_ckan(resource)
        if not resource.url:
            raise ValueError("resource.url must not be empty.")

        local_path = self._resource_path(dataset_name, resource)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not local_path.exists():
            self._download_url(resource.url, local_path)

        profile = profile_table_file(
            local_path,
            resource_format=infer_resource_format(resource),
            sample_rows=sample_rows,
            sample_columns=sample_columns,
        )
        profile_path = self._profile_path(dataset_name, resource)
        profile_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded_at = datetime.now(timezone.utc).isoformat()
        table = DownloadedTable(
            dataset_name=dataset_name,
            resource=resource,
            local_path=str(local_path),
            profile_path=str(profile_path),
            profile=profile,
            source_url=resource.url,
            downloaded_at=downloaded_at,
        )
        profile_path.write_text(
            json.dumps(
                {
                    "dataset_name": table.dataset_name,
                    "resource": asdict(table.resource),
                    "local_path": table.local_path,
                    "source_url": table.source_url,
                    "downloaded_at": table.downloaded_at,
                    "profile": asdict(table.profile),
                    "context_payload": table.context_payload(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return table

    def _resource_path(self, dataset_name: str, resource: NsedcResource) -> Path:
        dataset_dir = self.storage_dir / "raw" / safe_slug(dataset_name)
        filename = safe_filename(resource.name) or safe_filename(Path(urlparse(resource.url).path).name)
        if not filename:
            filename = f"{resource.id or 'resource'}.{infer_resource_format(resource).lower()}"
        return dataset_dir / f"{safe_slug(resource.id or resource.name)}-{filename}"

    def _profile_path(self, dataset_name: str, resource: NsedcResource) -> Path:
        dataset_dir = self.storage_dir / "profiles" / safe_slug(dataset_name)
        return dataset_dir / f"{safe_slug(resource.id or resource.name)}.json"

    def _download_url(self, url: str, local_path: Path) -> None:
        temporary_path = local_path.with_suffix(f"{local_path.suffix}.tmp-{int(time.time())}")
        try:
            if requests is not None:
                with requests.get(
                    url,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout,
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    with temporary_path.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
            else:
                request = Request(url, headers={"User-Agent": self.user_agent})
                with urlopen(request, timeout=self.timeout) as response:
                    with temporary_path.open("wb") as output:
                        shutil.copyfileobj(response, output)
            temporary_path.replace(local_path)
        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise


def resource_from_ckan(payload: dict[str, Any]) -> NsedcResource:
    return NsedcResource(
        id=str(payload.get("id") or ""),
        name=str(payload.get("name") or payload.get("url") or ""),
        format=str(payload.get("format") or "").upper(),
        url=str(payload.get("url") or ""),
        description=payload.get("description"),
        mimetype=payload.get("mimetype"),
        package_id=payload.get("package_id"),
        position=payload.get("position"),
    )


def infer_resource_format(resource: NsedcResource) -> str:
    name = f"{resource.name} {urlparse(resource.url).path}".lower()
    explicit = (resource.format or "").upper()
    if name.endswith(".csv.zst") or ".csv.zst" in name:
        return "CSV_ZST"
    if name.endswith(".tsv") or explicit == "TSV":
        return "TSV"
    if name.endswith(".csv") or explicit == "CSV":
        return "CSV"
    if name.endswith(".xlsx") or explicit == "XLSX":
        return "XLSX"
    if name.endswith(".xls") or explicit == "XLS":
        return "XLS"
    if name.endswith(".parquet") or explicit == "PARQUET":
        return "PARQUET"
    if name.endswith(".json") or explicit == "JSON":
        return "JSON"
    if name.endswith(".xml") or explicit == "XML":
        return "XML"
    return explicit or "OTHER"


def profile_table_file(
    path: str | Path,
    resource_format: str,
    sample_rows: int = 5,
    sample_columns: int = 30,
) -> TableProfile:
    path = Path(path)
    fmt = resource_format.upper()
    size_bytes = path.stat().st_size
    try:
        if fmt in {"CSV", "TSV"}:
            return _profile_delimited_file(path, fmt, size_bytes, sample_rows, sample_columns)
        if fmt == "XLSX":
            return _profile_xlsx_file(path, size_bytes, sample_rows, sample_columns)
        if fmt == "JSON":
            return _profile_json_file(path, size_bytes, sample_rows, sample_columns)
    except Exception as exc:
        return TableProfile(
            format=fmt,
            size_bytes=size_bytes,
            columns=[],
            sample_rows=[],
            row_count=None,
            notes=[f"Could not build table preview: {exc.__class__.__name__}: {exc}"],
        )

    return TableProfile(
        format=fmt,
        size_bytes=size_bytes,
        columns=[],
        sample_rows=[],
        row_count=None,
        notes=[
            f"Preview for {fmt} is not supported by the lightweight client.",
            "Use the local_path for downstream processing instead of passing the full table to the LLM.",
        ],
    )


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-._").lower()
    return slug[:120] or "item"


def safe_filename(value: str) -> str:
    filename = Path(value.strip()).name
    filename = re.sub(r"[^a-zA-Z0-9_.@()+ -]+", "-", filename).strip(" .-")
    return filename[:180]


def compact_text(value: str | None, limit: int = 700) -> str | None:
    if not value:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}..."


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _resource_sort_key(resource: NsedcResource, priority: dict[str, int]) -> tuple[int, int, int]:
    fmt = infer_resource_format(resource)
    marker_penalty = _non_data_marker_penalty(resource)
    position = resource.position if resource.position is not None else 9999
    return (marker_penalty, priority.get(fmt, len(priority) + 1), position)


def _non_data_marker_penalty(resource: NsedcResource) -> int:
    text = f"{resource.name} {resource.description or ''} {urlparse(resource.url).path}".lower()
    return 1 if any(marker in text for marker in NON_DATA_RESOURCE_MARKERS) else 0


def _profile_delimited_file(
    path: Path,
    fmt: str,
    size_bytes: int,
    sample_rows: int,
    sample_columns: int,
) -> TableProfile:
    raw = path.read_bytes()[:1_000_000]
    text = _decode_text(raw)
    delimiter = "\t" if fmt == "TSV" else _detect_delimiter(text)
    reader = csv.reader(StringIO(text), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return TableProfile(fmt, size_bytes, [], [], None, ["File is empty."])

    columns = [_clean_cell(item) for item in rows[0]][:sample_columns]
    samples = [
        _row_to_dict(columns, row[:sample_columns])
        for row in rows[1 : sample_rows + 1]
    ]
    notes = [
        "Profile was built from the first 1 MB of the file.",
        "row_count is omitted to avoid scanning a potentially large table.",
    ]
    return TableProfile(fmt, size_bytes, columns, samples, None, notes)


def _profile_xlsx_file(
    path: Path,
    size_bytes: int,
    sample_rows: int,
    sample_columns: int,
) -> TableProfile:
    with zipfile.ZipFile(path) as workbook:
        shared_strings = _read_xlsx_shared_strings(workbook)
        sheet_name = _first_xlsx_sheet_name(workbook)
        if sheet_name is None:
            return TableProfile("XLSX", size_bytes, [], [], None, ["Workbook has no worksheet XML."])
        sheet_xml = workbook.read(sheet_name)

    rows = _read_xlsx_rows(sheet_xml, shared_strings, sample_rows + 1, sample_columns)
    if not rows:
        return TableProfile("XLSX", size_bytes, [], [], None, ["First worksheet is empty."])

    columns = [_clean_cell(item) or f"column_{index + 1}" for index, item in enumerate(rows[0])]
    samples = [_row_to_dict(columns, row) for row in rows[1 : sample_rows + 1]]
    return TableProfile(
        "XLSX",
        size_bytes,
        columns,
        samples,
        None,
        [
            "Profile was built from the first worksheet only.",
            "row_count is omitted to avoid loading the full workbook.",
        ],
    )


def _profile_json_file(
    path: Path,
    size_bytes: int,
    sample_rows: int,
    sample_columns: int,
) -> TableProfile:
    text = path.read_text(encoding="utf-8-sig")[:1_000_000]
    payload = json.loads(text)
    rows = _extract_json_rows(payload)
    if not rows:
        return TableProfile("JSON", size_bytes, [], [], None, ["Could not find a list of JSON objects."])

    columns = list(dict.fromkeys(key for row in rows for key in row.keys()))[:sample_columns]
    samples = [
        {key: _clean_cell(row.get(key)) for key in columns}
        for row in rows[:sample_rows]
    ]
    return TableProfile(
        "JSON",
        size_bytes,
        columns,
        samples,
        len(rows),
        ["JSON row_count reflects rows found in the parsed preview payload."],
    )


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _detect_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
    except csv.Error:
        candidates = [",", ";", "\t", "|"]
        return max(candidates, key=text.count)


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _row_to_dict(columns: list[str], row: list[Any]) -> dict[str, str]:
    return {
        column or f"column_{index + 1}": _clean_cell(row[index] if index < len(row) else "")
        for index, column in enumerate(columns)
    }


def _read_xlsx_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []
    root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall(".//{*}si"):
        strings.append("".join(node.text or "" for node in item.findall(".//{*}t")))
    return strings


def _first_xlsx_sheet_name(workbook: zipfile.ZipFile) -> str | None:
    sheets = sorted(
        name
        for name in workbook.namelist()
        if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
    )
    return sheets[0] if sheets else None


def _read_xlsx_rows(
    sheet_xml: bytes,
    shared_strings: list[str],
    max_rows: int,
    max_columns: int,
) -> list[list[str]]:
    root = ElementTree.fromstring(sheet_xml)
    rows: list[list[str]] = []
    for row in root.findall(".//{*}sheetData/{*}row"):
        values: list[str] = []
        for cell in row.findall("{*}c"):
            column_index = _xlsx_column_index(cell.attrib.get("r", ""))
            if column_index is None or column_index >= max_columns:
                continue
            while len(values) <= column_index:
                values.append("")
            values[column_index] = _xlsx_cell_value(cell, shared_strings)
        rows.append(values[:max_columns])
        if len(rows) >= max_rows:
            break
    return rows


def _xlsx_column_index(cell_ref: str) -> int | None:
    match = re.match(r"([A-Z]+)", cell_ref.upper())
    if not match:
        return None
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _xlsx_cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find("{*}v")
    if cell_type == "inlineStr":
        return _clean_cell("".join(node.text or "" for node in cell.findall(".//{*}t")))
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        try:
            return shared_strings[int(value.text)]
        except (IndexError, ValueError):
            return value.text
    return _clean_cell(value.text)


def _extract_json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value[:20]):
                return value
        return [payload]
    return []
