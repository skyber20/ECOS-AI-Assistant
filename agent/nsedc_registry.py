from __future__ import annotations

import re
from typing import Any

try:
    from agent.dataset_search_planner import (
        DatasetCandidate,
        DatasetSearchRequest,
    )
    from agent.nsedc_client import (
        NsedcClient,
        NsedcResource,
        compact_text,
        infer_resource_format,
        resource_from_ckan,
    )
except ImportError:  # pragma: no cover - keeps direct `python agent/...` runs working.
    from dataset_search_planner import (  # type: ignore
        DatasetCandidate,
        DatasetSearchRequest,
    )
    from nsedc_client import (  # type: ignore
        NsedcClient,
        NsedcResource,
        compact_text,
        infer_resource_format,
        resource_from_ckan,
    )


class NsedcDatasetRegistry:
    """DatasetRegistry adapter backed by the NSEDC CKAN repository."""

    backend_name = "nsedc_ckan"

    def __init__(
        self,
        client: NsedcClient | None = None,
        max_candidates: int = 10,
    ) -> None:
        self.client = client or NsedcClient()
        self.max_candidates = max_candidates

    def search(self, request: DatasetSearchRequest) -> list[DatasetCandidate]:
        query = build_nsedc_search_query(request)
        result = self.client.search_datasets(query=query, rows=self.max_candidates)
        candidates = [
            self._dataset_to_candidate(dataset, request)
            for dataset in result.get("results", [])
        ]
        return candidates

    def download_best_table(self, dataset_id: str):
        return self.client.download_best_table(dataset_id)

    def _dataset_to_candidate(
        self,
        dataset: dict[str, Any],
        request: DatasetSearchRequest,
    ) -> DatasetCandidate:
        resources = [resource_from_ckan(item) for item in dataset.get("resources", [])]
        best_resource = self.client.choose_table_resource(resources)
        notes = _candidate_notes(dataset, resources, best_resource, self.client.storage_dir)
        columns = _extract_column_names(dataset)
        tags = [item.get("name", "") for item in dataset.get("tags", []) if item.get("name")]

        return DatasetCandidate(
            dataset_id=dataset.get("name") or dataset.get("id") or "",
            title=dataset.get("title") or dataset.get("name") or "",
            description=compact_text(dataset.get("notes")),
            source_name=_source_name(dataset),
            source_url=f"{self.client.base_url}/dataset/{dataset.get('name')}",
            row_grain=None,
            time_coverage=_infer_time_coverage(dataset, tags),
            geography_coverage=_matched_geographies(dataset, request.geography_coverage),
            frequency=_infer_frequency(dataset),
            columns=columns,
            indicators=_unique_texts([dataset.get("title"), *tags]),
            dimensions=_dimension_columns(columns),
            join_keys=_join_key_columns(columns),
            coverage_notes=notes,
        )


def build_nsedc_search_query(request: DatasetSearchRequest) -> str:
    terms = [
        request.target_dataset_name,
        *[column.name for column in request.required_indicators],
        *[column.name for column in request.required_dimensions],
        *request.geography_coverage,
        *(request.source_requirements[:3]),
    ]
    query = " ".join(term for term in _unique_texts(terms) if term)
    return query or request.original_query or "*:*"


def _candidate_notes(
    dataset: dict[str, Any],
    resources: list[NsedcResource],
    best_resource: NsedcResource | None,
    storage_dir: Any,
) -> list[str]:
    notes = [
        f"NSEDC package modified: {dataset.get('metadata_modified') or 'unknown'}.",
        f"NSEDC resources found: {len(resources)}.",
    ]
    if best_resource:
        notes.append(
            "Best table resource: "
            f"{best_resource.name} [{infer_resource_format(best_resource)}]."
        )
    notes.append(
        "Full data is not included in prompt context; download it to "
        f"{storage_dir}/raw and pass the generated profile JSON to the agent."
    )
    return notes


def _source_name(dataset: dict[str, Any]) -> str | None:
    organization = dataset.get("organization") or {}
    if isinstance(organization, dict):
        return organization.get("title") or organization.get("name")
    groups = dataset.get("groups") or []
    if groups and isinstance(groups[0], dict):
        return groups[0].get("title") or groups[0].get("name")
    return None


def _extract_column_names(dataset: dict[str, Any]) -> list[str]:
    text_parts = [dataset.get("notes") or "", dataset.get("title") or ""]
    for resource in dataset.get("resources", []):
        text_parts.extend(
            [
                str(resource.get("name") or ""),
                str(resource.get("description") or ""),
            ]
        )

    text = "\n".join(text_parts)
    code_terms = re.findall(r"`([^`]{1,80})`", text)
    bold_code_terms = re.findall(r"\*\*([^*]{1,80})\*\*\s*\(([^)]{1,80})\)", text)
    columns = [term for term in code_terms]
    columns.extend(code for _, code in bold_code_terms)
    columns.extend(_common_schema_terms(text))
    return _unique_texts(_clean_column_name(term) for term in columns if term)


def _common_schema_terms(text: str) -> list[str]:
    known_terms = (
        "DATAFLOW",
        "REF_AREA",
        "FREQ",
        "TIME_PERIOD",
        "OBS_VALUE",
        "UNIT_MEASURE",
        "OBS_STATUS",
        "year",
        "country",
        "region",
        "value",
    )
    lowered = text.lower()
    return [term for term in known_terms if term.lower() in lowered]


def _clean_column_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value.strip("`'\" ")


def _dimension_columns(columns: list[str]) -> list[str]:
    markers = ("area", "region", "country", "time", "period", "freq", "year", "date")
    return [column for column in columns if any(marker in column.lower() for marker in markers)]


def _join_key_columns(columns: list[str]) -> list[str]:
    markers = ("area", "region", "country", "time", "period", "year", "date")
    return [column for column in columns if any(marker in column.lower() for marker in markers)]


def _matched_geographies(
    dataset: dict[str, Any],
    required_geographies: list[str],
) -> list[str]:
    if not required_geographies:
        return []
    text = _dataset_text(dataset).lower()
    return [
        geography
        for geography in required_geographies
        if geography.lower() in text
    ]


def _infer_frequency(dataset: dict[str, Any]) -> str | None:
    text = _dataset_text(dataset).lower()
    if any(term in text for term in ("годовая", "annual", "yearly", "freq a")):
        return "годовая"
    if any(term in text for term in ("квартальная", "quarterly", "freq q")):
        return "квартальная"
    if any(term in text for term in ("месячная", "monthly", "freq m")):
        return "месячная"
    return None


def _infer_time_coverage(dataset: dict[str, Any], tags: list[str]) -> str | None:
    text = " ".join([_dataset_text(dataset), *tags])
    ranges = re.findall(r"(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}", text)
    if ranges:
        return ranges[0].replace(" ", "")
    years = re.findall(r"(?:19|20)\d{2}", text)
    if len(years) >= 2:
        return f"{min(years)}-{max(years)}"
    return years[0] if years else None


def _dataset_text(dataset: dict[str, Any]) -> str:
    parts = [
        dataset.get("name") or "",
        dataset.get("title") or "",
        dataset.get("notes") or "",
    ]
    parts.extend(tag.get("name", "") for tag in dataset.get("tags", []) if isinstance(tag, dict))
    parts.extend(group.get("title", "") for group in dataset.get("groups", []) if isinstance(group, dict))
    return " ".join(parts)


def _unique_texts(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value:
            continue
        text = str(value).strip()
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
