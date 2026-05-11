import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from catalog_builder import CATALOG_BM25_INDEX_PATH, CATALOG_RECORDS_PATH, CHROMA_DIR, ROOT
from dataset_structure import TargetDatasetStructure
from parser.intent_parser import LLMSettings, ResearchIntent, none_to_empty_list
from designer.research_designer import ResearchStudyDesign


BM25_INDEX_PATH = CATALOG_BM25_INDEX_PATH
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class SourceDatasetReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str | None = None
    dataset_id: str | None = None
    title: str
    source: str | None = None
    source_name: str | None = None
    data_path: str | None = None
    source_url: str | None = None
    role: str = "predecessor"
    usefulness_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    why_selected: str
    possible_limitations: list[str] = Field(default_factory=list)

    @field_validator("possible_limitations", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class SourceAuditEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    record_id: str | None = None
    dataset_id: str | None = None
    source: str | None = None
    decision: str = Field(description="selected, rejected, considered, fallback, unavailable.")
    reason: str
    matched_by: list[str] = Field(default_factory=list)
    score: float | None = None

    @field_validator("matched_by", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class AssemblyStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    description: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)

    @field_validator("inputs", "outputs", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class DatasetBuildPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_dataset_name: str
    registry_checked: bool = False
    registry_assets: dict[str, bool] = Field(default_factory=dict)
    can_use_existing_dataset: bool = False
    can_build_from_local_sources: bool = False
    selected_datasets: list[SourceDatasetReference] = Field(default_factory=list)
    predecessor_datasets: list[SourceDatasetReference] = Field(default_factory=list)
    source_audit: list[SourceAuditEntry] = Field(default_factory=list)
    assembly_steps: list[AssemblyStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)

    @field_validator(
        "selected_datasets",
        "predecessor_datasets",
        "source_audit",
        "assembly_steps",
        "warnings",
        "blocking_reasons",
        mode="before",
    )
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


def plan_dataset_build(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
    structure: TargetDatasetStructure,
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
    use_registry: bool = True,
) -> DatasetBuildPlan:
    registry_assets = registry_assets_status()
    warnings: list[str] = []
    audit: list[SourceAuditEntry] = []
    selected: list[SourceDatasetReference] = []

    if use_registry:
        if not registry_assets["catalog_records"]:
            warnings.append(
                "Локальный реестр датасетов не найден: data/catalog_records.jsonl отсутствует."
            )
            audit.append(
                SourceAuditEntry(
                    title="Локальный реестр датасетов",
                    decision="unavailable",
                    reason="Файл catalog_records.jsonl отсутствует, поэтому RAG-поиск пропущен.",
                )
            )
        else:
            retrieved, retrieval_warnings = _retrieve_registry_candidates(intent, design)
            warnings.extend(retrieval_warnings)
            audit.extend(_audit_retrieved_candidates(retrieved))
            if retrieved:
                selected, selection_warnings = _select_registry_candidates(
                    intent=intent,
                    design=design,
                    candidates=retrieved,
                    settings=settings,
                    provider=provider,
                    model=model,
                )
                warnings.extend(selection_warnings)

    if not selected:
        fallback_sources = _fallback_source_candidates(intent, design)
        selected = fallback_sources
        audit.extend(
            SourceAuditEntry(
                title=source.title,
                dataset_id=source.dataset_id,
                source=source.source,
                decision="fallback",
                reason=source.why_selected,
                score=source.usefulness_confidence,
            )
            for source in fallback_sources
        )

    direct_dataset = _direct_dataset_candidate(intent, selected)
    predecessor_datasets = selected if not direct_dataset else [direct_dataset]
    has_local_data = any(_has_local_data_path(source) for source in predecessor_datasets)
    blocking_reasons = _blocking_reasons(predecessor_datasets, has_local_data)

    return DatasetBuildPlan(
        target_dataset_name=_target_dataset_name(intent),
        registry_checked=use_registry and registry_assets["catalog_records"],
        registry_assets=registry_assets,
        can_use_existing_dataset=direct_dataset is not None,
        can_build_from_local_sources=has_local_data,
        selected_datasets=[direct_dataset] if direct_dataset else [],
        predecessor_datasets=predecessor_datasets,
        source_audit=audit,
        assembly_steps=_assembly_steps(structure, predecessor_datasets),
        warnings=warnings,
        blocking_reasons=blocking_reasons,
    )


def registry_assets_status() -> dict[str, bool]:
    return {
        "catalog_records": CATALOG_RECORDS_PATH.exists(),
        "bm25_index": BM25_INDEX_PATH.exists(),
        "chroma_index": CHROMA_DIR.exists(),
    }


def _retrieve_registry_candidates(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
) -> tuple[list[dict[str, Any]], list[str]]:
    query = _build_registry_query(intent, design)
    try:
        from hybrid_candidate_retriever import retrieve_candidate_datasets

        return retrieve_candidate_datasets(query), []
    except Exception as exc:  # noqa: BLE001 - this is a prototype fallback boundary.
        return [], [f"RAG retrieval недоступен: {exc}"]


def _select_registry_candidates(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
    candidates: list[dict[str, Any]],
    settings: LLMSettings | None,
    provider: str | None,
    model: str | None,
) -> tuple[list[SourceDatasetReference], list[str]]:
    warnings: list[str] = []
    try:
        from dataset_reranker import rerank_dataset_candidates

        response = rerank_dataset_candidates(
            user_query=_build_registry_query(intent, design),
            candidates=candidates,
            settings=settings,
            provider=provider,
            model=model,
        )
        if response.results:
            return [
                SourceDatasetReference(
                    record_id=result.record_id,
                    dataset_id=result.dataset_id,
                    title=result.title or result.record_id,
                    source=result.source,
                    source_name=result.source,
                    data_path=result.data_path,
                    source_url=result.source_url,
                    role="predecessor",
                    usefulness_confidence=result.usefulness_confidence,
                    why_selected=result.why_matched,
                    possible_limitations=result.possible_limitations,
                )
                for result in response.results
            ], warnings
        if response.no_results_reason:
            warnings.append(response.no_results_reason)
    except Exception as exc:  # noqa: BLE001 - LLM rerank can fail without keys or network.
        warnings.append(f"LLM reranker недоступен, применен эвристический отбор: {exc}")

    return _heuristic_select_candidates(intent, design, candidates), warnings


def _heuristic_select_candidates(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
    candidates: list[dict[str, Any]],
    limit: int = 5,
) -> list[SourceDatasetReference]:
    query_tokens = _query_tokens(intent, design)
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        if candidate.get("is_invalid"):
            continue
        text = _candidate_text(candidate)
        tokens = _tokens(text)
        overlap = len(query_tokens & tokens)
        bonus = _candidate_bonus(intent, candidate)
        penalty = _candidate_penalty(intent, candidate)
        title = str(candidate.get("title") or "").lower()
        for indicator in intent.indicators:
            if indicator.lower() in title:
                bonus += 2
        score = overlap + bonus - penalty
        if score > 0:
            scored.append((float(score), candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return []

    max_score = scored[0][0]
    threshold = max_score * 0.7
    return [
        SourceDatasetReference(
            record_id=_optional_text(candidate.get("record_id")),
            dataset_id=_optional_text(candidate.get("dataset_id")),
            title=_optional_text(candidate.get("title")) or _optional_text(candidate.get("record_id")) or "Dataset",
            source=_optional_text(candidate.get("source")),
            source_name=_optional_text(candidate.get("source_name")),
            data_path=_optional_text(candidate.get("data_path")),
            source_url=_optional_text(candidate.get("source_url")),
            role="predecessor",
            usefulness_confidence=min(1.0, score / max(max_score, 1.0)),
            why_selected="Эвристический отбор по пересечению слов запроса с metadata.",
            possible_limitations=_candidate_limitations(candidate),
        )
        for score, candidate in scored[:limit]
        if score >= threshold
    ]


def _candidate_bonus(intent: ResearchIntent, candidate: dict[str, Any]) -> int:
    title = str(candidate.get("title") or "").lower()
    bonus = 0
    for indicator in intent.indicators:
        indicator_lower = indicator.lower()
        if indicator_lower == "врп" and "валовой региональный продукт" in title:
            bonus += 8
        if indicator_lower == "ввп" and "валовой внутренний продукт" in title:
            bonus += 8
    if "оквэд 2" in title or "оквэд2" in title:
        bonus += 2
    return bonus


def _candidate_penalty(intent: ResearchIntent, candidate: dict[str, Any]) -> int:
    query = intent.original_query.lower()
    title = str(candidate.get("title") or "").lower()
    penalty = 0
    if "на душу" in title and "на душу" not in query:
        penalty += 5
    if "индекс" in title and "индекс" not in query:
        penalty += 4
    if "структура" in title and "структур" not in query:
        penalty += 4
    if "доля" in title and "доля" not in query:
        penalty += 4
    if intent.time_range and intent.time_range.end_year and intent.time_range.end_year > 2016:
        if "оквэд-2007" in title or "оквэд 2007" in title:
            penalty += 2
    return penalty


def _fallback_source_candidates(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
) -> list[SourceDatasetReference]:
    sources: list[SourceDatasetReference] = []
    seen: set[str] = set()

    for candidate in intent.source_candidates:
        key = (candidate.name, candidate.dataset_or_indicator)
        if str(key) in seen:
            continue
        seen.add(str(key))
        sources.append(
            SourceDatasetReference(
                title=candidate.dataset_or_indicator or candidate.name,
                source=candidate.name,
                source_name=candidate.name,
                role=candidate.role or "candidate_source",
                usefulness_confidence=0.45,
                why_selected=(
                    candidate.notes
                    or "Источник предложен на этапе формализации запроса, но не проверен в локальном реестре."
                ),
                possible_limitations=[
                    "Источник не был подтвержден через локальный RAG-реестр.",
                ],
            )
        )

    for measurement in design.required_measurements:
        for source_name in measurement.source_candidates:
            key = (source_name, measurement.name)
            if str(key) in seen:
                continue
            seen.add(str(key))
            sources.append(
                SourceDatasetReference(
                    title=measurement.name,
                    source=source_name,
                    source_name=source_name,
                    role="measurement_source",
                    usefulness_confidence=0.4,
                    why_selected=(
                        "Источник указан в дизайне исследования как кандидат для измерения "
                        f"'{measurement.name}'."
                    ),
                    possible_limitations=[
                        "Источник нужно проверить вручную или через наполненный локальный реестр.",
                    ],
                )
            )

    return sources


def _audit_retrieved_candidates(candidates: list[dict[str, Any]], limit: int = 20) -> list[SourceAuditEntry]:
    entries: list[SourceAuditEntry] = []
    for candidate in candidates[:limit]:
        decision = "rejected" if candidate.get("is_invalid") else "considered"
        reason = (
            "Запись помечена как invalid в catalog_records.jsonl."
            if candidate.get("is_invalid")
            else "Кандидат найден retrieval-этапом и передан на ранжирование."
        )
        score = candidate.get("vector_score", candidate.get("bm25_score"))
        entries.append(
            SourceAuditEntry(
                title=_optional_text(candidate.get("title")) or _optional_text(candidate.get("record_id")) or "Dataset",
                record_id=_optional_text(candidate.get("record_id")),
                dataset_id=_optional_text(candidate.get("dataset_id")),
                source=_optional_text(candidate.get("source")),
                decision=decision,
                reason=reason,
                matched_by=[
                    item for item in candidate.get("matched_by", []) if isinstance(item, str)
                ],
                score=float(score) if isinstance(score, (int, float)) else None,
            )
        )
    return entries


def _direct_dataset_candidate(
    intent: ResearchIntent,
    selected: list[SourceDatasetReference],
) -> SourceDatasetReference | None:
    if len(selected) != 1:
        return None
    candidate = selected[0]
    text = f"{candidate.title} {candidate.why_selected}".lower()
    indicators = [indicator.lower() for indicator in intent.indicators]
    if indicators and all(indicator in text for indicator in indicators):
        return SourceDatasetReference.model_validate(
            {
                **candidate.model_dump(mode="json"),
                "role": "ready_dataset",
                "why_selected": (
                    candidate.why_selected
                    + " Кандидат выглядит как готовый датасет для целевого показателя."
                ),
            }
        )
    return None


def _assembly_steps(
    structure: TargetDatasetStructure,
    sources: list[SourceDatasetReference],
) -> list[AssemblyStep]:
    source_titles = [source.title for source in sources]
    columns = [column.name for column in structure.columns]
    return [
        AssemblyStep(
            id="inspect_sources",
            title="Проверить источники",
            description="Зафиксировать выбранные и отвергнутые источники, проверить наличие локальных файлов и ссылок.",
            inputs=source_titles,
            outputs=["source_manifest.json"],
        ),
        AssemblyStep(
            id="load_predecessors",
            title="Загрузить датасеты-предшественники",
            description="Прочитать доступные CSV/JSON/JSONL-файлы или оставить шаблон, если локальных файлов нет.",
            inputs=source_titles,
            outputs=["raw_records"],
        ),
        AssemblyStep(
            id="normalize_grain",
            title="Привести зернистость",
            description=f"Привести данные к зернистости '{structure.row_grain}' и ключу {', '.join(structure.primary_key)}.",
            inputs=["raw_records"],
            outputs=["normalized_records"],
        ),
        AssemblyStep(
            id="compute_metrics",
            title="Рассчитать показатели",
            description="Сопоставить исходные поля с целевыми индикаторами и рассчитать производные метрики по формулам из дизайна.",
            inputs=["normalized_records"],
            outputs=columns,
        ),
        AssemblyStep(
            id="validate_and_export",
            title="Проверить и выгрузить",
            description="Проверить обязательные ключи, записать итоговый CSV и JSON с metadata/lineage.",
            inputs=columns,
            outputs=["target_dataset.csv", "target_dataset.metadata.json"],
        ),
    ]


def _blocking_reasons(
    predecessor_datasets: list[SourceDatasetReference],
    has_local_data: bool,
) -> list[str]:
    if not predecessor_datasets:
        return [
            "Не найдено ни готового датасета, ни датасетов-предшественников; нужен наполненный реестр или ручной источник."
        ]
    if not has_local_data:
        return [
            "У выбранных источников нет локального data_path; сгенерированный скрипт создаст шаблон и metadata, но не соберет фактические наблюдения."
        ]
    return []


def _build_registry_query(intent: ResearchIntent, design: ResearchStudyDesign) -> str:
    parts = [
        intent.original_query,
        intent.topic,
        " ".join(intent.geography),
        " ".join(intent.indicators),
        " ".join(measurement.name for measurement in design.required_measurements),
        intent.frequency,
    ]
    return " ".join(part for part in parts if part)


def _query_tokens(intent: ResearchIntent, design: ResearchStudyDesign) -> set[str]:
    return _tokens(_build_registry_query(intent, design))


def _candidate_text(candidate: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in (
        "record_id",
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
    ):
        value = candidate.get(field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 2
    }


def _candidate_limitations(candidate: dict[str, Any]) -> list[str]:
    limitations = []
    if not candidate.get("data_path"):
        limitations.append("В metadata нет локального data_path.")
    if not candidate.get("source_url"):
        limitations.append("В metadata нет source_url.")
    if candidate.get("limitations"):
        limitations.append(str(candidate["limitations"]))
    return limitations


def _has_local_data_path(source: SourceDatasetReference) -> bool:
    if not source.data_path:
        return False
    path = Path(source.data_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.exists()


def _target_dataset_name(intent: ResearchIntent) -> str:
    topic = intent.topic or intent.original_query
    clean = " ".join(topic.split())
    return clean[:120] or "target_dataset"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
