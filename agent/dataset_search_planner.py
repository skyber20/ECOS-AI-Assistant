from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from agent.intent_parser import ResearchIntent, none_to_empty_list
    from agent.research_designer import ResearchStudyDesign
    from agent.target_dataset_designer import DatasetColumn, TargetDatasetStructure
except ImportError:  # pragma: no cover - keeps direct `python agent/...` runs working.
    from intent_parser import ResearchIntent, none_to_empty_list  # type: ignore
    from research_designer import ResearchStudyDesign  # type: ignore
    from target_dataset_designer import DatasetColumn, TargetDatasetStructure  # type: ignore


class DatasetMatchStatus(str, Enum):
    READY = "ready"
    PARTIAL = "partial"
    PREDECESSOR = "predecessor"
    NOT_MATCH = "not_match"


class BuildStrategy(str, Enum):
    USE_READY_DATASET = "use_ready_dataset"
    ASSEMBLE_FROM_PREDECESSORS = "assemble_from_predecessors"
    NEEDS_SOURCE_DISCOVERY = "needs_source_discovery"
    BLOCKED = "blocked"


class DatasetSearchColumnRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    data_type: str
    unit: str | None = None
    definition: str
    is_required: bool = True
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("depends_on", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class DatasetSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str
    target_dataset_name: str
    row_grain: str
    primary_key: list[str] = Field(default_factory=list)
    time_coverage: str | None = None
    geography_coverage: list[str] = Field(default_factory=list)
    frequency: str | None = None
    required_dimensions: list[DatasetSearchColumnRequirement] = Field(default_factory=list)
    required_indicators: list[DatasetSearchColumnRequirement] = Field(default_factory=list)
    required_derived_metrics: list[DatasetSearchColumnRequirement] = Field(default_factory=list)
    required_metadata_columns: list[DatasetSearchColumnRequirement] = Field(default_factory=list)
    source_requirements: list[str] = Field(default_factory=list)
    validation_rules: list[str] = Field(default_factory=list)
    can_build_target: bool = True

    @field_validator(
        "primary_key",
        "geography_coverage",
        "required_dimensions",
        "required_indicators",
        "required_derived_metrics",
        "required_metadata_columns",
        "source_requirements",
        "validation_rules",
        mode="before",
    )
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class DatasetCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    title: str
    description: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    row_grain: str | None = None
    time_coverage: str | None = None
    geography_coverage: list[str] = Field(default_factory=list)
    frequency: str | None = None
    columns: list[str] = Field(default_factory=list)
    indicators: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    join_keys: list[str] = Field(default_factory=list)
    match_status: DatasetMatchStatus = DatasetMatchStatus.PARTIAL
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage_notes: list[str] = Field(default_factory=list)
    missing_required_columns: list[str] = Field(default_factory=list)

    @field_validator(
        "geography_coverage",
        "columns",
        "indicators",
        "dimensions",
        "join_keys",
        "coverage_notes",
        "missing_required_columns",
        mode="before",
    )
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class FieldMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_column: str
    target_role: str
    source_dataset_id: str | None = None
    source_column: str | None = None
    transformation: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_manual_mapping: bool = False


class BuildStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int
    operation: str
    description: str
    input_datasets: list[str] = Field(default_factory=list)
    output_dataset: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input_datasets", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class DatasetBuildPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: BuildStrategy
    can_build: bool
    ready_dataset_id: str | None = None
    source_dataset_ids: list[str] = Field(default_factory=list)
    predecessor_requirements: list[str] = Field(default_factory=list)
    field_mappings: list[FieldMapping] = Field(default_factory=list)
    steps: list[BuildStep] = Field(default_factory=list)
    transformations: list[str] = Field(default_factory=list)
    validation_checks: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    user_review_required: bool = True
    notes: str | None = None

    @field_validator(
        "source_dataset_ids",
        "predecessor_requirements",
        "field_mappings",
        "steps",
        "transformations",
        "validation_checks",
        "blocking_reasons",
        "assumptions",
        mode="before",
    )
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class DatasetMatchReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_request: DatasetSearchRequest
    candidates: list[DatasetCandidate] = Field(default_factory=list)
    selected_candidate: DatasetCandidate | None = None
    coverage_summary: str
    build_plan: DatasetBuildPlan
    registry_backend: str
    rag_required: bool = True

    @field_validator("candidates", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        return none_to_empty_list(value)


class DatasetRegistry(Protocol):
    backend_name: str

    def search(self, request: DatasetSearchRequest) -> list[DatasetCandidate]:
        """Return registry candidates for a target dataset request."""


class EmptyDatasetRegistry:
    backend_name = "empty_registry"

    def search(self, request: DatasetSearchRequest) -> list[DatasetCandidate]:
        return []


class InMemoryDatasetRegistry:
    backend_name = "in_memory_registry"

    def __init__(self, candidates: list[DatasetCandidate]) -> None:
        self.candidates = candidates

    def search(self, request: DatasetSearchRequest) -> list[DatasetCandidate]:
        scored = [_score_candidate(candidate, request) for candidate in self.candidates]
        return sorted(scored, key=lambda item: item.coverage_score, reverse=True)


def create_dataset_match_report(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
    target: TargetDatasetStructure,
    registry: DatasetRegistry | None = None,
) -> DatasetMatchReport:
    request = build_dataset_search_request(intent, design, target)
    registry = registry or EmptyDatasetRegistry()
    candidates = registry.search(request)
    selected_candidate = _select_candidate(candidates)
    build_plan = _build_dataset_plan(
        request=request,
        target=target,
        candidates=candidates,
        selected_candidate=selected_candidate,
    )

    return DatasetMatchReport(
        search_request=request,
        candidates=candidates,
        selected_candidate=selected_candidate,
        coverage_summary=_build_coverage_summary(request, candidates, selected_candidate),
        build_plan=build_plan,
        registry_backend=registry.backend_name,
        rag_required=build_plan.strategy == BuildStrategy.NEEDS_SOURCE_DISCOVERY
        or (not build_plan.can_build and build_plan.strategy != BuildStrategy.BLOCKED),
    )


def build_dataset_search_request(
    intent: ResearchIntent,
    design: ResearchStudyDesign,
    target: TargetDatasetStructure,
) -> DatasetSearchRequest:
    return DatasetSearchRequest(
        original_query=target.original_query or intent.original_query,
        target_dataset_name=target.dataset_name,
        row_grain=target.row_grain,
        primary_key=target.primary_key,
        time_coverage=target.time_coverage or _time_range_to_text(intent),
        geography_coverage=target.geography_coverage or intent.geography,
        frequency=target.frequency or intent.frequency,
        required_dimensions=_column_requirements(target.dimensions),
        required_indicators=_column_requirements(target.indicators),
        required_derived_metrics=_column_requirements(target.derived_metrics),
        required_metadata_columns=_column_requirements(target.metadata_columns),
        source_requirements=_unique_texts(
            [
                *target.source_requirements,
                *[
                    item.name
                    for item in intent.source_candidates
                    if item.name
                ],
                *[
                    source
                    for measurement in design.required_measurements
                    for source in measurement.source_candidates
                ],
            ]
        ),
        validation_rules=target.validation_rules,
        can_build_target=target.can_build,
    )


def _column_requirements(
    columns: list[DatasetColumn],
) -> list[DatasetSearchColumnRequirement]:
    return [
        DatasetSearchColumnRequirement(
            name=column.name,
            role=column.role,
            data_type=column.data_type,
            unit=column.unit,
            definition=column.definition,
            is_required=column.is_required,
            depends_on=column.depends_on,
        )
        for column in columns
    ]


def _build_dataset_plan(
    request: DatasetSearchRequest,
    target: TargetDatasetStructure,
    candidates: list[DatasetCandidate],
    selected_candidate: DatasetCandidate | None,
) -> DatasetBuildPlan:
    validation_checks = _unique_texts(
        [
            *request.validation_rules,
            "Проверить уникальность primary_key итогового датасета.",
            "Проверить обязательные колонки и типы данных перед экспортом.",
        ]
    )

    if not request.can_build_target:
        return DatasetBuildPlan(
            strategy=BuildStrategy.BLOCKED,
            can_build=False,
            validation_checks=validation_checks,
            blocking_reasons=target.blocking_reasons
            or ["Структура целевого датасета помечена как невозможная к сборке."],
            notes="План сборки заблокирован предыдущим этапом.",
        )

    if selected_candidate and selected_candidate.match_status == DatasetMatchStatus.READY:
        return DatasetBuildPlan(
            strategy=BuildStrategy.USE_READY_DATASET,
            can_build=True,
            ready_dataset_id=selected_candidate.dataset_id,
            source_dataset_ids=[selected_candidate.dataset_id],
            field_mappings=_build_field_mappings(request, selected_candidate),
            steps=_ready_dataset_steps(request, selected_candidate),
            transformations=[
                "Отфильтровать готовый датасет по требуемой географии, периоду и частоте.",
                "Привести имена и типы колонок к TargetDatasetStructure.",
            ],
            validation_checks=validation_checks,
            assumptions=[
                "Готовый датасет считается пригодным только после проверки покрытия и лицензии.",
            ],
            user_review_required=False,
            notes="Найден кандидат, который покрывает обязательные поля целевого датасета.",
        )

    if candidates:
        source_dataset_ids = [candidate.dataset_id for candidate in candidates[:5]]
        missing = _unique_texts(
            [
                missing
                for candidate in candidates
                for missing in candidate.missing_required_columns
            ]
        )
        return DatasetBuildPlan(
            strategy=BuildStrategy.ASSEMBLE_FROM_PREDECESSORS,
            can_build=not missing,
            source_dataset_ids=source_dataset_ids,
            predecessor_requirements=_predecessor_requirements(request, missing),
            field_mappings=_build_field_mappings(request, selected_candidate),
            steps=_assembly_steps(request, source_dataset_ids),
            transformations=_assembly_transformations(request),
            validation_checks=validation_checks,
            blocking_reasons=(
                [
                    "Для сборки не хватает обязательных полей: "
                    f"{', '.join(missing)}."
                ]
                if missing
                else []
            ),
            assumptions=[
                "Частичные кандидаты требуют проверки методологии и сопоставления колонок.",
            ],
            user_review_required=True,
            notes="Найдены частичные кандидаты, из которых можно собрать целевую таблицу.",
        )

    unresolved_requirements = _predecessor_requirements(request, [])
    return DatasetBuildPlan(
        strategy=BuildStrategy.NEEDS_SOURCE_DISCOVERY,
        can_build=False,
        predecessor_requirements=unresolved_requirements,
        field_mappings=_build_field_mappings(request, None),
        steps=_assembly_steps(request, []),
        transformations=_assembly_transformations(request),
        validation_checks=validation_checks,
        blocking_reasons=[
            "В текущем реестре не найден готовый датасет или датасеты-предшественники."
        ],
        assumptions=[
            "Когда RAG/реестр будет подключен, этот план нужно повторно прогнать через поиск.",
        ],
        user_review_required=True,
        notes="Это предварительный план сборки без опоры на конкретный реестр данных.",
    )


def _score_candidate(
    candidate: DatasetCandidate,
    request: DatasetSearchRequest,
) -> DatasetCandidate:
    required_columns = _required_column_names(request)
    available = _normal_set(
        [
            *candidate.columns,
            *candidate.indicators,
            *candidate.dimensions,
        ]
    )
    missing = [name for name in required_columns if _normalize(name) not in available]
    covered_count = len(required_columns) - len(missing)
    field_score = covered_count / len(required_columns) if required_columns else 0.0

    row_score = 1.0 if _compatible(candidate.row_grain, request.row_grain) else 0.0
    frequency_score = 1.0 if _compatible(candidate.frequency, request.frequency) else 0.0
    geography_score = _geography_score(candidate.geography_coverage, request.geography_coverage)
    coverage_score = (field_score * 0.7) + (row_score * 0.1) + (frequency_score * 0.1) + (
        geography_score * 0.1
    )

    if not missing and coverage_score >= 0.85:
        status = DatasetMatchStatus.READY
    elif covered_count > 0:
        status = DatasetMatchStatus.PARTIAL
    else:
        status = DatasetMatchStatus.PREDECESSOR

    return candidate.model_copy(
        update={
            "match_status": status,
            "coverage_score": round(coverage_score, 4),
            "missing_required_columns": missing,
            "coverage_notes": _coverage_notes(candidate, request, missing),
        }
    )


def _select_candidate(candidates: list[DatasetCandidate]) -> DatasetCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.coverage_score)


def _build_field_mappings(
    request: DatasetSearchRequest,
    candidate: DatasetCandidate | None,
) -> list[FieldMapping]:
    available = (
        _normal_set([*candidate.columns, *candidate.indicators, *candidate.dimensions])
        if candidate
        else set()
    )
    source_dataset_id = candidate.dataset_id if candidate else None

    mappings: list[FieldMapping] = []
    for column in _all_required_columns(request):
        direct_match = _normalize(column.name) in available
        mappings.append(
            FieldMapping(
                target_column=column.name,
                target_role=column.role,
                source_dataset_id=source_dataset_id if direct_match else None,
                source_column=column.name if direct_match else None,
                transformation=(
                    "Прямое переименование/приведение типа из найденного датасета."
                    if direct_match
                    else "Нужно найти источник и сопоставить поле через реестр/RAG."
                ),
                confidence=0.9 if direct_match else 0.0,
                requires_manual_mapping=not direct_match,
            )
        )
    return mappings


def _ready_dataset_steps(
    request: DatasetSearchRequest,
    candidate: DatasetCandidate,
) -> list[BuildStep]:
    return [
        BuildStep(
            order=1,
            operation="load_ready_dataset",
            description="Загрузить готовый датасет-кандидат из реестра.",
            input_datasets=[candidate.dataset_id],
            output_dataset="raw_ready_dataset",
            parameters={"dataset_id": candidate.dataset_id},
        ),
        BuildStep(
            order=2,
            operation="filter_scope",
            description="Отфильтровать данные по периоду, географии и частоте целевого датасета.",
            input_datasets=["raw_ready_dataset"],
            output_dataset="scoped_dataset",
            parameters={
                "time_coverage": request.time_coverage,
                "geography_coverage": request.geography_coverage,
                "frequency": request.frequency,
            },
        ),
        BuildStep(
            order=3,
            operation="standardize_schema",
            description="Привести имена, типы и единицы измерения к контракту TargetDatasetStructure.",
            input_datasets=["scoped_dataset"],
            output_dataset=request.target_dataset_name,
        ),
        BuildStep(
            order=4,
            operation="validate_output",
            description="Проверить итоговый датасет по validation_rules и будущей Pandera-схеме.",
            input_datasets=[request.target_dataset_name],
            output_dataset=request.target_dataset_name,
        ),
    ]


def _assembly_steps(
    request: DatasetSearchRequest,
    source_dataset_ids: list[str],
) -> list[BuildStep]:
    steps = [
        BuildStep(
            order=1,
            operation="discover_sources",
            description="Найти в реестре готовые источники или датасеты-предшественники для обязательных полей.",
            input_datasets=source_dataset_ids,
            output_dataset="resolved_source_plan",
            parameters={"source_requirements": request.source_requirements},
        ),
        BuildStep(
            order=2,
            operation="load_predecessors",
            description="Загрузить выбранные датасеты-предшественники.",
            input_datasets=source_dataset_ids,
            output_dataset="raw_predecessors",
        ),
        BuildStep(
            order=3,
            operation="standardize_dimensions",
            description="Нормализовать ключи времени, географии и объектов наблюдения.",
            input_datasets=["raw_predecessors"],
            output_dataset="standardized_predecessors",
            parameters={"primary_key": request.primary_key},
        ),
        BuildStep(
            order=4,
            operation="join_sources",
            description="Соединить показатели и измерения на уровне зернистости целевого датасета.",
            input_datasets=["standardized_predecessors"],
            output_dataset="joined_dataset",
            parameters={"row_grain": request.row_grain},
        ),
    ]

    next_order = 5
    if request.required_derived_metrics:
        steps.append(
            BuildStep(
                order=next_order,
                operation="calculate_derived_metrics",
                description="Рассчитать производные метрики по формулам из контракта целевого датасета.",
                input_datasets=["joined_dataset"],
                output_dataset="dataset_with_derived_metrics",
            )
        )
        next_order += 1
        validation_input = "dataset_with_derived_metrics"
    else:
        validation_input = "joined_dataset"

    steps.extend(
        [
            BuildStep(
                order=next_order,
                operation="validate_output",
                description="Проверить итоговый датасет по validation_rules и будущей Pandera-схеме.",
                input_datasets=[validation_input],
                output_dataset="validated_dataset",
            ),
            BuildStep(
                order=next_order + 1,
                operation="export_dataset",
                description="Сохранить итоговый датасет с метаданными и ссылками на источники.",
                input_datasets=["validated_dataset"],
                output_dataset=request.target_dataset_name,
            ),
        ]
    )
    return steps


def _assembly_transformations(request: DatasetSearchRequest) -> list[str]:
    transformations = [
        "Привести географические и временные ключи к единому формату.",
        "Привести единицы измерения и типы данных к контракту целевого датасета.",
    ]
    if request.required_derived_metrics:
        transformations.append("Рассчитать производные метрики после соединения базовых показателей.")
    if request.frequency:
        transformations.append(f"Согласовать частоту наблюдений: {request.frequency}.")
    return transformations


def _predecessor_requirements(
    request: DatasetSearchRequest,
    missing_columns: list[str],
) -> list[str]:
    required_columns = missing_columns or [
        column.name
        for column in [
            *request.required_indicators,
            *request.required_dimensions,
            *request.required_derived_metrics,
        ]
        if column.is_required
    ]
    requirements = [
        f"Найти источник/датасет-предшественник для поля '{name}'."
        for name in required_columns
    ]
    requirements.extend(
        f"Учесть требование к источникам: {requirement}"
        for requirement in request.source_requirements
    )
    return _unique_texts(requirements)


def _build_coverage_summary(
    request: DatasetSearchRequest,
    candidates: list[DatasetCandidate],
    selected_candidate: DatasetCandidate | None,
) -> str:
    if not request.can_build_target:
        return "Поиск не выполнялся как готовая сборка: целевой датасет заблокирован предыдущим этапом."
    if not candidates:
        return "Реестр не вернул кандидатов; нужен RAG/поиск по каталогу датасетов."
    if selected_candidate is None:
        return "Кандидаты найдены, но подходящий датасет не выбран."
    return (
        f"Лучший кандидат: {selected_candidate.dataset_id}, "
        f"статус {selected_candidate.match_status.value}, "
        f"покрытие {selected_candidate.coverage_score:.2f}."
    )


def _coverage_notes(
    candidate: DatasetCandidate,
    request: DatasetSearchRequest,
    missing: list[str],
) -> list[str]:
    notes: list[str] = []
    if missing:
        notes.append(f"Не покрыты обязательные поля: {', '.join(missing)}.")
    if not _compatible(candidate.row_grain, request.row_grain):
        notes.append("Зернистость кандидата отличается от целевого датасета.")
    if not _compatible(candidate.frequency, request.frequency):
        notes.append("Частота кандидата отличается от целевой частоты.")
    if _geography_score(candidate.geography_coverage, request.geography_coverage) < 1.0:
        notes.append("Географическое покрытие требует проверки.")
    return notes


def _all_required_columns(
    request: DatasetSearchRequest,
) -> list[DatasetSearchColumnRequirement]:
    return [
        *request.required_dimensions,
        *request.required_indicators,
        *request.required_derived_metrics,
        *request.required_metadata_columns,
    ]


def _required_column_names(request: DatasetSearchRequest) -> list[str]:
    return [column.name for column in _all_required_columns(request) if column.is_required]


def _time_range_to_text(intent: ResearchIntent) -> str | None:
    if intent.time_range is None:
        return None
    if intent.time_range.raw:
        return intent.time_range.raw
    if intent.time_range.start_year and intent.time_range.end_year:
        return f"{intent.time_range.start_year}-{intent.time_range.end_year}"
    if intent.time_range.start_year:
        return f"с {intent.time_range.start_year}"
    if intent.time_range.end_year:
        return f"до {intent.time_range.end_year}"
    return None


def _geography_score(candidate_values: list[str], request_values: list[str]) -> float:
    if not request_values:
        return 1.0
    if not candidate_values:
        return 0.0
    candidate_normalized = _normal_set(candidate_values)
    request_normalized = _normal_set(request_values)
    covered = len(candidate_normalized.intersection(request_normalized))
    return covered / len(request_normalized)


def _compatible(candidate_value: str | None, request_value: str | None) -> bool:
    if not request_value:
        return True
    if not candidate_value:
        return False
    return _normalize(candidate_value) == _normalize(request_value)


def _normal_set(values: list[str]) -> set[str]:
    return {_normalize(value) for value in values if value}


def _normalize(value: str) -> str:
    return value.casefold().replace(" ", "_").replace("-", "_")


def _unique_texts(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result
