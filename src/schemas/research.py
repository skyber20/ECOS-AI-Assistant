from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


GeoType = Literal[
    "country",
    "country_group",
    "federal_subject",
    "region",
    "city",
    "municipality",
    "organization",
    "custom_area",
    "unknown",
]

GeoRole = Literal[
    "target",
    "counterpart",
    "comparison",
    "filter",
    "unknown",
]

TimeGranularity = Literal[
    "day",
    "month",
    "quarter",
    "year",
    "multi_year",
    "unknown",
]

TaskType = Literal[
    "build_dataset",
    "find_value",
    "trend_analysis",
    "comparison",
    "relationship_analysis",
    "overview",
    "unknown",
]

AmbiguityField = Literal[
    "topic",
    "geography",
    "time",
    "indicator",
    "unit",
    "methodology",
    "granularity",
    "other",
]

Severity = Literal["blocking", "non_blocking"]


class GeoEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str
    name: str
    type: GeoType = "unknown"
    country: str | None = None
    role: GeoRole = "unknown"


class TimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str | None = None

    # Use strings because research queries often contain years, quarters,
    # months, open intervals, or incomplete dates.
    start: str | None = None
    end: str | None = None
    points: list[str] = Field(default_factory=list)

    granularity: TimeGranularity = "unknown"
    is_missing: bool = False


class IndicatorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str
    canonical_name: str
    display_name: str

    is_explicit: bool = True
    kind: Literal["direct", "derived", "unknown"] = "unknown"

    unit: str | None = None
    aggregation: Literal[
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "rate",
        "index",
        "none",
        "unknown",
    ] = "unknown"

    time_granularity: TimeGranularity | None = None


class Ambiguity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: AmbiguityField
    issue: str
    severity: Severity
    question: str | None = None
    default_assumption: str | None = None


class FormalizedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str
    task_type: TaskType = "unknown"
    goal: str

    geographies: list[GeoEntity] = Field(default_factory=list)
    time: TimeSpec

    indicators: list[IndicatorRequest] = Field(default_factory=list)

    # Example: ["country", "year"], ["region", "month"], ["country", "partner_country", "year"]
    row_grain: list[str] = Field(default_factory=list)

    # Extra analytical dimensions: industry, product, sex, age_group, sector, etc.
    dimensions: list[str] = Field(default_factory=list)

    ambiguities: list[Ambiguity] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    def has_blocking_ambiguity(self) -> bool:
        return any(item.severity == "blocking" for item in self.ambiguities)
