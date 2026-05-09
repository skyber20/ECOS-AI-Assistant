from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


DimensionType = Literal[
    "industry",
    "product",
    "sex",
    "age_group",
    "sector",
    "ownership",
    "size_class",
    "other",
]

ChartType = Literal[
    "line",
    "bar",
    "stacked_bar",
    "grouped_bar",
    "scatter",
    "bubble",
    "heatmap",
    "choropleth",
    "pie",
    "donut",
    "waterfall",
    "box_plot",
    "radar",
    "combo",
    "table",
    "other",
]

StatMethod = Literal[
    "descriptive",
    "correlation",
    "regression",
    "t_test",
    "anova",
    "chi_square",
    "decomposition",
    "clustering",
    "index_calculation",
    "trend_analysis",
    "other",
]

NormMethod = Literal[
    "min_max",
    "z_score",
    "per_capita",
    "per_area",
    "percentage",
    "log",
    "index_base_year",
    "none",
]

AggFunc = Literal[
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
    "weighted_average",
    "last",
    "first",
    "none",
]


class Hypothesis(BaseModel):
    """Исследовательская гипотеза"""
    model_config = ConfigDict(extra="forbid")
    
    id: str = Field(description="Короткий идентификатор: H1, H2, ...")
    statement: str = Field(description="Формулировка гипотезы")
    null_hypothesis: str = Field(description="Нулевая гипотеза (H0)")
    direction: Literal["positive", "negative", "u_shaped", "n_shaped", "none"] = Field(
        description="Ожидаемое направление связи"
    )
    variables: list[str] = Field(description="Переменные, участвующие в проверке")
    expected_effect_size: Literal["small", "medium", "large", "unknown"] = "unknown"


class Measurement(BaseModel):
    """Необходимое измерение/показатель"""
    model_config = ConfigDict(extra="forbid")
    
    canonical_name: str = Field(description="Каноническое название (как в IndicatorRequest)")
    display_name: str = Field(description="Человекочитаемое название")
    unit: str = Field(description="Единица измерения")
    source: str = Field(
        default="Росстат",
        description="Ожидаемый источник данных: Росстат, ЦБ РФ, ведомство..."
    )
    is_critical: bool = Field(
        default=True,
        description="Критичен ли показатель для исследования"
    )
    alternative_sources: list[str] = Field(
        default_factory=list,
        description="Альтернативные источники, если основной недоступен"
    )


class DerivedMetric(BaseModel):
    """Производная метрика с формулой"""
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(description="Название метрики")
    display_name: str = Field(description="Отображаемое название")
    formula: str = Field(description="Формула в LaTeX или псевдокоде")
    formula_explanation: str = Field(description="Текстовое объяснение формулы")
    
    source_indicators: list[str] = Field(
        description="Исходные показатели (canonical_name)"
    )
    normalization: NormMethod = Field(
        default="none",
        description="Метод нормализации"
    )
    aggregation_rule: AggFunc = Field(
        default="none",
        description="Правило агрегирования"
    )
    
    unit: str = Field(description="Результирующая единица измерения")
    interpretation: str = Field(description="Как интерпретировать значения")


class GroupingRule(BaseModel):
    """Правило группировки данных"""
    model_config = ConfigDict(extra="forbid")
    
    dimension: str = Field(description="Измерение для группировки")
    dimension_type: DimensionType = Field(description="Тип измерения")
    levels: list[str] = Field(
        default_factory=list,
        description="Конкретные уровни группировки"
    )
    rationale: str = Field(description="Обоснование группировки")
    min_observations: Optional[int] = Field(
        default=None,
        description="Минимальное число наблюдений в группе"
    )


class VisualizationSpec(BaseModel):
    """Спецификация визуализации"""
    model_config = ConfigDict(extra="forbid")
    
    id: str = Field(description="Идентификатор: V1, V2, ...")
    title: str = Field(description="Заголовок графика")
    chart_type: ChartType = Field(description="Тип графика")
    
    x_axis: str = Field(description="Переменная/измерение для оси X")
    y_axis: str = Field(description="Переменная/измерение для оси Y")
    grouping: Optional[str] = Field(
        default=None,
        description="Переменная для группировки/цвета"
    )
    facet: Optional[str] = Field(
        default=None,
        description="Переменная для фасетирования (разбивки на подграфики)"
    )
    
    metrics_used: list[str] = Field(
        description="Какие метрики визуализируются (включая производные)"
    )
    hypothesis_id: Optional[str] = Field(
        default=None,
        description="Какую гипотезу проверяет (H1, H2, ...)"
    )
    
    description: str = Field(description="Что показывает и какой инсайт ожидается")
    interpretation_guide: str = Field(description="Как читать этот график")


class StatisticalTest(BaseModel):
    """Статистический тест"""
    model_config = ConfigDict(extra="forbid")
    
    method: StatMethod = Field(description="Метод анализа")
    hypothesis_id: str = Field(description="Проверяемая гипотеза")
    variables: list[str] = Field(description="Переменные для теста")
    expected_output: str = Field(description="Что ожидаем получить")
    assumptions: list[str] = Field(
        default_factory=list,
        description="Допущения метода (нормальность, гомоскедастичность...)"
    )
    software_suggestion: str = Field(
        default="Python (scipy, statsmodels)",
        description="Инструмент для расчёта"
    )


class DataQualityCheck(BaseModel):
    """Проверка качества данных"""
    model_config = ConfigDict(extra="forbid")
    
    check: str = Field(description="Что проверяем")
    method: str = Field(description="Как проверяем")
    threshold: str = Field(description="Пороговое значение")
    action_if_failed: str = Field(description="Что делать при провале проверки")


class ResearchDesign(BaseModel):
    """Полный дизайн исследования"""
    model_config = ConfigDict(extra="forbid")
    
    # Гипотезы
    hypotheses: list[Hypothesis] = Field(
        description="Исследовательские гипотезы (3-5 шт.)"
    )
    
    # Измерения
    required_measurements: list[Measurement] = Field(
        description="Необходимые измерения и показатели"
    )
    
    # Группировки
    grouping_rules: list[GroupingRule] = Field(
        description="Правила группировки данных"
    )
    
    # Производные метрики
    derived_metrics: list[DerivedMetric] = Field(
        default_factory=list,
        description="Производные метрики с формулами"
    )
    
    # Статистические методы
    statistical_methods: list[StatisticalTest] = Field(
        description="Статистические методы и тесты"
    )
    
    # Визуализации
    visualizations: list[VisualizationSpec] = Field(
        description="Спецификации визуализаций (5-10 шт.)"
    )
    
    # Контроль качества
    data_quality_checks: list[DataQualityCheck] = Field(
        default_factory=list,
        description="Проверки качества данных"
    )
    
    # Методологические заметки
    methodology_notes: str = Field(
        description="Общие методологические заметки и ограничения"
    )
    
    # Связь с исходным запросом
    task_type_match: str = Field(
        description="Как дизайн соответствует task_type исходного запроса"
    )
    required_row_grain: list[str] = Field(
        description="Необходимая детализация данных (country, year, region...)"
    )
    minimal_observations: int = Field(
        description="Минимально необходимое число наблюдений"
    )
