"""Strictly validated data, metric, report, and benchmark models."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
NonEmptyText = Annotated[str, StringConstraints(min_length=1)]
Score = Annotated[float, Field(ge=0, le=1, strict=True, allow_inf_nan=False)]
Status = Literal["ok", "skipped", "error"]
MetricKind = Literal["deterministic", "llm"]


class Model(BaseModel):
    """Reject unknown fields and non-finite numbers; avoid in-place field assignment."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class RetrievedContext(Model):
    id: Identifier
    text: NonEmptyText
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("context text must not be blank")
        return value


class EvaluationSample(Model):
    id: Identifier
    question: NonEmptyText
    answer: str
    contexts: tuple[RetrievedContext, ...]
    reference_answer: str | None = None
    reference_context_ids: tuple[Identifier, ...] | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @model_validator(mode="after")
    def unique_contexts(self) -> Self:
        ids = [context.id for context in self.contexts]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieved context IDs must be unique within a sample")
        refs = self.reference_context_ids
        if refs is not None and len(refs) != len(set(refs)):
            raise ValueError("reference context IDs must be unique within a sample")
        return self


class EvaluationDataset(Model):
    name: Identifier
    samples: Annotated[tuple[EvaluationSample, ...], Field(min_length=1)]
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        ids = [sample.id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("sample IDs must be unique within a dataset")
        return self


class Evidence(Model):
    context_id: Identifier
    quote: NonEmptyText
    claim: str = ""


class MetricResult(Model):
    metric: Identifier
    status: Status = "ok"
    score: Score | None = None
    explanation: Annotated[str, Field(max_length=2000)] = ""
    evidence: tuple[Evidence, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent_status(self) -> Self:
        if self.status == "ok" and self.score is None:
            raise ValueError("an ok result must have a score")
        if self.status != "ok" and self.score is not None:
            raise ValueError("skipped/error results must not have a score")
        if self.status != "ok" and not self.explanation.strip():
            raise ValueError("skipped/error results must explain why")
        return self


class MetricDefinition(Model):
    name: Identifier
    kind: MetricKind
    higher_is_better: bool = True
    version: Identifier = "1"
    description: str


class EvaluationConfig(Model):
    metrics: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    k: Annotated[int, Field(strict=True, ge=1)] | None = None
    enable_llm: bool = False
    fail_fast: bool = False

    @field_validator("metrics")
    @classmethod
    def unique_metrics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("metrics must not contain duplicates")
        return value


class AggregateStatistics(Model):
    count: Annotated[int, Field(ge=0)]
    skipped: Annotated[int, Field(ge=0)] = 0
    errors: Annotated[int, Field(ge=0)] = 0
    mean: Score | None = None
    median: Score | None = None
    stddev: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    minimum: Score | None = None
    maximum: Score | None = None

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        values = (self.mean, self.median, self.stddev, self.minimum, self.maximum)
        if self.count == 0 and any(value is not None for value in values):
            raise ValueError("empty aggregates must have null statistics")
        if self.count > 0 and any(value is None for value in values):
            raise ValueError("nonempty aggregates require all statistics")
        return self


class EvaluationResult(Model):
    sample_id: Identifier
    comparison_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    metrics: tuple[MetricResult, ...]

    @model_validator(mode="after")
    def unique_metrics(self) -> Self:
        names = [metric.metric for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("result contains duplicate metrics")
        return self


class DatasetInfo(Model):
    name: Identifier
    sample_count: Annotated[int, Field(ge=1)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class EvaluationReport(Model):
    schema_version: Literal["1.0"] = "1.0"
    toolkit_version: str = "0.1.0"
    created_at: AwareDatetime
    dataset: DatasetInfo
    configuration: EvaluationConfig
    metric_definitions: tuple[MetricDefinition, ...]
    judge: dict[str, JsonValue] | None = None
    results: tuple[EvaluationResult, ...]
    aggregates: dict[str, AggregateStatistics]
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def consistent_results(self) -> Self:
        ids = [result.sample_id for result in self.results]
        if len(ids) != self.dataset.sample_count or len(ids) != len(set(ids)):
            raise ValueError("report sample count/IDs are inconsistent")
        names = set(self.configuration.metrics)
        definitions = [definition.name for definition in self.metric_definitions]
        if set(definitions) != names or len(definitions) != len(names):
            raise ValueError("metric definitions do not match configuration")
        if set(self.aggregates) != names:
            raise ValueError("aggregate names do not match configuration")
        if any({metric.metric for metric in result.metrics} != names for result in self.results):
            raise ValueError("sample metric names do not match configuration")
        for name, stats in self.aggregates.items():
            metrics = [metric for result in self.results for metric in result.metrics if metric.metric == name]
            counts = tuple(sum(metric.status == status for metric in metrics) for status in ("ok", "skipped", "error"))
            if counts != (stats.count, stats.skipped, stats.errors):
                raise ValueError("aggregate status counts do not match sample results")
        return self


class MetricComparison(Model):
    metric: str
    paired_count: int
    excluded_count: int
    coverage_regression: bool = False
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    delta: float | None = None
    improvement: float | None = None
    status: Literal["improved", "regressed", "unchanged", "not_applicable"]


class RunComparison(Model):
    baseline_dataset: str
    candidate_dataset: str
    threshold: Score
    metrics: tuple[MetricComparison, ...]
    has_regression: bool
    warnings: tuple[str, ...] = ()


class BenchmarkRun(Model):
    dataset: Identifier
    configuration: Identifier
    path: NonEmptyText


class BenchmarkConfig(Model):
    name: Identifier
    baseline: Identifier
    evaluation: EvaluationConfig
    runs: Annotated[tuple[BenchmarkRun, ...], Field(min_length=2)]
    regression_threshold: Score = 0.0

    @model_validator(mode="after")
    def validate_runs(self) -> Self:
        keys = [(run.dataset, run.configuration) for run in self.runs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate dataset/configuration pair in benchmark")
        all_configs = {run.configuration for run in self.runs}
        for dataset in {run.dataset for run in self.runs}:
            configs = {run.configuration for run in self.runs if run.dataset == dataset}
            if self.baseline not in configs or len(configs) < 2:
                raise ValueError("each dataset needs the baseline and at least one candidate")
            if configs != all_configs:
                raise ValueError("all datasets must contain the same configuration set")
        return self


class BenchmarkRunResult(Model):
    dataset: str
    configuration: str
    report: EvaluationReport


class BenchmarkComparison(Model):
    dataset: str
    configuration: str
    comparison: RunComparison


class BenchmarkResult(Model):
    schema_version: Literal["1.0"] = "1.0"
    name: str
    created_at: AwareDatetime
    configuration: BenchmarkConfig
    runs: tuple[BenchmarkRunResult, ...]
    comparisons: tuple[BenchmarkComparison, ...]
    aggregates: dict[str, dict[str, AggregateStatistics]]
    has_regression: bool
