from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Architecture(str, Enum):
    ENCODER = "encoder_only"
    DECODER = "decoder_only"
    ENCODER_DECODER = "encoder_decoder"


class WorkloadType(str, Enum):
    ENCODER_INFERENCE = "encoder_inference"
    AUTOREGRESSIVE_GENERATION = "autoregressive_generation"
    ENCODER_DECODER_GENERATION = "encoder_decoder_generation"
    MULTI_TURN_CHAT = "multi_turn_chat"


DEFAULT_MODELS: dict[Architecture, str] = {
    Architecture.ENCODER: "distilbert/distilbert-base-uncased",
    Architecture.DECODER: "Qwen/Qwen2.5-0.5B-Instruct",
    Architecture.ENCODER_DECODER: "google/flan-t5-small",
}


class WorkloadRequest(BaseModel):
    architecture: Architecture
    model_id: str | None = None
    workload_type: WorkloadType
    input_tokens: int = Field(ge=1, le=32768)
    requested_output_tokens: int = Field(default=0, ge=0, le=8192)
    batch_size: int = Field(default=1, ge=1, le=128)
    concurrent_requests: int = Field(default=1, ge=1, le=64)
    history_turns: int = Field(default=0, ge=0, le=100)
    cache_enabled: bool = False
    precision: Literal["fp32", "fp16", "bf16", "int8"] = "fp32"

    @model_validator(mode="after")
    def validate_compatibility(self) -> "WorkloadRequest":
        generation = {
            WorkloadType.AUTOREGRESSIVE_GENERATION,
            WorkloadType.ENCODER_DECODER_GENERATION,
            WorkloadType.MULTI_TURN_CHAT,
        }
        if self.architecture == Architecture.ENCODER and self.workload_type != WorkloadType.ENCODER_INFERENCE:
            raise ValueError("encoder_only supports encoder_inference only")
        if self.architecture == Architecture.DECODER and self.workload_type not in {
            WorkloadType.AUTOREGRESSIVE_GENERATION, WorkloadType.MULTI_TURN_CHAT
        }:
            raise ValueError("decoder_only supports autoregressive_generation or multi_turn_chat")
        if self.architecture == Architecture.ENCODER_DECODER and self.workload_type != WorkloadType.ENCODER_DECODER_GENERATION:
            raise ValueError("encoder_decoder supports encoder_decoder_generation only")
        if self.workload_type not in generation and self.requested_output_tokens:
            raise ValueError("encoder inference cannot request output tokens")
        if self.cache_enabled and self.architecture == Architecture.ENCODER:
            raise ValueError("KV cache is not applicable to encoder_only workloads")
        return self

    @property
    def resolved_model_id(self) -> str:
        return self.model_id or DEFAULT_MODELS[self.architecture]


class BenchmarkRecord(BaseModel):
    id: str
    created_at: str
    source: Literal["synthetic", "measured"]
    benchmark_version: str
    workload: WorkloadRequest
    end_to_end_latency_ms: float | None = None
    model_load_time_ms: float | None = None
    tokenization_time_ms: float | None = None
    prefill_latency_ms: float | None = None
    time_to_first_token_ms: float | None = None
    time_per_generated_token_ms: float | None = None
    total_generation_latency_ms: float | None = None
    tokens_per_second: float | None = None
    requests_per_second: float | None = None
    peak_memory_mb: float | None = None
    input_token_count: int | None = None
    output_token_count: int | None = None
    variability_ms: float | None = None
    runtime_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    notes: str | None = None


class BenchmarkRequest(WorkloadRequest):
    warmup_runs: int = Field(default=1, ge=0, le=5)
    repetitions: int = Field(default=3, ge=1, le=20)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    allow_model_download: bool = False


class PredictionInterval(BaseModel):
    lower: float
    upper: float


class TwinPrediction(BaseModel):
    prediction_source: Literal["analytical", "calibrated"]
    workload: WorkloadRequest
    predicted_end_to_end_latency_ms: float
    predicted_peak_memory_mb: float
    predicted_tokens_per_second: float | None = None
    latency_interval_ms: PredictionInterval
    assumptions: list[str]
    calibration_sample_count: int = 0


class ErrorMetrics(BaseModel):
    mae: float
    rmse: float
    mape_percent: float | None = None
    r_squared: float | None = None
    sample_count: int


class CalibrationReport(BaseModel):
    status: Literal["untrained", "trained", "insufficient_data"]
    measured_record_count: int
    training_record_count: int = 0
    held_out_record_count: int = 0
    split_description: str
    baseline_latency: ErrorMetrics | None = None
    calibrated_latency: ErrorMetrics | None = None
    residual_standard_deviation_ms: float | None = None
    limitations: list[str] = Field(default_factory=list)


class Overview(BaseModel):
    record_count: int
    synthetic_record_count: int
    measured_record_count: int
    calibration_status: Literal["untrained", "trained"]
    device_summary: str
    supported_defaults: dict[str, str]


class ScalingAxis(str, Enum):
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "requested_output_tokens"
    BATCH_SIZE = "batch_size"


class ScalingStudyRequest(BaseModel):
    base: BenchmarkRequest
    axis: ScalingAxis
    values: list[int] = Field(min_length=2, max_length=12)

    @model_validator(mode="after")
    def validate_values(self) -> "ScalingStudyRequest":
        if any(v < 1 for v in self.values) or len(set(self.values)) != len(self.values):
            raise ValueError("values must be unique positive integers")
        return self


class ScalingPoint(BaseModel):
    value: int
    record_id: str
    end_to_end_latency_ms: float | None
    tokens_per_second: float | None
    peak_memory_mb: float | None


class ScalingStudy(BaseModel):
    id: str
    created_at: str
    kind: Literal["scaling"] = "scaling"
    axis: ScalingAxis
    base: BenchmarkRequest
    points: list[ScalingPoint]
    latency_scaling_exponent: float | None = None
    notes: list[str] = Field(default_factory=list)


class CacheComparisonRequest(BaseModel):
    base: BenchmarkRequest

    @model_validator(mode="after")
    def validate_generation(self) -> "CacheComparisonRequest":
        if self.base.architecture == Architecture.ENCODER or self.base.requested_output_tokens < 2:
            raise ValueError("cache comparison needs a generative architecture with at least 2 output tokens")
        return self


class CacheComparison(BaseModel):
    id: str
    created_at: str
    kind: Literal["cache_comparison"] = "cache_comparison"
    base: BenchmarkRequest
    cache_on_record_id: str
    cache_off_record_id: str
    cache_on_latency_ms: float | None
    cache_off_latency_ms: float | None
    speedup: float | None
    notes: list[str] = Field(default_factory=list)


class MultiTurnRequest(BaseModel):
    model_id: str | None = None
    turns: int = Field(default=4, ge=2, le=20)
    new_tokens_per_turn: int = Field(default=32, ge=1, le=1024)
    output_tokens_per_turn: int = Field(default=16, ge=1, le=256)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    allow_model_download: bool = False


class MultiTurnTurn(BaseModel):
    turn: int
    context_tokens_before: int
    prefill_tokens: int
    prefill_ms: float
    turn_ms: float


class MultiTurnMode(BaseModel):
    mode: Literal["recompute_history", "reuse_kv_cache"]
    turns: list[MultiTurnTurn]
    total_ms: float


class MultiTurnStudy(BaseModel):
    id: str
    created_at: str
    kind: Literal["multi_turn"] = "multi_turn"
    request: MultiTurnRequest
    model_id: str
    modes: list[MultiTurnMode]
    total_speedup: float | None = None
    runtime_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class RagDocumentIn(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9._-]{1,100}\.(md|txt)$")
    text: str = Field(min_length=1, max_length=2_000_000)


class RagIndexRequest(BaseModel):
    chunk_words: int = Field(default=120, ge=20, le=1000)
    overlap_words: int = Field(default=30, ge=0, le=500)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    allow_model_download: bool = False

    @model_validator(mode="after")
    def validate_overlap(self) -> "RagIndexRequest":
        if self.overlap_words >= self.chunk_words:
            raise ValueError("overlap_words must be smaller than chunk_words")
        return self


class RagSource(BaseModel):
    label: str
    path: str
    file_count: int


class RagIndexStatus(BaseModel):
    indexed: bool
    state: Literal["idle", "building", "ready", "error"] = "idle"
    stage: str | None = None
    done: int = 0
    total: int = 0
    error: str | None = None
    skipped: list[str] = Field(default_factory=list)
    documents_changed: bool = False
    sources: list[RagSource] = Field(default_factory=list)
    document_count: int = 0
    chunk_count: int = 0
    embedding_model: str | None = None
    index_backend: str | None = None
    chunk_words: int | None = None
    overlap_words: int | None = None
    built_at: str | None = None


class RagQuery(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)
    generate: bool = False
    model_id: str | None = None
    allow_model_download: bool = False


class Citation(BaseModel):
    rank: int
    document: str
    chunk_id: str
    char_start: int
    char_end: int
    location: str = ""
    score: float
    text: str


class RagAnswer(BaseModel):
    question: str
    citations: list[Citation]
    answer: str | None = None
    answer_model: str | None = None
    answer_seconds: float | None = None
    notes: list[str]


class RagEvalCase(BaseModel):
    question: str
    document: str
    must_contain: str = Field(min_length=1)


class RagEvalRequest(BaseModel):
    cases: list[RagEvalCase] = Field(min_length=1, max_length=500)
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])


class RagEvalCaseResult(BaseModel):
    question: str
    first_relevant_rank: int | None


class RagEvalReport(BaseModel):
    case_count: int
    retrieval_latency_ms_median: float | None = None
    evaluated_at: str | None = None
    chunk_words: int | None = None
    overlap_words: int | None = None
    hit_rate_at_k: dict[str, float]
    mrr: float
    per_case: list[RagEvalCaseResult]
    embedding_model: str
    limitations: list[str]
