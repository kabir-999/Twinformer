"""Synthetic records are a UI/demo aid; they are never physical measurements."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid5, NAMESPACE_URL

from app.schemas.workloads import Architecture, BenchmarkRecord, WorkloadRequest, WorkloadType


def synthetic_record(workload: WorkloadRequest, index: int) -> BenchmarkRecord:
    """Deterministic, clearly labelled illustrative estimates for an empty install."""
    architecture_factor = {
        Architecture.ENCODER: 0.55,
        Architecture.DECODER: 1.0,
        Architecture.ENCODER_DECODER: 1.25,
    }[workload.architecture]
    attention_cost = workload.input_tokens * workload.input_tokens / 50000
    output_cost = workload.requested_output_tokens * (1.8 if workload.cache_enabled else 3.0)
    latency = (8 + attention_cost + output_cost) * architecture_factor * workload.batch_size
    latency *= 1 + 0.08 * (workload.concurrent_requests - 1)
    memory = 120 + workload.input_tokens * workload.batch_size * 0.08
    memory += workload.requested_output_tokens * workload.batch_size * (0.07 if workload.cache_enabled else 0.02)
    total_tokens = workload.input_tokens + workload.requested_output_tokens
    record_key = f"phase1/{index}/{workload.model_dump_json()}"
    return BenchmarkRecord(
        id=str(uuid5(NAMESPACE_URL, record_key)),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        source="synthetic",
        benchmark_version="synthetic-demo-v1",
        workload=workload,
        end_to_end_latency_ms=round(latency, 2),
        tokens_per_second=round((total_tokens * workload.batch_size) / (latency / 1000), 2),
        peak_memory_mb=round(memory, 2),
        notes="Synthetic demonstration estimate; no model was loaded or benchmarked.",
    )


def demo_records() -> list[BenchmarkRecord]:
    configs = [
        (Architecture.ENCODER, WorkloadType.ENCODER_INFERENCE, 64, 0, False),
        (Architecture.ENCODER, WorkloadType.ENCODER_INFERENCE, 256, 0, False),
        (Architecture.DECODER, WorkloadType.AUTOREGRESSIVE_GENERATION, 64, 32, True),
        (Architecture.DECODER, WorkloadType.AUTOREGRESSIVE_GENERATION, 256, 96, True),
        (Architecture.ENCODER_DECODER, WorkloadType.ENCODER_DECODER_GENERATION, 64, 32, False),
        (Architecture.ENCODER_DECODER, WorkloadType.ENCODER_DECODER_GENERATION, 256, 96, False),
    ]
    return [synthetic_record(WorkloadRequest(architecture=a, workload_type=t, input_tokens=i, requested_output_tokens=o, cache_enabled=c), n)
            for n, (a, t, i, o, c) in enumerate(configs)]
