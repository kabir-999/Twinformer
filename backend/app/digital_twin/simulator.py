from __future__ import annotations

from dataclasses import dataclass

from app.schemas.workloads import Architecture, PredictionInterval, TwinPrediction, WorkloadRequest


@dataclass(frozen=True)
class ModelProfile:
    parameter_count_m: float
    layers: int
    hidden_size: int


DEFAULT_PROFILES = {
    Architecture.ENCODER: ModelProfile(66, 6, 768),
    Architecture.DECODER: ModelProfile(494, 24, 896),
    Architecture.ENCODER_DECODER: ModelProfile(77, 12, 512),
}


# Parameter counts (millions) of models this project runs; unknown models fall back to a size in the name, then to the architecture default.
KNOWN_PARAMS_M = {
    "sentence-transformers/all-MiniLM-L6-v2": 22.7, "distilbert/distilbert-base-uncased": 66, "google/flan-t5-small": 77, "google/flan-t5-base": 248,
    "google/flan-t5-large": 783, "Qwen/Qwen1.5-0.5B": 464, "Qwen/Qwen2.5-0.5B-Instruct": 494, "Qwen/Qwen2.5-1.5B-Instruct": 1544,
    "facebook/nllb-200-distilled-600M": 615,
}
# Assumed effective laptop-CPU speeds. These are assumptions, not measurements: calibration against real runs replaces them.
ASSUMED_FLOPS = 8e11        # sustained matrix-multiply throughput, FLOP/s
ASSUMED_BANDWIDTH = 5e10    # memory bandwidth for streaming weights, bytes/s
FIXED_OVERHEAD_S = 0.03


def model_params_m(workload: WorkloadRequest) -> float:
    model_id = workload.resolved_model_id
    if model_id in KNOWN_PARAMS_M:
        return KNOWN_PARAMS_M[model_id]
    import re
    named = re.search(r"(\d+(?:\.\d+)?)\s*([bm])\b", model_id, re.I)
    if named:
        return float(named.group(1)) * (1000 if named.group(2).lower() == "b" else 1)
    return DEFAULT_PROFILES[workload.architecture].parameter_count_m


def analytical_terms(workload: WorkloadRequest) -> tuple[float, float]:
    """(prefill seconds, decode seconds) under the assumed machine speeds. Calibration learns correction factors for these two terms."""
    params_m = model_params_m(workload)
    sequence, output, batch = workload.input_tokens, workload.requested_output_tokens, workload.batch_size
    precision_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}[workload.precision]
    share = 0.5 if workload.architecture == Architecture.ENCODER_DECODER else 1.0   # encoder-decoder weights split roughly in half
    reading_params = params_m * share * 1e6
    prefill_s = 2 * reading_params * sequence * batch / ASSUMED_FLOPS
    if workload.architecture == Architecture.ENCODER or output == 0:
        return prefill_s, 0.0
    if workload.cache_enabled:
        return prefill_s, output * reading_params * precision_bytes / ASSUMED_BANDWIDTH
    return prefill_s, sum(2 * reading_params * (sequence + k) * batch / ASSUMED_FLOPS for k in range(output))   # no cache: re-read the growing sequence


def analytical_prediction(workload: WorkloadRequest) -> TwinPrediction:
    """First-principles baseline: prefill is compute-bound, each generated token streams the weights from memory.
    It uses assumed machine speeds, so it is a starting guess that calibration against real runs replaces."""
    profile = DEFAULT_PROFILES[workload.architecture]
    params_m = model_params_m(workload)
    sequence, output, batch = workload.input_tokens, workload.requested_output_tokens, workload.batch_size
    precision_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}[workload.precision]
    prefill_s, decode_s = analytical_terms(workload)
    latency = (FIXED_OVERHEAD_S + prefill_s + decode_s) * 1000
    latency *= 1 + 0.08 * (workload.concurrent_requests - 1)
    weight_memory = params_m * precision_bytes
    kv_memory = 0.0
    if workload.architecture != Architecture.ENCODER:
        kv_memory = 2 * profile.layers * profile.hidden_size * (sequence + output) * batch * precision_bytes / (1024 * 1024)
    activation_memory = sequence * batch * profile.hidden_size * profile.layers * precision_bytes / (1024 * 1024)
    memory = weight_memory + kv_memory + activation_memory + 150
    uncertainty = max(5.0, latency * 0.50)
    return TwinPrediction(
        prediction_source="analytical", workload=workload, predicted_end_to_end_latency_ms=round(latency, 3),
        predicted_peak_memory_mb=round(memory, 3), predicted_tokens_per_second=round(output / (latency / 1000), 3) if output else None,
        latency_interval_ms=PredictionInterval(lower=round(max(0, latency - uncertainty), 3), upper=round(latency + uncertainty, 3)),
        assumptions=[
            "Portable analytical estimate from first principles; it is not a hardware benchmark.",
            f"Uses {params_m:g}M parameters for {workload.resolved_model_id}. Prefill is compute-bound at an assumed {ASSUMED_FLOPS / 1e9:g} GFLOP/s; "
            f"each generated token streams the weights at an assumed {ASSUMED_BANDWIDTH / 1e9:g} GB/s. Real machines differ, which is what calibration corrects.",
            "Ignores beam search, tokenisation, Python overhead and thermal throttling; latency uncertainty is a broad ±50% pre-calibration interval.",
            "Memory combines approximate weights, activations and decoder KV cache where applicable; it is not process RSS or GPU allocation.",
        ],
    )
