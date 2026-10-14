from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from app.benchmark.engine import BenchmarkUnavailable, LoadedModel, engine
from app.schemas.workloads import (
    Architecture, BenchmarkRecord, BenchmarkRequest, CacheComparison, CacheComparisonRequest, MultiTurnMode,
    MultiTurnRequest, MultiTurnStudy, MultiTurnTurn, ScalingAxis, ScalingPoint, ScalingStudy, ScalingStudyRequest, WorkloadType,
)

Runner = Callable[[BenchmarkRequest], BenchmarkRecord]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_log_exponent(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares slope of log(latency) vs log(value); None when it cannot be estimated."""
    pairs = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y is not None and y > 0]
    if len(pairs) < 3:
        return None
    mx, my = sum(p[0] for p in pairs) / len(pairs), sum(p[1] for p in pairs) / len(pairs)
    var = sum((p[0] - mx) ** 2 for p in pairs)
    return round(sum((p[0] - mx) * (p[1] - my) for p in pairs) / var, 3) if var > 0 else None


def run_scaling_study(study: ScalingStudyRequest, runner: Runner) -> tuple[ScalingStudy, list[BenchmarkRecord]]:
    """Runs every point or fails; a partial sweep is never returned as a study."""
    records: list[BenchmarkRecord] = []
    for value in study.values:
        request = study.base.model_copy(update={study.axis.value: value})
        request = BenchmarkRequest.model_validate(request.model_dump())  # re-run compatibility validation
        records.append(runner(request))
    points = [ScalingPoint(value=v, record_id=r.id, end_to_end_latency_ms=r.end_to_end_latency_ms, tokens_per_second=r.tokens_per_second, peak_memory_mb=r.peak_memory_mb) for v, r in zip(study.values, records)]
    notes = ["Each point is a median of the configured repetitions; see per-record variability_ms for spread."]
    exponent = log_log_exponent([float(v) for v in study.values], [p.end_to_end_latency_ms for p in points])
    if exponent is None:
        notes.append("Scaling exponent needs at least 3 positive-latency points.")
    else:
        notes.append("Exponent is a log-log fit over the sampled range only; it is not an asymptotic complexity claim.")
    if study.axis == ScalingAxis.INPUT_TOKENS:
        notes.append("Actual tokenized lengths can differ from requested values; check input_token_count on each record.")
    return ScalingStudy(id=str(uuid4()), created_at=_now(), axis=study.axis, base=study.base, points=points, latency_scaling_exponent=exponent, notes=notes), records


def run_cache_comparison(study: CacheComparisonRequest, runner: Runner) -> tuple[CacheComparison, list[BenchmarkRecord]]:
    on = runner(study.base.model_copy(update={"cache_enabled": True}))
    off = runner(study.base.model_copy(update={"cache_enabled": False}))
    speedup = round(off.end_to_end_latency_ms / on.end_to_end_latency_ms, 3) if on.end_to_end_latency_ms and off.end_to_end_latency_ms else None
    notes = [
        "cache_enabled=true carries past_key_values and feeds one new token per step.",
        "cache_enabled=false re-feeds the full sequence every step with use_cache=False (encoder output is still reused for encoder-decoder).",
        "Speedup depends on hardware, output length and prompt length; it is a measurement, not a guarantee.",
    ]
    return CacheComparison(id=str(uuid4()), created_at=_now(), base=study.base, cache_on_record_id=on.id, cache_off_record_id=off.id,
                           cache_on_latency_ms=on.end_to_end_latency_ms, cache_off_latency_ms=off.end_to_end_latency_ms, speedup=speedup, notes=notes), [on, off]


def _multi_turn_pass(loaded: LoadedModel, request: MultiTurnRequest, reuse: bool) -> MultiTurnMode:
    import torch
    device, model, tokenizer = loaded.device, loaded.model, loaded.tokenizer
    new = tokenizer(engine._prompt(request.new_tokens_per_turn), return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    context = new.new_empty((1, 0))  # every token seen so far (user + generated), used by the recompute mode
    pending = None                   # last generated token not yet processed by the KV cache (reuse mode)
    past, turns, total_start = None, [], time.perf_counter()
    with torch.inference_mode():
        for turn in range(request.turns):
            turn_start = time.perf_counter()
            context_before = int(context.shape[-1])
            if reuse:
                inp = new if pending is None else torch.cat([pending, new], dim=-1)
                out = model(input_ids=inp, past_key_values=past, use_cache=True)
                past = out.past_key_values
            else:
                inp = torch.cat([context, new], dim=-1)
                out = model(input_ids=inp, use_cache=True)
                past = out.past_key_values
            engine._sync(device)
            prefill_ms = (time.perf_counter() - turn_start) * 1000
            context = torch.cat([context, new], dim=-1)
            token = out.logits[:, -1:].argmax(dim=-1)
            generated = [token]
            for _ in range(request.output_tokens_per_turn - 1):
                step = model(input_ids=token, past_key_values=past, use_cache=True)
                past, token = step.past_key_values, step.logits[:, -1:].argmax(dim=-1)
                generated.append(token)
            engine._sync(device)
            context = torch.cat([context, *generated], dim=-1)
            pending = token  # sampled but not yet fed through the model
            if not reuse:
                past = None  # recompute mode discards the cache between turns
            turns.append(MultiTurnTurn(turn=turn + 1, context_tokens_before=context_before, prefill_tokens=int(inp.shape[-1]), prefill_ms=round(prefill_ms, 3), turn_ms=round((time.perf_counter() - turn_start) * 1000, 3)))
    return MultiTurnMode(mode="reuse_kv_cache" if reuse else "recompute_history", turns=turns, total_ms=round((time.perf_counter() - total_start) * 1000, 3))


def run_multi_turn_study(request: MultiTurnRequest) -> MultiTurnStudy:
    probe = BenchmarkRequest(architecture=Architecture.DECODER, workload_type=WorkloadType.AUTOREGRESSIVE_GENERATION, model_id=request.model_id, input_tokens=1,
                             device=request.device, allow_model_download=request.allow_model_download)
    loaded = engine.registry.load(probe)
    max_positions = getattr(loaded.model.config, "max_position_embeddings", None)
    needed = request.turns * (request.new_tokens_per_turn + request.output_tokens_per_turn)
    if max_positions and needed > max_positions:
        raise BenchmarkUnavailable(f"Conversation needs about {needed} tokens but the model supports {max_positions}.")
    _multi_turn_pass(loaded, request.model_copy(update={"turns": 2, "output_tokens_per_turn": 2}), reuse=True)  # warmup
    modes = [_multi_turn_pass(loaded, request, reuse=False), _multi_turn_pass(loaded, request, reuse=True)]
    speedup = round(modes[0].total_ms / modes[1].total_ms, 3) if modes[1].total_ms else None
    notes = [
        "recompute_history re-encodes the whole conversation each turn; reuse_kv_cache keeps past_key_values across turns and only processes new tokens.",
        "Both modes see identical token content (greedy decoding), single run each after one warmup; no repetition statistics.",
        "KV memory grows with context; peak memory is not measured in this study.",
    ]
    return MultiTurnStudy(id=str(uuid4()), created_at=_now(), request=request, model_id=probe.resolved_model_id, modes=modes, total_speedup=speedup, runtime_metadata=loaded.runtime, notes=notes)
