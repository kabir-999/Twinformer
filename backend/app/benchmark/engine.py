from __future__ import annotations

import gc
import os
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import psutil

from app.schemas.workloads import Architecture, BenchmarkRecord, BenchmarkRequest


class BenchmarkUnavailable(RuntimeError):
    """Raised when requested local model/runtime resources are unavailable."""


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: str
    load_time_ms: float
    runtime: dict[str, str | int | float | bool | None]


class LocalModelRegistry:
    """In-process model cache. Models are only downloaded on explicit opt-in."""
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str], LoadedModel] = {}

    def load(self, request: BenchmarkRequest) -> LoadedModel:
        try:
            import torch
            import transformers
            from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise BenchmarkUnavailable("Local inference dependencies are absent. Install backend/requirements.txt.") from exc
        requested_device = request.device
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise BenchmarkUnavailable("CUDA was requested but is not available.")
        device = "cuda" if requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available()) else "cpu"
        cache_key = (request.resolved_model_id, device, request.precision)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return LoadedModel(cached.model, cached.tokenizer, cached.device, 0.0, {**cached.runtime, "model_cached": True})
        dtype = torch.float32
        precision_note = request.precision
        if device == "cuda" and request.precision == "fp16": dtype = torch.float16
        elif device == "cuda" and request.precision == "bf16": dtype = torch.bfloat16
        elif request.precision != "fp32": precision_note = f"{request.precision}_requested_but_fp32_used"
        if request.precision == "int8":
            raise BenchmarkUnavailable("int8 loading is not enabled in Phase 2; select fp32/fp16/bf16.")
        started = time.perf_counter()
        try:
            tokenizer = AutoTokenizer.from_pretrained(request.resolved_model_id, local_files_only=not request.allow_model_download)
            if request.architecture == Architecture.ENCODER:
                model = AutoModel.from_pretrained(request.resolved_model_id, torch_dtype=dtype, local_files_only=not request.allow_model_download)
            elif request.architecture == Architecture.DECODER:
                model = AutoModelForCausalLM.from_pretrained(request.resolved_model_id, torch_dtype=dtype, local_files_only=not request.allow_model_download)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(request.resolved_model_id, torch_dtype=dtype, local_files_only=not request.allow_model_download)
        except Exception as exc:
            mode = "Download is disabled; cache the model locally or set allow_model_download=true." if not request.allow_model_download else "Check model availability, network, license acceptance, and available memory."
            raise BenchmarkUnavailable(f"Could not load {request.resolved_model_id}. {mode} Original error: {type(exc).__name__}: {exc}") from exc
        model.to(device).eval()
        loaded = LoadedModel(model, tokenizer, device, (time.perf_counter() - started) * 1000, {
            "device": device, "precision_effective": precision_note, "torch_version": torch.__version__,
            "transformers_version": transformers.__version__, "python": platform.python_version(),
            "model_id": request.resolved_model_id, "model_cached": False,
        })
        self._cache[cache_key] = loaded
        return loaded


class BenchmarkEngine:
    def __init__(self, registry: LocalModelRegistry | None = None) -> None:
        self.registry = registry or LocalModelRegistry()

    @staticmethod
    def _sync(device: str) -> None:
        if device == "cuda":
            import torch
            torch.cuda.synchronize()

    @staticmethod
    def _prompt(tokens: int) -> str:
        # Tokenizers differ; actual token counts, not this requested length, are stored below.
        return " ".join(["transformer"] * max(1, tokens))

    def _run_once(self, loaded: LoadedModel, request: BenchmarkRequest) -> dict[str, float | int | None]:
        import torch
        process = psutil.Process()
        if loaded.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        memory_before = process.memory_info().rss / (1024 * 1024)
        started = time.perf_counter()
        token_start = time.perf_counter()
        encoded = loaded.tokenizer([self._prompt(request.input_tokens)] * request.batch_size, return_tensors="pt", truncation=True, padding=True)
        encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
        self._sync(loaded.device)
        tokenization_ms = (time.perf_counter() - token_start) * 1000
        input_count = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            prefill_start = time.perf_counter()
            if request.architecture == Architecture.ENCODER:
                _ = loaded.model(**encoded)
                self._sync(loaded.device)
                prefill_ms = (time.perf_counter() - prefill_start) * 1000
                output_count, ttft_ms, generation_ms, per_token_ms = 0, None, None, None
            elif request.architecture == Architecture.DECODER:
                outputs = loaded.model(**encoded, use_cache=request.cache_enabled)
                self._sync(loaded.device)
                prefill_ms = (time.perf_counter() - prefill_start) * 1000
                ttft_ms = prefill_ms
                generated, past, next_token = 0, outputs.past_key_values if request.cache_enabled else None, outputs.logits[:, -1:].argmax(dim=-1)
                sequence = encoded["input_ids"]
                decode_start = time.perf_counter()
                while generated < request.requested_output_tokens:
                    generated += 1
                    if generated == request.requested_output_tokens: break
                    if request.cache_enabled:
                        step = loaded.model(input_ids=next_token, past_key_values=past, use_cache=True)
                    else:  # no cache: recompute attention over the whole sequence every step
                        sequence = torch.cat([sequence, next_token], dim=-1)
                        step = loaded.model(input_ids=sequence, use_cache=False)
                    past, next_token = (step.past_key_values if request.cache_enabled else None), step.logits[:, -1:].argmax(dim=-1)
                self._sync(loaded.device)
                generation_ms = (time.perf_counter() - decode_start) * 1000 + prefill_ms
                per_token_ms = generation_ms / generated if generated else None
                output_count = generated
            else:
                from transformers.modeling_outputs import BaseModelOutput
                start_id = loaded.model.config.decoder_start_token_id or loaded.tokenizer.pad_token_id
                decoder_ids = torch.full((encoded["input_ids"].shape[0], 1), start_id, dtype=torch.long, device=loaded.device)
                outputs = loaded.model(**encoded, decoder_input_ids=decoder_ids, use_cache=request.cache_enabled)
                self._sync(loaded.device)
                prefill_ms = (time.perf_counter() - prefill_start) * 1000
                ttft_ms = prefill_ms
                generated, past, next_token = 0, outputs.past_key_values if request.cache_enabled else None, outputs.logits[:, -1:].argmax(dim=-1)
                decoder_sequence = decoder_ids
                encoder_out = BaseModelOutput(last_hidden_state=outputs.encoder_last_hidden_state)
                decode_start = time.perf_counter()
                while generated < request.requested_output_tokens:
                    generated += 1
                    if generated == request.requested_output_tokens: break
                    if request.cache_enabled:
                        step = loaded.model(encoder_outputs=encoder_out, attention_mask=encoded.get("attention_mask"), decoder_input_ids=next_token, past_key_values=past, use_cache=True)
                    else:  # no cache: re-run the decoder over all generated tokens (encoder output is reused)
                        decoder_sequence = torch.cat([decoder_sequence, next_token], dim=-1)
                        step = loaded.model(encoder_outputs=encoder_out, attention_mask=encoded.get("attention_mask"), decoder_input_ids=decoder_sequence, use_cache=False)
                    past, next_token = (step.past_key_values if request.cache_enabled else None), step.logits[:, -1:].argmax(dim=-1)
                self._sync(loaded.device)
                generation_ms = (time.perf_counter() - decode_start) * 1000 + prefill_ms
                per_token_ms = generation_ms / generated if generated else None
                output_count = generated
        end_to_end_ms = (time.perf_counter() - started) * 1000
        rss_mb = process.memory_info().rss / (1024 * 1024)
        peak_mb = max(memory_before, rss_mb)
        if loaded.device == "cuda": peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() / (1024 * 1024))
        return {"tokenization_ms": tokenization_ms, "prefill_ms": prefill_ms, "ttft_ms": ttft_ms, "generation_ms": generation_ms, "per_token_ms": per_token_ms, "end_to_end_ms": end_to_end_ms, "peak_memory_mb": peak_mb, "input_count": input_count, "output_count": output_count}

    def run(self, request: BenchmarkRequest) -> BenchmarkRecord:
        if request.concurrent_requests != 1:
            raise BenchmarkUnavailable("Phase 2 measures one request at a time. Concurrent scheduling is not implemented.")
        loaded = self.registry.load(request)
        for _ in range(request.warmup_runs): self._run_once(loaded, request)
        samples = [self._run_once(loaded, request) for _ in range(request.repetitions)]
        median = lambda key: statistics.median(s[key] for s in samples if s[key] is not None)
        e2e = float(median("end_to_end_ms"))
        output_count = int(median("output_count"))
        note = "Measured local inference. CPU RSS is process-level current RSS (not a per-request peak)."
        if loaded.device == "cuda": note += " GPU memory is torch max allocated during the request and excludes all driver/runtime allocations."
        return BenchmarkRecord(
            id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(), source="measured", benchmark_version="phase4-local-v2", workload=request,
            model_load_time_ms=loaded.load_time_ms, tokenization_time_ms=float(median("tokenization_ms")), prefill_latency_ms=float(median("prefill_ms")), time_to_first_token_ms=median("ttft_ms"), time_per_generated_token_ms=median("per_token_ms"), total_generation_latency_ms=median("generation_ms"), end_to_end_latency_ms=e2e,
            tokens_per_second=round((output_count * request.batch_size) / (e2e / 1000), 3) if output_count else None,
            requests_per_second=round(1000 / e2e, 3), peak_memory_mb=float(median("peak_memory_mb")), input_token_count=int(median("input_count")), output_token_count=output_count,
            variability_ms=round(statistics.pstdev(s["end_to_end_ms"] for s in samples), 3), runtime_metadata=loaded.runtime, notes=note,
        )


registry = LocalModelRegistry()
engine = BenchmarkEngine(registry)
