"""Connects the chat to the digital twin: chat answers become measurements, and the twin predicts them."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.calibration.service import CalibrationService
from app.rag.qa import ArchAnswer
from app.schemas.workloads import Architecture, BenchmarkRecord, WorkloadRequest, WorkloadType

CHAT_VERSION = "chat-turn-v1"
TYPES = {"decoder_only": (Architecture.DECODER, WorkloadType.AUTOREGRESSIVE_GENERATION),
         "encoder_decoder": (Architecture.ENCODER_DECODER, WorkloadType.ENCODER_DECODER_GENERATION)}


def _workload(arch: str, model_id: str, input_tokens: int, output_tokens: int) -> WorkloadRequest:
    architecture, workload_type = TYPES[arch]
    return WorkloadRequest(architecture=architecture, workload_type=workload_type, model_id=model_id, input_tokens=max(1, input_tokens),
                           requested_output_tokens=max(1, output_tokens), cache_enabled=True)


def usable(answer: ArchAnswer) -> bool:
    """Only text-generating answers with known token counts, from a warm model, are representative measurements."""
    return (answer.architecture in TYPES and bool(answer.answer) and answer.seconds is not None and not answer.cold_start
            and bool(answer.input_tokens) and bool(answer.output_tokens))


def annotate_predictions(answers: list[ArchAnswer], calibration: CalibrationService) -> None:
    """Ask the twin what it expected for each answer, using the real prompt/output lengths, before it learns from them."""
    for a in answers:
        if a.architecture in TYPES and a.input_tokens and a.output_tokens and not a.cold_start:
            a.predicted_seconds = round(calibration.predict(_workload(a.architecture, a.model_id, a.input_tokens, a.output_tokens)).predicted_end_to_end_latency_ms / 1000, 2)


def records_from_answers(answers: list[ArchAnswer]) -> list[BenchmarkRecord]:
    records = []
    for a in answers:
        if not usable(a):
            continue
        workload = _workload(a.architecture, a.model_id, a.input_tokens, a.output_tokens)
        ms = a.seconds * 1000
        records.append(BenchmarkRecord(
            id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(), source="measured", benchmark_version=CHAT_VERSION, workload=workload,
            end_to_end_latency_ms=round(ms, 1), tokens_per_second=round(a.output_tokens / a.seconds, 2) if a.seconds else None,
            input_token_count=a.input_tokens, output_token_count=a.output_tokens,
            runtime_metadata={"origin": "chat turn", "device": "cpu", "model_id": a.model_id},
            notes="Timed inside a chat turn: tokenisation + generation, warm model, batch size 1."))
    return records


def estimate(calibration: CalibrationService, models: dict[str, str], top_k: int) -> list[dict]:
    """Expected time per model for a typical turn, shown before the question is sent."""
    prompt = 100 + top_k * 190          # question + instructions + retrieved passages, roughly
    plan = [("decoder_only", prompt, 60), ("encoder_decoder", min(prompt, 420), 30)]   # flan-t5 truncates its context
    out = []
    for arch, input_tokens, output_tokens in plan:
        p = calibration.predict(_workload(arch, models[arch], input_tokens, output_tokens))
        out.append({"architecture": arch, "model_id": models[arch], "seconds": round(p.predicted_end_to_end_latency_ms / 1000, 1),
                    "low": round(p.latency_interval_ms.lower / 1000, 1), "high": round(p.latency_interval_ms.upper / 1000, 1),
                    "source": p.prediction_source, "samples": p.calibration_sample_count})
    return out


def status(records: list[BenchmarkRecord], calibration: CalibrationService) -> dict:
    chat = sorted((r for r in records if r.benchmark_version == CHAT_VERSION), key=lambda r: r.created_at)
    measured = sum(r.source == "measured" and r.end_to_end_latency_ms is not None for r in records)
    points = [{"created_at": r.created_at, "architecture": r.workload.architecture.value, "model_id": r.workload.resolved_model_id,
               "input_tokens": r.input_token_count, "output_tokens": r.output_token_count, "actual_ms": r.end_to_end_latency_ms,
               "predicted_ms": calibration.predict(r.workload).predicted_end_to_end_latency_ms} for r in chat[-60:]]
    corrections = calibration.state.pipeline.corrections() if calibration.state else []
    return {"corrections": corrections, "measured_total": measured, "chat_records": len(chat), "calibration": calibration.last_report.model_dump(), "points": points,
            "lengths_seen": len({r.workload.input_tokens for r in records if r.source == "measured"})}
