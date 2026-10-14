from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.calibration.service import CATEGORICAL, FEATURE_COLUMNS, PhysicsModel, InsufficientData, _features, _metric, split_by_input_length
from app.digital_twin.simulator import analytical_prediction
from app.schemas.workloads import BenchmarkRecord, ErrorMetrics, RagEvalReport

CURRENT_BENCHMARK_VERSION = "phase4-local-v2"
BACKEND_ROOT = Path(__file__).resolve().parents[1]


class AblationRow(BaseModel):
    variant: str
    metrics: ErrorMetrics


class AblationReport(BaseModel):
    status: str
    split_description: str
    training_record_count: int = 0
    held_out_record_count: int = 0
    rows: list[AblationRow] = []
    interval_coverage_percent: float | None = None
    notes: list[str] = []


def _fit(train: list[BenchmarkRecord], columns: list[str]):
    y = [r.end_to_end_latency_ms for r in train]
    if not columns:
        return DummyRegressor(strategy="mean").fit(pd.DataFrame([_features(r.workload) for r in train]), y)
    numeric = [c for c in columns if c not in CATEGORICAL]
    categorical = [c for c in columns if c in CATEGORICAL]
    parts = ([("numeric", StandardScaler(), numeric)] if numeric else []) + ([("categories", OneHotEncoder(handle_unknown="ignore"), categorical)] if categorical else [])
    return Pipeline([("features", ColumnTransformer(parts)), ("regressor", Ridge(alpha=1.0))]).fit(pd.DataFrame([_features(r.workload) for r in train]), y)


def run_ablation(records: list[BenchmarkRecord]) -> AblationReport:
    """Leave-one-feature-out ablation on the same sequence-length held-out split used for calibration."""
    measured = [r for r in records if r.source == "measured" and r.end_to_end_latency_ms is not None]
    try:
        train, heldout, _, _ = split_by_input_length(measured)
    except InsufficientData as exc:
        return AblationReport(status="insufficient_data", split_description=str(exc), notes=exc.limitations)
    actual = [float(r.end_to_end_latency_ms) for r in heldout]
    x_held = pd.DataFrame([_features(r.workload) for r in heldout])
    rows = [AblationRow(variant="analytical_baseline", metrics=_metric(actual, [analytical_prediction(r.workload).predicted_end_to_end_latency_ms for r in heldout])),
            AblationRow(variant="intercept_only (train mean)", metrics=_metric(actual, [float(v) for v in _fit(train, []).predict(x_held)]))]
    full = _fit(train, FEATURE_COLUMNS)
    rows.append(AblationRow(variant="ridge_all_features", metrics=_metric(actual, [float(v) for v in full.predict(x_held)])))
    physics = PhysicsModel().fit([r.workload for r in train], [float(r.end_to_end_latency_ms) for r in train])
    rows.append(AblationRow(variant="physics_calibrated (used by the twin)", metrics=_metric(actual, [float(v) for v in physics.predict([r.workload for r in heldout])])))
    for dropped in FEATURE_COLUMNS:
        model = _fit(train, [c for c in FEATURE_COLUMNS if c != dropped])
        rows.append(AblationRow(variant=f"ridge_without_{dropped}", metrics=_metric(actual, [float(v) for v in model.predict(x_held)])))
    train_resid = np.asarray([r.end_to_end_latency_ms for r in train]) - full.predict(pd.DataFrame([_features(r.workload) for r in train]))
    half_width = 1.96 * float(np.std(train_resid, ddof=1))
    coverage = float(np.mean(np.abs(np.asarray(actual) - full.predict(x_held)) <= half_width) * 100)
    versions = {r.benchmark_version for r in measured}
    return AblationReport(status="complete", split_description="Train on shorter input lengths; hold out the longest quarter of distinct lengths (same rule as calibration).", training_record_count=len(train), held_out_record_count=len(heldout), rows=rows, interval_coverage_percent=round(coverage, 1), notes=[
        "Features whose removal raises held-out error are carrying signal on this split; features that do not change error may simply not vary in the collected data.",
        f"Measured records come from benchmark versions: {', '.join(sorted(versions))}." + (" Mixed versions are not strictly comparable." if len(versions) > 1 else ""),
        *([f"Only {len(heldout)} held-out records: differences between variants are not statistically meaningful."] if len(heldout) < 10 else []),
        "Interval coverage is the share of held-out points inside ±1.96 training-residual standard deviations; with few points it is only indicative.",
    ])


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def code_digest() -> str:
    h = hashlib.sha256()
    for path in sorted(BACKEND_ROOT.rglob("*.py")):
        h.update(str(path.relative_to(BACKEND_ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def build_manifest(records: list[BenchmarkRecord], studies: list[dict], rag_eval: RagEvalReport | None, rag_status: dict) -> dict:
    measured = sorted((r for r in records if r.source == "measured"), key=lambda r: r.id)
    import psutil
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": code_digest(),
        "environment": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
                        "memory_gb": round(psutil.virtual_memory().total / 1024 ** 3, 1),
                        "packages": {n: _package_version(n) for n in ("torch", "transformers", "scikit-learn", "numpy", "faiss-cpu", "fastapi", "pydantic")}},
        "measured_records": {"count": len(measured), "ids_sha256": hashlib.sha256("\n".join(r.id for r in measured).encode()).hexdigest(),
                             "payload_sha256": hashlib.sha256("\n".join(r.model_dump_json() for r in measured).encode()).hexdigest(),
                             "benchmark_versions": sorted({r.benchmark_version for r in measured}),
                             "models": sorted({r.workload.resolved_model_id for r in measured}),
                             "devices": sorted({str(r.runtime_metadata.get("device")) for r in measured if r.runtime_metadata}),
                             "precisions": sorted({str(r.runtime_metadata.get("precision_effective", r.workload.precision)) for r in measured})},
        "synthetic_records_excluded": sum(r.source == "synthetic" for r in records),
        "studies": [{"id": s["id"], "kind": s["kind"], "created_at": s["created_at"]} for s in studies],
        "rag": {"index": rag_status, "last_evaluation_at": rag_eval.evaluated_at if rag_eval else None},
    }


def save_manifest(manifest: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"manifest-{manifest['created_at'][:19].replace(':', '')}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(records: list[BenchmarkRecord], manifest: dict, calibration, ablation: AblationReport, studies: list[dict], rag_eval: RagEvalReport | None, manifest_path: str) -> str:
    """Every table row is computed from stored data; sections without completed experiments say so."""
    out = ["# TransformerTwin evaluation report", "", f"> Generated {manifest['created_at']} from saved experiments only. Synthetic records ({manifest['synthetic_records_excluded']}) are excluded from every number below.", ""]
    env, mr = manifest["environment"], manifest["measured_records"]
    out += ["## Environment and reproducibility", "", f"- Code SHA-256 (backend/app/**/*.py): `{manifest['code_sha256'][:16]}…`", f"- Hardware and OS: {env['platform']} · {env['machine']} · {env['cpu_count']} CPUs · {env['memory_gb']} GB RAM",
            f"- Packages: " + ", ".join(f"{k} {v}" for k, v in env["packages"].items() if v), f"- Python: {env['python']}",
            f"- Models: {', '.join(mr['models']) or 'none'} · devices: {', '.join(mr['devices']) or 'none'} · precision: {', '.join(mr['precisions']) or 'none'}",
            f"- Measured records: {mr['count']} (benchmark versions: {', '.join(mr['benchmark_versions']) or 'none'}); payload SHA-256 `{mr['payload_sha256'][:16]}…`",
            f"- Manifest file: `{manifest_path}`", ""]
    out += ["## Methods", "", "Each measured record is the median of its configured repetitions after warmup runs; variability is the population standard deviation of end-to-end latency across repetitions. CPU memory is process RSS, not a per-request peak. Calibration trains on shorter input lengths and evaluates on held-out longer lengths.", ""]
    current = [r for r in records if r.source == "measured" and r.benchmark_version == CURRENT_BENCHMARK_VERSION and r.end_to_end_latency_ms is not None]
    older = sum(r.source == "measured" for r in records) - len(current)
    out += ["## Measurement results", ""]
    if not current:
        out += ["Not run: no measured records from the current benchmark version.", ""]
    else:
        groups: dict[tuple, list[BenchmarkRecord]] = defaultdict(list)
        for r in current:
            groups[(r.workload.architecture.value, r.workload.workload_type.value, r.workload.cache_enabled)].append(r)
        out += ["| Architecture | Workload | KV cache | N | Median latency (ms) | Median variability (ms) | Median tokens/s | Median memory (MB) |", "|---|---|---|---:|---:|---:|---:|---:|"]
        med = lambda xs: statistics.median(xs) if xs else None
        for (arch, wl, cache), rs in sorted(groups.items()):
            out.append(f"| {arch} | {wl} | {'on' if cache else 'off'} | {len(rs)} | {_fmt(med([r.end_to_end_latency_ms for r in rs]))} | {_fmt(med([r.variability_ms for r in rs if r.variability_ms is not None]))} | {_fmt(med([r.tokens_per_second for r in rs if r.tokens_per_second]))} | {_fmt(med([r.peak_memory_mb for r in rs if r.peak_memory_mb]), 1)} |")
        out.append("")
    if older:
        out += [f"{older} measured record(s) from older benchmark versions are excluded from this table because earlier engines mis-implemented cache-off decoding and batching (see docs/phase4-advanced-inference.md).", ""]
    out += ["## Digital-twin validation", ""]
    if calibration.status != "trained":
        out += [f"Not run or insufficient data (status: {calibration.status}). {calibration.split_description}", ""]
    else:
        b, c = calibration.baseline_latency, calibration.calibrated_latency
        out += ["| Target | Baseline MAE (ms) | Calibrated MAE (ms) | Calibrated MAPE % | Held-out regime | Interval coverage % |", "|---|---:|---:|---:|---|---:|",
                f"| end-to-end latency | {_fmt(b.mae)} | {_fmt(c.mae)} | {_fmt(c.mape_percent)} | {calibration.split_description.split('Held out: ')[-1].split(';')[0]} | {_fmt(ablation.interval_coverage_percent, 1)} |", ""]
    out += ["### Calibration ablation", ""]
    if ablation.status != "complete":
        out += [f"Not run: {ablation.split_description}", ""]
    else:
        out += [f"Train n={ablation.training_record_count}, held-out n={ablation.held_out_record_count}.", "", "| Variant | MAE (ms) | RMSE (ms) | MAPE % |", "|---|---:|---:|---:|"]
        out += [f"| {r.variant} | {_fmt(r.metrics.mae)} | {_fmt(r.metrics.rmse)} | {_fmt(r.metrics.mape_percent)} |" for r in ablation.rows]
        out += ["", *[f"- {n}" for n in ablation.notes], ""]
    out += ["## Caching, scaling and multi-turn studies", ""]
    cache = [s for s in studies if s["kind"] == "cache_comparison"]
    scaling = [s for s in studies if s["kind"] == "scaling"]
    multi = [s for s in studies if s["kind"] == "multi_turn"]
    if not (cache or scaling or multi):
        out += ["Not run: no completed studies.", ""]
    for s in cache:
        out.append(f"- Cache comparison {s['created_at'][:19]}: on {_fmt(s['cache_on_latency_ms'], 1)} ms vs off {_fmt(s['cache_off_latency_ms'], 1)} ms → speedup {s['speedup']}× ({s['base']['architecture']}, in={s['base']['input_tokens']}, out={s['base']['requested_output_tokens']}).")
    for s in scaling:
        out.append(f"- Scaling over {s['axis']} {s['created_at'][:19]}: values {[p['value'] for p in s['points']]}, latency (ms) {[round(p['end_to_end_latency_ms'], 1) for p in s['points'] if p['end_to_end_latency_ms'] is not None]}, log-log exponent {s['latency_scaling_exponent']} (sampled range only).")
    for s in multi:
        modes = {m["mode"]: m["total_ms"] for m in s["modes"]}
        out.append(f"- Multi-turn {s['created_at'][:19]} ({s['model_id']}, {s['request']['turns']} turns): recompute {_fmt(modes.get('recompute_history'), 1)} ms vs KV reuse {_fmt(modes.get('reuse_kv_cache'), 1)} ms → {s['total_speedup']}×; single run per mode.")
    out.append("")
    out += ["## RAG retrieval evaluation", ""]
    if rag_eval is None:
        out += ["Not run: no retrieval evaluation has been completed.", ""]
    else:
        out += ["| Evaluation set | Cases | " + " | ".join(f"Recall@{k}" for k in rag_eval.hit_rate_at_k) + " | MRR | Median retrieval latency (ms) | Notes |", "|---|---:|" + "---:|" * len(rag_eval.hit_rate_at_k) + "---:|---:|---|",
                f"| data/rag/eval_cases.json ({rag_eval.embedding_model}, chunk {rag_eval.chunk_words}/{rag_eval.overlap_words} words) | {rag_eval.case_count} | " + " | ".join(str(v) for v in rag_eval.hit_rate_at_k.values()) + f" | {rag_eval.mrr} | {_fmt(rag_eval.retrieval_latency_ms_median)} | single relevant passage per question |", "", *[f"- {l}" for l in rag_eval.limitations], ""]
    out += ["## Limitations and threats to validity", "",
            "- Hardware, OS load and thermal state affect every latency; no cross-machine generalisation is claimed.",
            "- CPU memory is current RSS, not a per-request peak; GPU memory excludes driver allocations.",
            "- Calibration features do not include model identity, so records from different models are pooled; the held-out split is by input length only.",
            "- Workload coverage is limited to the lengths, batch sizes and models actually run; extrapolation outside that range is untested.",
            "- Multi-turn and cache studies use greedy decoding on repeated-word prompts; real prompts may behave differently.",
            "- The RAG evaluation set is small and written by the project author, so it is not independent of corpus knowledge; hit rates carry wide uncertainty.",
            "- Synthetic demonstration records are never used for any figure above.", ""]
    return "\n".join(out)
