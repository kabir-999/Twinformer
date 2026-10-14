from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.digital_twin.simulator import analytical_prediction, analytical_terms
from app.schemas.workloads import BenchmarkRecord, CalibrationReport, ErrorMetrics, PredictionInterval, TwinPrediction, WorkloadRequest


CATEGORICAL = ["architecture", "model"]
FEATURE_COLUMNS = ["prefill_s", "decode_s", "input_tokens", "output_tokens", "batch_size", "concurrent_requests", "cache_enabled"] + CATEGORICAL


def _features(workload: WorkloadRequest) -> dict[str, float | int | str]:
    prefill_s, decode_s = analytical_terms(workload)   # physics terms: the model learns how far the assumed machine speeds are off
    return {"prefill_s": prefill_s, "decode_s": decode_s, "input_tokens": workload.input_tokens, "output_tokens": workload.requested_output_tokens, "batch_size": workload.batch_size, "concurrent_requests": workload.concurrent_requests, "cache_enabled": int(workload.cache_enabled), "architecture": workload.architecture.value, "model": workload.resolved_model_id}


def _metric(actual: list[float], predicted: list[float]) -> ErrorMetrics:
    errors = np.asarray(predicted) - np.asarray(actual)
    mae, rmse = float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))
    nonzero = np.abs(np.asarray(actual)) > 1e-6
    mape = float(np.mean(np.abs(errors[nonzero] / np.asarray(actual)[nonzero])) * 100) if np.any(nonzero) else None
    denominator = float(np.sum((np.asarray(actual) - np.mean(actual)) ** 2))
    r2 = float(1 - np.sum(errors ** 2) / denominator) if denominator > 1e-9 else None
    return ErrorMetrics(mae=round(mae, 4), rmse=round(rmse, 4), mape_percent=round(mape, 4) if mape is not None else None, r_squared=round(r2, 4) if r2 is not None else None, sample_count=len(actual))


class InsufficientData(ValueError):
    def __init__(self, message: str, limitations: list[str]) -> None:
        super().__init__(message)
        self.limitations = limitations


def split_by_input_length(measured: list[BenchmarkRecord]):
    """Train on shorter input lengths, hold out the longest quarter (by distinct length)."""
    unique_lengths = sorted({r.workload.input_tokens for r in measured})
    if len(measured) < 8 or len(unique_lengths) < 3:
        raise InsufficientData("Requires at least 8 measured records spanning at least 3 input-token lengths.", ["Synthetic records are excluded."])
    cutoff_index = max(1, math.floor(len(unique_lengths) * 0.75))
    train_lengths, heldout_lengths = set(unique_lengths[:cutoff_index]), set(unique_lengths[cutoff_index:])
    train, heldout = [r for r in measured if r.workload.input_tokens in train_lengths], [r for r in measured if r.workload.input_tokens in heldout_lengths]
    if len(train) < 5 or len(heldout) < 2:
        raise InsufficientData("Unable to form a ≥5-record training and ≥2-record high-sequence-length held-out split.", ["Collect more distinct workload lengths."])
    return train, heldout, train_lengths, heldout_lengths


ARCHS = ["encoder_only", "decoder_only", "encoder_decoder"]


def physics_features(workload: WorkloadRequest) -> dict[str, float]:
    """The formula's prefill and decode seconds, separated per architecture so each family gets its own correction factors."""
    prefill_s, decode_s = analytical_terms(workload)
    arch = workload.architecture.value
    row: dict[str, float] = {}
    for a in ARCHS:
        row[f"prefill_{a}"] = prefill_s if arch == a else 0.0
        row[f"decode_{a}"] = decode_s if arch == a else 0.0
    return row


class PhysicsModel:
    """time = intercept + a*prefill + b*decode per architecture, with a, b >= 0 (non-negative least squares).
    About five parameters, so it does not overfit a few dozen runs, and a and b say how far the assumed machine speeds are off."""
    def __init__(self) -> None:
        self.reg = LinearRegression(positive=True)
        self.archs: set[str] = set()

    def fit(self, workloads: list[WorkloadRequest], latency_ms: list[float]) -> "PhysicsModel":
        self.archs = {w.architecture.value for w in workloads}
        self.reg.fit(pd.DataFrame([physics_features(w) for w in workloads]), latency_ms)
        return self

    def predict(self, workloads: list[WorkloadRequest]) -> np.ndarray:
        raw = self.reg.predict(pd.DataFrame([physics_features(w) for w in workloads]))
        # an architecture never measured has no learned coefficients: fall back to the formula for it
        return np.asarray([max(float(v), 20.0) if w.architecture.value in self.archs else analytical_prediction(w).predicted_end_to_end_latency_ms for v, w in zip(raw, workloads)])

    def corrections(self) -> list[dict[str, float | str]]:
        """How the learned coefficients compare with the formula (1.0 = the assumed machine speed was right)."""
        coef = dict(zip(self.reg.feature_names_in_, self.reg.coef_))
        return [{"architecture": a, "prefill_factor": round(coef[f"prefill_{a}"] / 1000, 2), "decode_factor": round(coef[f"decode_{a}"] / 1000, 2)} for a in ARCHS if a in self.archs]


@dataclass
class CalibrationState:
    pipeline: PhysicsModel
    residual_std: float
    training_count: int
    report: CalibrationReport


class CalibrationService:
    """Fits only measured latency records; synthetic records are structurally excluded."""
    def __init__(self) -> None:
        self.state: CalibrationState | None = None
        self.last_report = CalibrationReport(status="untrained", measured_record_count=0, split_description="No calibration has been requested.")

    @staticmethod
    def _eligible(records: list[BenchmarkRecord]) -> list[BenchmarkRecord]:
        return [r for r in records if r.source == "measured" and r.end_to_end_latency_ms is not None]

    def train(self, records: list[BenchmarkRecord]) -> CalibrationReport:
        measured = self._eligible(records)
        try:
            train, heldout, train_lengths, heldout_lengths = split_by_input_length(measured)
        except InsufficientData as exc:
            self.state = None
            self.last_report = CalibrationReport(status="insufficient_data", measured_record_count=len(measured), split_description=str(exc), limitations=exc.limitations)
            return self.last_report
        y_train = [float(r.end_to_end_latency_ms) for r in train]
        model = PhysicsModel().fit([r.workload for r in train], y_train)
        actual = [float(r.end_to_end_latency_ms) for r in heldout]
        predicted = [float(v) for v in model.predict([r.workload for r in heldout])]
        baseline = [analytical_prediction(r.workload).predicted_end_to_end_latency_ms for r in heldout]
        train_residuals = np.asarray(y_train) - model.predict([r.workload for r in train])
        residual_std = float(np.std(train_residuals, ddof=1)) if len(train_residuals) > 1 else 0.0
        pipeline = model
        report = CalibrationReport(status="trained", measured_record_count=len(measured), training_record_count=len(train), held_out_record_count=len(heldout), split_description=f"Train: input lengths ≤ {max(train_lengths)}. Held out: input lengths ≥ {min(heldout_lengths)}; split by sequence length, not repeated random samples.", baseline_latency=_metric(actual, baseline), calibrated_latency=_metric(actual, predicted), residual_standard_deviation_ms=round(residual_std, 4), limitations=["Validation is limited to the collected hardware, model revisions and workload range.", "Interval uses training residual standard deviation and is not a formal coverage guarantee."])
        self.state, self.last_report = CalibrationState(pipeline, residual_std, len(train), report), report
        return report

    def predict(self, workload: WorkloadRequest) -> TwinPrediction:
        if self.state is None:
            return analytical_prediction(workload)
        analytical = analytical_prediction(workload)
        if workload.architecture.value not in self.state.pipeline.archs:
            return analytical   # nothing measured for this architecture yet
        latency = float(self.state.pipeline.predict([workload])[0])
        interval = 1.96 * self.state.residual_std
        return TwinPrediction(prediction_source="calibrated", workload=workload, predicted_end_to_end_latency_ms=round(latency, 3), predicted_peak_memory_mb=analytical.predicted_peak_memory_mb, predicted_tokens_per_second=round(workload.requested_output_tokens / (latency / 1000), 3) if workload.requested_output_tokens and latency else None, latency_interval_ms=PredictionInterval(lower=round(max(0, latency - interval), 3), upper=round(latency + interval, 3)), calibration_sample_count=self.state.training_count, assumptions=["Latency = intercept + a·prefill + b·decode for this architecture, with non-negative a and b fitted by least squares to measured runs only; the physics terms come from the analytical formula.", "Peak memory remains an analytical estimate in Phase 3; no learned memory calibrator has been trained.", "Interval is ±1.96 training-residual standard deviations; interpret as heuristic uncertainty, not a guarantee."])


calibration_service = CalibrationService()
