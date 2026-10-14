import os

from app.research.service import build_manifest, render_report, run_ablation
from app.calibration.service import CalibrationService
from tests.test_calibration import measured_record
from app.schemas.workloads import BenchmarkRecord


def records():
    return [measured_record(i, n).model_copy(update={"benchmark_version": "phase4-local-v2", "variability_ms": 0.5}) for i, n in enumerate([16, 16, 32, 32, 64, 64, 128, 128, 256, 256])]


def test_ablation_has_baselines_and_one_row_per_dropped_feature():
    report = run_ablation(records())
    names = [r.variant for r in report.rows]
    assert report.status == "complete" and report.held_out_record_count == 4
    assert names[:3] == ["analytical_baseline", "intercept_only (train mean)", "ridge_all_features"]
    assert sum(n.startswith("ridge_without_") for n in names) == 9
    assert 0 <= report.interval_coverage_percent <= 100
    # latency is linear in input length in the fixture, so dropping input_tokens must hurt
    mae = {r.variant: r.metrics.mae for r in report.rows}
    assert mae["ridge_without_input_tokens"] > mae["ridge_all_features"]


def test_ablation_insufficient_data():
    assert run_ablation(records()[:3]).status == "insufficient_data"


def test_report_is_populated_only_from_saved_data():
    recs = records()
    cal = CalibrationService()
    cal.train(recs)
    manifest = build_manifest(recs, [], None, {"indexed": False})
    text = render_report(recs, manifest, cal.last_report, run_ablation(recs), [], None, "m.json")
    assert "Not run: no completed studies." in text
    assert "Not run: no retrieval evaluation" in text
    assert "| decoder_only | autoregressive_generation | on | 10 |" in text
    assert manifest["measured_records"]["count"] == 10


def test_report_flags_missing_measurements_and_excludes_synthetic():
    synthetic = measured_record(0, 16).model_copy(update={"source": "synthetic"})
    manifest = build_manifest([synthetic], [], None, {})
    cal = CalibrationService()
    cal.train([synthetic])
    text = render_report([synthetic], manifest, cal.last_report, run_ablation([synthetic]), [], None, "m.json")
    assert "no measured records from the current benchmark version" in text
    assert manifest["synthetic_records_excluded"] == 1
