from app.calibration.service import CalibrationService
from app.digital_twin.simulator import analytical_prediction
from app.schemas.workloads import Architecture, BenchmarkRecord, WorkloadRequest, WorkloadType


def measured_record(index: int, input_tokens: int) -> BenchmarkRecord:
    workload = WorkloadRequest(architecture=Architecture.DECODER, workload_type=WorkloadType.AUTOREGRESSIVE_GENERATION, input_tokens=input_tokens, requested_output_tokens=16, cache_enabled=True)
    # Deliberately a measured test fixture, never generated or shown as a real benchmark claim.
    latency = 10 + 0.12 * input_tokens + (index % 2)
    return BenchmarkRecord(id=f"measurement-{index}", created_at="2026-01-01T00:00:00Z", source="measured", benchmark_version="test", workload=workload, end_to_end_latency_ms=latency)


def test_analytical_twin_labels_assumptions():
    prediction = analytical_prediction(WorkloadRequest(architecture=Architecture.DECODER, workload_type=WorkloadType.AUTOREGRESSIVE_GENERATION, input_tokens=128, requested_output_tokens=32, cache_enabled=True))
    assert prediction.prediction_source == "analytical"
    assert prediction.latency_interval_ms.upper > prediction.latency_interval_ms.lower
    assert "Portable analytical" in prediction.assumptions[0]


def test_calibration_uses_measured_records_and_holds_out_larger_lengths():
    service = CalibrationService()
    records = [measured_record(i, length) for i, length in enumerate([16, 16, 32, 32, 64, 64, 128, 128, 256, 256])]
    report = service.train(records)
    assert report.status == "trained"
    assert report.held_out_record_count == 4
    assert report.calibrated_latency is not None
    prediction = service.predict(records[-1].workload)
    assert prediction.prediction_source == "calibrated"
    assert prediction.calibration_sample_count == report.training_record_count


def test_calibration_never_trains_on_synthetic_only():
    service = CalibrationService()
    synthetic = measured_record(1, 64).model_copy(update={"source": "synthetic"})
    report = service.train([synthetic] * 10)
    assert report.status == "insufficient_data"
    assert report.measured_record_count == 0
