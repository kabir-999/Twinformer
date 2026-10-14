import os
from pathlib import Path

test_db = "/tmp/transformer_twin_test.sqlite3"
Path(test_db).unlink(missing_ok=True)
os.environ["TRANSFORMER_TWIN_DB"] = test_db

from fastapi.testclient import TestClient
from app.main import app
from app.schemas.workloads import BenchmarkRecord, WorkloadRequest


def test_overview_and_synthetic_simulation():
    with TestClient(app) as client:
        overview = client.get("/api/v1/overview")
        assert overview.status_code == 200
        assert overview.json()["measured_record_count"] == 0
        response = client.post("/api/v1/simulation/synthetic", json={
            "architecture": "decoder_only", "workload_type": "autoregressive_generation",
            "input_tokens": 128, "requested_output_tokens": 32, "cache_enabled": True,
        })
        assert response.status_code == 201
        assert response.json()["source"] == "synthetic"


def test_rejects_unsupported_workload():
    with TestClient(app) as client:
        response = client.post("/api/v1/simulation/synthetic", json={
            "architecture": "encoder_only", "workload_type": "autoregressive_generation", "input_tokens": 32,
        })
    assert response.status_code == 422


def test_measured_benchmark_is_persisted_only_after_engine_success(monkeypatch):
    def fake_run(request):
        return BenchmarkRecord(id="real-run", created_at="2026-01-01T00:00:00Z", source="measured", benchmark_version="test", workload=WorkloadRequest(**request.model_dump()), end_to_end_latency_ms=12.0)
    monkeypatch.setattr("app.api.routes.engine.run", fake_run)
    with TestClient(app) as client:
        response = client.post("/api/v1/benchmarks", json={"architecture": "encoder_only", "workload_type": "encoder_inference", "input_tokens": 32, "repetitions": 1, "warmup_runs": 0})
    assert response.status_code == 201
    assert response.json()["source"] == "measured"
