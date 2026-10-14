import os
from pathlib import Path

test_db = "/tmp/transformer_twin_studies_test.sqlite3"
Path(test_db).unlink(missing_ok=True)
os.environ["TRANSFORMER_TWIN_DB"] = test_db

from fastapi.testclient import TestClient

from app.advanced.service import log_log_exponent
from app.benchmark.engine import BenchmarkUnavailable
from app.main import app
from app.schemas.workloads import BenchmarkRecord, WorkloadRequest

BASE = {"architecture": "decoder_only", "workload_type": "autoregressive_generation", "input_tokens": 16, "requested_output_tokens": 8, "repetitions": 1, "warmup_runs": 0}


def fake_run(request):
    latency = request.input_tokens * 2.0 / (2 if request.cache_enabled else 1) + request.batch_size
    return BenchmarkRecord(id=f"r-{request.input_tokens}-{request.batch_size}-{request.cache_enabled}", created_at="2026-01-01T00:00:00Z", source="measured",
                           benchmark_version="test", workload=WorkloadRequest(**request.model_dump()), end_to_end_latency_ms=latency)


def test_log_log_exponent_recovers_linear_and_quadratic():
    assert log_log_exponent([1, 2, 4, 8], [3, 6, 12, 24]) == 1.0
    assert log_log_exponent([1, 2, 4, 8], [1, 4, 16, 64]) == 2.0
    assert log_log_exponent([1, 2], [1, 2]) is None


def test_scaling_study_persists_records_and_study(monkeypatch):
    monkeypatch.setattr("app.api.routes.engine.run", fake_run)
    with TestClient(app) as client:
        before = client.get("/api/v1/overview").json()["measured_record_count"]
        response = client.post("/api/v1/studies/scaling", json={"base": BASE, "axis": "input_tokens", "values": [16, 32, 64]})
        assert response.status_code == 201
        body = response.json()
        assert [p["value"] for p in body["points"]] == [16, 32, 64]
        assert body["latency_scaling_exponent"] is not None
        assert client.get("/api/v1/studies", params={"kind": "scaling"}).json()[0]["id"] == body["id"]
        assert client.get("/api/v1/overview").json()["measured_record_count"] == before + 3


def test_failed_study_persists_nothing(monkeypatch):
    calls = []

    def flaky(request):
        calls.append(request)
        if len(calls) == 2:
            raise BenchmarkUnavailable("boom")
        return fake_run(request)
    monkeypatch.setattr("app.api.routes.engine.run", flaky)
    with TestClient(app) as client:
        before = client.get("/api/v1/overview").json()["measured_record_count"]
        response = client.post("/api/v1/studies/scaling", json={"base": BASE, "axis": "batch_size", "values": [1, 2, 4]})
        assert response.status_code == 503
        assert client.get("/api/v1/overview").json()["measured_record_count"] == before


def test_cache_comparison_and_validation(monkeypatch):
    monkeypatch.setattr("app.api.routes.engine.run", fake_run)
    with TestClient(app) as client:
        response = client.post("/api/v1/studies/cache", json={"base": BASE})
        assert response.status_code == 201
        assert response.json()["speedup"] > 1
        bad = client.post("/api/v1/studies/cache", json={"base": {**BASE, "requested_output_tokens": 1}})
        assert bad.status_code == 422


def test_scaling_rejects_duplicate_values():
    with TestClient(app) as client:
        assert client.post("/api/v1/studies/scaling", json={"base": BASE, "axis": "input_tokens", "values": [8, 8]}).status_code == 422
