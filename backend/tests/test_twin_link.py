import os

os.environ.setdefault("TRANSFORMER_TWIN_DB", "/tmp/transformer_twin_test.sqlite3")

from app import twin_link
from app.calibration.service import CalibrationService
from app.rag.qa import ArchAnswer


def answer(arch="decoder_only", seconds=2.0, inp=800, out=40, cold=False, text="ok", model="Qwen/Qwen2.5-1.5B-Instruct"):
    return ArchAnswer(architecture=arch, model_id=model, method="x", answer=text, seconds=seconds, input_tokens=inp, output_tokens=out, cold_start=cold)


def test_only_warm_generated_answers_become_measurements():
    answers = [answer(), answer("encoder_decoder", 1.0, 300, 20, model="google/flan-t5-base"), answer(cold=True), answer(text=None),
               ArchAnswer(architecture="encoder_only", model_id="m", method="x", answer="quoted", seconds=0.1)]
    records = twin_link.records_from_answers(answers)
    assert [r.workload.architecture.value for r in records] == ["decoder_only", "encoder_decoder"]
    first = records[0]
    assert first.source == "measured" and first.benchmark_version == twin_link.CHAT_VERSION and first.end_to_end_latency_ms == 2000
    assert (first.input_token_count, first.output_token_count, first.workload.resolved_model_id) == (800, 40, "Qwen/Qwen2.5-1.5B-Instruct")


def test_predictions_are_made_before_learning_and_skip_cold_runs():
    calibration = CalibrationService()
    warm, cold = answer(), answer(cold=True)
    twin_link.annotate_predictions([warm, cold], calibration)
    assert warm.predicted_seconds and warm.predicted_seconds > 0 and cold.predicted_seconds is None


def test_twin_learns_from_chat_turns_and_gets_better_than_the_formula():
    calibration = CalibrationService()
    records = []
    for i, inp in enumerate([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]):
        for arch, model in (("decoder_only", "Qwen/Qwen2.5-1.5B-Instruct"), ("encoder_decoder", "google/flan-t5-base")):
            seconds = (0.004 * inp + 0.05 * 40 + 1.0) if arch == "decoder_only" else (0.001 * inp + 0.02 * 20 + 0.3)
            records += twin_link.records_from_answers([answer(arch, seconds + (i % 2) * 0.05, inp, 40 if arch == "decoder_only" else 20, model=model)])
    report = calibration.train(records)
    assert report.status == "trained"
    assert report.calibrated_latency.mae < report.baseline_latency.mae           # learning from real timings beats the first-principles guess
    assert report.calibrated_latency.mae < 200                                    # and is accurate to within 0.2 s on held-out lengths here
    status = twin_link.status(records, calibration)
    assert status["chat_records"] == 20 and len(status["points"]) == 20 and status["calibration"]["status"] == "trained"
    estimate = twin_link.estimate(calibration, {"decoder_only": "Qwen/Qwen2.5-1.5B-Instruct", "encoder_decoder": "google/flan-t5-base"}, 4)
    assert [e["architecture"] for e in estimate] == ["decoder_only", "encoder_decoder"] and all(e["source"] == "calibrated" for e in estimate)
    assert estimate[0]["seconds"] > estimate[1]["seconds"]


def test_endpoints_exist():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        est = client.get("/api/v1/twin/estimate", params={"top_k": 4}).json()
        assert {e["architecture"] for e in est} == {"decoder_only", "encoder_decoder"} and all(e["seconds"] > 0 for e in est)
        st = client.get("/api/v1/twin/status").json()
        assert {"measured_total", "chat_records", "calibration", "points"} <= set(st)
