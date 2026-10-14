import os

os.environ.setdefault("TRANSFORMER_TWIN_DB", "/tmp/transformer_twin_test.sqlite3")

from fastapi.testclient import TestClient

from app.compare.service import CompareRequest, CompareResult, CompareService, ModelRun, build_prompt, heatmap, pretty_token, tidy_output
from app.main import app


def test_each_family_gets_its_own_prompt_style():
    assert build_prompt("encoder_only", "summarize", " hello ") == "hello"
    assert build_prompt("encoder_decoder", "translate_fr", "hi").startswith("Translate English to French:")
    assert build_prompt("decoder_only", "answer", "Why?") == "Question: Why?\nAnswer:"
    assert build_prompt("decoder_only", "continue", "Once upon") == "Once upon"
    grounded = build_prompt("decoder_only", "answer", "Passage text.", "What is it?")
    assert "Context: Passage text." in grounded and "Question: What is it?" in grounded
    assert "Context: Passage text." in build_prompt("encoder_decoder", "answer", "Passage text.", "What is it?")


def test_token_cleanup_and_output_trimming():
    assert pretty_token("Ġworld") == " world" and pretty_token("▁the") == " the" and pretty_token("Ċ") == "\\n"
    assert tidy_output("Paris.\nWhat is next?\nParis.", "answer") == "Paris."
    assert tidy_output("line one\nline two", "continue") == "line one\nline two"


def test_heatmap_is_capped_and_flags_truncation():
    n = 40
    h = heatmap("self_causal", [f"t{i}" for i in range(n)], [f"t{i}" for i in range(n)], [[1 / n] * n for _ in range(n)], 3)
    assert len(h.rows) == len(h.cols) == len(h.values) == 32 and h.truncated and h.layer == 3
    small = heatmap("cross", ["a"], ["x", "y"], [[0.3, 0.7]], 0)
    assert not small.truncated and small.values == [[0.3, 0.7]]


def test_heatmap_keeps_the_most_attended_tokens_in_original_order():
    import numpy as np
    n = 50
    m = np.full((n, n), 0.001)
    m[:, 41] = 1.0   # every token attends heavily to token 41
    m[:, 7] = 0.8
    h = heatmap("self_bidirectional", [f"t{i}" for i in range(n)], [f"t{i}" for i in range(n)], m.tolist(), 0)
    assert h.col_index == sorted(h.col_index) and 41 in h.col_index and 7 in h.col_index
    assert h.row_index == h.col_index and h.cols[h.col_index.index(41)] == "t41" and h.truncated


def test_missing_model_is_reported_per_model_not_fatal(monkeypatch):
    service = CompareService()
    monkeypatch.setattr(service, "available_models", lambda: {"defaults": {"encoder_only": "a/enc", "decoder_only": "a/dec", "encoder_decoder": "a/s2s"}})
    result = service.run(CompareRequest(text="hello"))
    assert [r.status for r in result.runs] == ["unavailable"] * 3
    assert "Allow downloading" in result.runs[0].error


def test_compare_endpoint_persists_result(monkeypatch):
    fake = CompareResult(id="c1", created_at="2026-01-01T00:00:00Z", request=CompareRequest(text="hi"), notes=[],
                         runs=[ModelRun(architecture="encoder_only", model_id="m", embedding_dim=4)])
    monkeypatch.setattr("app.api.routes.compare_service.run", lambda request: fake)
    with TestClient(app) as client:
        response = client.post("/api/v1/compare", json={"text": "hi"})
        assert response.status_code == 200 and response.json()["runs"][0]["embedding_dim"] == 4
        assert any(s["id"] == "c1" for s in client.get("/api/v1/studies", params={"kind": "compare"}).json())
        assert client.post("/api/v1/compare", json={"text": ""}).status_code == 422
