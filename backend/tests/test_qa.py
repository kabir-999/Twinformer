import json
import os

os.environ.setdefault("TRANSFORMER_TWIN_DB", "/tmp/transformer_twin_test.sqlite3")

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.qa import (
    AnswersRequest, ArchAnswer, JudgeError, Metrics, SourcePassage, ThreeWayAnswer, Verdict, clean_text, extractive_answer, grounding,
    groq_judge, heuristic_judge, pick_defaults, repetition, split_sentences,
)


def test_clean_text_joins_pdf_lines_and_fixes_ligatures():
    raw = "Simple and\nCompound Interest,\nEfﬁciency,\n\nSecond paragraph\nhere."
    assert clean_text(raw) == "Simple and Compound Interest, Efficiency,\n\nSecond paragraph here."


def test_split_sentences_drops_fragments():
    text = "Short.\n\nThe QA section covers middle-school mathematics and arithmetic. Tiny bit. Modern maths has grown in recent years."
    assert split_sentences(text) == ["The QA section covers middle-school mathematics and arithmetic.", "Modern maths has grown in recent years."]


def test_split_sentences_wraps_unpunctuated_run_ons():
    run_on = " ".join(f"Topic{i}" for i in range(120))
    pieces = split_sentences(run_on)
    assert len(pieces) > 2 and all(len(p) <= 260 for p in pieces)


def test_very_short_answer_is_scored_low_but_not_as_missing():
    short = answer("encoder_decoder", "Algebra", grounding=1.0, relevance=0.2, repetition=0.0)
    verdict = heuristic_judge([short])
    assert verdict.per_answer[0].score == 20.0 and "very short" in verdict.per_answer[0].comment


def test_grounding_and_repetition():
    sources = "The QA section covers algebra arithmetic geometry and modern maths"
    assert grounding("QA covers algebra and geometry", sources) == 1.0
    assert grounding("Bananas grow tropical", sources) == 0.0
    assert grounding("and the", sources) is None
    assert repetition("a b c d e f g") == 0.0
    assert repetition("go go go go go go go") > 0.5


def test_extractive_answer_picks_nearest_sentences_in_order():
    passages = [SourcePassage(rank=1, document="d", location="p1", score=0.5, text="Cats purr when they are content and relaxed at home. Rockets burn propellant to produce thrust for launch.")]
    vocab = {"cats": 0, "rockets": 1}

    def embed(texts):
        vectors = np.array([[1.0 if "cat" in t.lower() else 0.0, 1.0 if "rocket" in t.lower() else 0.0] for t in texts])
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    answer = extractive_answer(embed, "why do cats purr?", passages, n=1)
    assert answer.startswith("Cats purr") and answer.endswith("[1]")


def answer(arch, text, **metrics):
    return ArchAnswer(architecture=arch, model_id="m", method="x", answer=text, metrics=Metrics(words=len((text or "").split()), **metrics))


def test_heuristic_judge_prefers_grounded_relevant_answer_over_repetitive_one():
    good = answer("decoder_only", "QA covers algebra and geometry topics broadly.", grounding=0.95, relevance=0.7, repetition=0.0)
    bad = answer("encoder_decoder", "maths maths maths maths maths maths maths", grounding=0.3, relevance=0.2, repetition=0.8)
    declined = answer("encoder_only", "The sources do not contain the answer to this.", grounding=0.9, relevance=0.5, repetition=0.0)
    verdict = heuristic_judge([declined, good, bad])
    assert verdict.winner == "decoder_only" and verdict.source == "heuristic"
    scores = {v.architecture: v.score for v in verdict.per_answer}
    assert scores["decoder_only"] > scores["encoder_only"] == 30.0 > scores["encoder_decoder"]
    assert "repeats itself" in {v.architecture: v.comment for v in verdict.per_answer}["encoder_decoder"]


def test_close_scores_are_a_tie_with_no_winner():
    a = answer("decoder_only", "QA covers algebra and geometry topics broadly.", grounding=0.95, relevance=0.7, repetition=0.0)
    b = answer("encoder_only", "QA covers algebra and geometry topics widely.", grounding=0.95, relevance=0.69, repetition=0.0)
    verdict = heuristic_judge([a, b])
    assert verdict.winner is None and "tied" in verdict.summary


def test_heuristic_judge_handles_no_answers():
    empty = ArchAnswer(architecture="decoder_only", model_id="m", method="x", error="not downloaded")
    verdict = heuristic_judge([empty])
    assert verdict.winner is None and "usable" in verdict.summary


def groq_client(payload, status=200, capture=None):
    def handler(request):
        if capture is not None:
            capture.append(json.loads(request.content))
        return httpx.Response(status, json={"choices": [{"message": {"content": json.dumps(payload)}}]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_groq_judge_is_blind_and_maps_ids_back_to_architectures():
    sources = [SourcePassage(rank=1, document="d", location="p1", score=0.5, text="Paris is the capital of France.")]
    answers = [answer("encoder_only", "Paris is the capital of France. [1]"), answer("decoder_only", "It is Lyon."), answer("encoder_decoder", "Paris.")]
    sent = []
    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        text = body["messages"][1]["content"]
        lyon_id = next(i for i in (1, 2, 3) if "Lyon" in text.split(f"Answer {i}: ")[1].split("\n")[0])
        best_id = next(i for i in (1, 2, 3) if "[1]" in text.split(f"Answer {i}: ")[1].split("\n")[0])
        result = {"answers": [{"id": i, "correctness": 1 if i == lyon_id else 5, "grounding": 5, "completeness": 4, "comment": f"c{i}"} for i in (1, 2, 3)], "best_id": best_id, "reasoning": "Because."}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})
    verdict = groq_judge("What is the capital of France?", sources, answers, "key", httpx.Client(transport=httpx.MockTransport(handler)))
    prompt = sent[0]["messages"][1]["content"]
    assert "encoder" not in prompt.lower() and "decoder" not in prompt.lower()
    assert verdict.winner == "encoder_only" and verdict.source == "groq"
    scores = {v.architecture: v.score for v in verdict.per_answer}
    assert scores["decoder_only"] < scores["encoder_decoder"]


def test_groq_judge_errors_are_clean():
    sources = [SourcePassage(rank=1, document="d", location="", score=0.1, text="t")]
    two = [answer("decoder_only", "a b c"), answer("encoder_decoder", "d e f")]
    with pytest.raises(JudgeError, match="API key"):
        groq_judge("q", sources, two, "bad", groq_client({}, status=401))
    with pytest.raises(JudgeError, match="unexpected format"):
        groq_judge("q", sources, two, "k", groq_client({"nope": 1}))
    with pytest.raises(JudgeError, match="at least two"):
        groq_judge("q", sources, [answer("decoder_only", "a b c")], "k", groq_client({}))


def test_pick_defaults_prefers_instruct_and_larger_t5():
    local = {"encoder_only": ["x/other", "sentence-transformers/all-MiniLM-L6-v2"], "decoder_only": ["Qwen/Qwen1.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"], "encoder_decoder": ["google/flan-t5-small", "google/flan-t5-base"]}
    assert pick_defaults(local) == {"encoder_only": "sentence-transformers/all-MiniLM-L6-v2", "decoder_only": "Qwen/Qwen2.5-0.5B-Instruct", "encoder_decoder": "google/flan-t5-base"}


def test_endpoints(monkeypatch, tmp_path):
    fake = ThreeWayAnswer(question="q", sources=[], answers=[], verdicts=[], notes=[])
    seen = {}
    monkeypatch.setattr("app.api.routes.answer_three_ways", lambda request, rag, compare, key: seen.update(key=key) or fake)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("# comment\nGROQ_API_KEY=\"secret\"\n")
    monkeypatch.setattr("app.secrets.PROJECT_ROOT", tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/v1/judge/status").json() == {"groq_configured": True}
        assert client.post("/api/v1/rag/answers", json={"question": "q", "use_groq": True}).status_code == 200
        assert seen["key"] == "secret"
        (tmp_path / ".env").write_text("")  # edited live, no restart
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert client.get("/api/v1/judge/status").json() == {"groq_configured": False}
        assert client.post("/api/v1/rag/answers", json={"question": ""}).status_code == 422
