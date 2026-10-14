import os

os.environ.setdefault("TRANSFORMER_TWIN_DB", "/tmp/transformer_twin_test.sqlite3")

import httpx
import pytest
from fastapi.testclient import TestClient

from app.chat.service import Chat, ChatRequest, rewrite_followup, route_question, run_turn, sentiment_result
from app.main import app
from app.rag.qa import ArchAnswer, ThreeWayAnswer, Verdict, AnswerVerdict
from app.rag.service import Citation


def test_router_picks_architecture_with_a_reason():
    cases = {
        "What is the sentiment of chapter 3?": ("sentiment", ["encoder_only"]),
        "Translate the introduction to French": ("translate", ["encoder_decoder"]),
        "translate first slide to hindi": ("translate", ["encoder_decoder"]),
        "Translate this into Tamil": ("translate", ["encoder_decoder"]),
        "Summarise chapter 1": ("summarize", ["decoder_only"]),
        "Where is DVFS explained?": ("lookup", ["encoder_only"]),
        "Why does quantization save energy?": ("qa", ["decoder_only"]),
    }
    for question, (intent, archs) in cases.items():
        route = route_question(question)
        assert (route.intent, route.architectures) == (intent, archs), question
        assert len(route.reason) > 40
    assert route_question("sentiment of the notes").sentiment is True


def test_followup_is_rewritten_by_groq_or_falls_back_to_previous_question():
    assert rewrite_followup([], "What is DVFS?", "k") == "What is DVFS?"
    history = [("What is DVFS?", "It adjusts voltage and frequency.")]
    assert rewrite_followup(history, "Why is it used?", None) == "What is DVFS — Why is it used?"

    def handler(request):
        assert b"Why is it used?" in request.content
        return httpx.Response(200, json={"choices": [{"message": {"content": "Why is DVFS used?"}}]})
    assert rewrite_followup(history, "Why is it used?", "k", httpx.Client(transport=httpx.MockTransport(handler))) == "Why is DVFS used?"
    failing = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500, json={})))
    assert rewrite_followup(history, "Why is it used?", "k", failing) == "What is DVFS — Why is it used?"


class FakeRag:
    def retrieve(self, question, top_k, documents=None, **_):
        return [Citation(rank=1, document="Sem 7/a.pptx", chunk_id="a#0", char_start=0, char_end=5, location="slide 2", score=0.5, text="This is great and wonderful news.")]


def test_sentiment_result_uses_the_classifier_and_reports_missing_model():
    ok = sentiment_result("sentiment?", FakeRag(), ["Sem 7/a.pptx"], 4, False, classifier=lambda texts, dl: [("POSITIVE", 0.98)] * len(texts))
    assert "mostly positive" in ok.answers[0].answer and ok.answers[0].architecture == "encoder_only" and not ok.verdicts

    def missing(texts, dl):
        raise OSError("not cached")
    bad = sentiment_result("sentiment?", FakeRag(), [], 4, False, classifier=missing)
    assert bad.answers[0].error and "Allow downloading" in bad.answers[0].error


def fake_result(question="q"):
    answers = [ArchAnswer(architecture="decoder_only", model_id="m", method="x", answer="Because of reasons.")]
    return ThreeWayAnswer(question=question, sources=[], answers=answers, notes=[],
                          verdicts=[Verdict(source="heuristic", winner="decoder_only", summary="s", per_answer=[AnswerVerdict(architecture="decoder_only", score=90, comment="c")], caveat="c")])


def test_run_turn_passes_files_route_and_rewritten_question_through():
    seen = {}

    def answer_fn(request, rag, compare, key, fixed_sources=None):
        seen["request"] = request
        return fake_result(request.question)
    first = run_turn(ChatRequest(question="Summarise chapter 1", documents=["Sem 7/a.pptx"], mode="auto"), None, FakeRag(), None, None, answer_fn=answer_fn)
    assert seen["request"].documents == ["Sem 7/a.pptx"] and seen["request"].architectures == ["decoder_only"] and seen["request"].top_k == 6
    assert first.route.intent == "summarize"
    chat = Chat(id="c", title="t", created_at="x", updated_at="x", documents=[], turns=[first])
    second = run_turn(ChatRequest(question="and chapter 2?", mode="all"), chat, FakeRag(), None, None, answer_fn=answer_fn)
    assert second.standalone_question == "Summarise chapter 1 — and chapter 2?" and second.route is None
    assert seen["request"].architectures is None and seen["request"].question == second.standalone_question


def test_chat_endpoints_persist_history(monkeypatch):
    monkeypatch.setattr("app.chat.service.answer_three_ways", lambda request, rag, compare, key: fake_result(request.question))
    monkeypatch.setattr("app.api.routes.get_groq_key", lambda: None)
    monkeypatch.setattr("app.api.routes.run_turn", lambda request, chat, rag, compare, key: run_turn(request, chat, rag, compare, key, answer_fn=lambda r, *a: fake_result(r.question)))
    with TestClient(app) as client:
        first = client.post("/api/v1/chat", json={"question": "What is DVFS?", "documents": ["Sem 7/a.pptx"]})
        assert first.status_code == 200 and len(first.json()["turns"]) == 1
        chat_id = first.json()["id"]
        second = client.post("/api/v1/chat", json={"chat_id": chat_id, "question": "Why is it used?", "documents": []})
        assert [t["question"] for t in second.json()["turns"]] == ["What is DVFS?", "Why is it used?"]
        listed = {c["id"]: c for c in client.get("/api/v1/chats").json()}
        assert listed[chat_id]["turn_count"] == 2 and listed[chat_id]["title"] == "What is DVFS?"
        assert len(client.get(f"/api/v1/chats/{chat_id}").json()["turns"]) == 2
        assert client.delete(f"/api/v1/chats/{chat_id}").status_code == 204
        assert client.get(f"/api/v1/chats/{chat_id}").status_code == 404
        assert client.post("/api/v1/chat", json={"chat_id": "missing", "question": "q"}).status_code == 404
