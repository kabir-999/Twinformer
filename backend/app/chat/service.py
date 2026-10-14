"""Chat over selected documents: follow-ups, optional architecture routing, three-way answers, persisted history."""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Callable, Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field

from app.compare.service import CompareService, CompareUnavailable
from app.rag.locator import locate_sources, parse_location
from app.rag.qa import Arch, AnswersRequest, ArchAnswer, JudgeError, SourcePassage, ThreeWayAnswer, answer_three_ways, clean_text, dedupe_repeats, groq_judge
from app.rag.service import RagService
from app.rag.translate import NLLB, decoder_translate, pick_translator, target_language, translate_with

SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
REWRITE_MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class ChatRequest(BaseModel):
    chat_id: str | None = None
    question: str = Field(min_length=1, max_length=1000)
    documents: list[str] = Field(default_factory=list)
    mode: Literal["all", "auto"] = "all"
    top_k: int = Field(default=4, ge=1, le=8)
    allow_model_download: bool = False


class Route(BaseModel):
    intent: str
    architectures: list[Arch]
    reason: str
    sentiment: bool = False
    language: str | None = None
    language_code: str | None = None


class ChatTurn(BaseModel):
    id: str
    created_at: str
    question: str
    standalone_question: str
    mode: Literal["all", "auto"]
    route: Route | None = None
    result: ThreeWayAnswer


class Chat(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    documents: list[str]
    turns: list[ChatTurn] = Field(default_factory=list)


class ChatSummary(BaseModel):
    id: str
    title: str
    updated_at: str
    turn_count: int
    document_count: int


# --- routing -----------------------------------------------------------------------------
def route_question(question: str) -> Route:
    """Pick the architecture that best fits the request. Rules, not a model, so the reason shown is the real reason."""
    q = question.lower()
    if re.search(r"\b(sentiment|positive or negative|emotion(al)? tone|tone of)\b", q):
        return Route(intent="sentiment", architectures=["encoder_only"], sentiment=True, reason=(
            "Sentiment analysis is a classification task. Encoder-only models read the whole text in both directions and are the standard choice; "
            "a DistilBERT fine-tuned for sentiment classifies each retrieved passage."))
    if re.search(r"\b(translate|translation|translated)\b|\bin (hindi|french|german|spanish|romanian|italian|russian|chinese|arabic)\b", q):
        language = target_language(question)
        if language is None:
            return Route(intent="translate", architectures=["encoder_decoder"], reason=(
                "Translation maps one sequence to another, which is what encoder-decoder models are built for, but you did not say which language to translate into."))
        code, name = language
        return Route(intent="translate", architectures=["encoder_decoder"], language=name, language_code=code, reason=(
            f"Translation maps one sequence to another, which is exactly what encoder-decoder models are built for. A dedicated translation model "
            f"(NLLB-200, or MarianMT if that is what is installed) is used: an encoder reads each sentence and a decoder writes the {name}."))
    if re.search(r"\b(summari[sz]e|summary|tl;?dr|key points|overview|main ideas)\b", q):
        return Route(intent="summarize", architectures=["decoder_only"], reason=(
            "Summaries need a long, free-form answer over several passages. The instruction-tuned decoder-only model copes with that context best here; "
            "the small encoder-decoder model is limited to about 512 input tokens."))
    if re.search(r"^\s*(where|which (page|slide|file|chapter)|find|locate|show me)\b", q):
        return Route(intent="lookup", architectures=["encoder_only"], reason=(
            "This is a lookup: the encoder-only model ranks sentences by meaning and returns the exact text from your files, with nothing invented."))
    return Route(intent="qa", architectures=["decoder_only"], reason=(
        "Open-ended questions need an explanation in the model's own words, which decoder-only language models do best. "
        "Switch to “Compare all three” to see how the other designs answer."))


# --- follow-ups --------------------------------------------------------------------------
def rewrite_followup(history: list[tuple[str, str]], question: str, groq_key: str | None, client: httpx.Client | None = None) -> str:
    """Make a follow-up question stand alone so retrieval works. Uses Groq when a key exists, else prefixes the previous question."""
    if not history:
        return question
    fallback = f"{history[-1][0].rstrip('?. ')} — {question}"
    if not groq_key:
        return fallback
    convo = "\n".join(f"User: {q}\nAssistant: {a[:300]}" for q, a in history[-3:])
    body = {"model": REWRITE_MODEL, "temperature": 0, "messages": [
        {"role": "system", "content": "Rewrite the user's last message as one standalone question that makes sense without the conversation. Keep the original language and intent. Reply with only the question."},
        {"role": "user", "content": f"{convo}\n\nLast message: {question}"}]}
    owns = client is None
    client = client or httpx.Client(timeout=20)
    try:
        response = client.post(GROQ_URL, headers={"Authorization": f"Bearer {groq_key}"}, json=body)
        text = response.json()["choices"][0]["message"]["content"].strip() if response.status_code == 200 else ""
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        text = ""
    finally:
        if owns:
            client.close()
    return text if 3 < len(text) <= 400 else fallback


# --- sentiment (encoder-only classifier) -------------------------------------------------
_sentiment_cache: dict[str, tuple] = {}


def classify_sentiment(texts: list[str], allow_download: bool) -> list[tuple[str, float]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if SENTIMENT_MODEL not in _sentiment_cache:
        local = not allow_download
        _sentiment_cache[SENTIMENT_MODEL] = (AutoTokenizer.from_pretrained(SENTIMENT_MODEL, local_files_only=local),
                                             AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL, local_files_only=local).eval())
    tokenizer, model = _sentiment_cache[SENTIMENT_MODEL]
    enc = tokenizer(texts, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.inference_mode():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    return [(model.config.id2label[int(p.argmax())], float(p.max())) for p in probs]


def sentiment_result(question: str, rag: RagService, documents: list[str], top_k: int, allow_download: bool,
                     classifier: Callable[[list[str], bool], list[tuple[str, float]]] = classify_sentiment) -> ThreeWayAnswer:
    citations = rag.retrieve(question, max(top_k, 6), documents=set(documents) or None)
    sources = [SourcePassage(rank=c.rank, document=c.document, location=c.location, text=clean_text(c.text), score=c.score) for c in citations]
    entry = ArchAnswer(architecture="encoder_only", model_id=SENTIMENT_MODEL, method="Classifies each retrieved passage as positive or negative")
    try:
        labels = classifier([s.text for s in sources], allow_download)
        positive = sum(label.upper().startswith("POS") for label, _ in labels)
        majority = "positive" if positive * 2 >= len(labels) else "negative"
        lines = " · ".join(f"[{s.rank}] {label.lower()} {round(p * 100)}%" for s, (label, p) in zip(sources, labels))
        entry.answer = f"Overall the passages read as mostly {majority} ({max(positive, len(labels) - positive)} of {len(labels)}). {lines}"
    except Exception as exc:  # model missing or failed to load: report it, do not crash the chat
        entry.error = f"Sentiment model unavailable: {type(exc).__name__}. Tick “Allow downloading” or download {SENTIMENT_MODEL}."
    return ThreeWayAnswer(question=question, sources=sources, answers=[entry], verdicts=[], notes=[
        "Sentiment is judged on the passages retrieved for your question, so it describes those passages, not the whole file.",
        "This classifier is trained on movie reviews; course notes are mostly neutral, so treat the labels with caution."])


# --- translation -------------------------------------------------------------------------
def _passages(question: str, rag: RagService, documents: list[str], top_k: int) -> list[SourcePassage]:
    return [SourcePassage(rank=c.rank, document=c.document, location=c.location, text=clean_text(c.text), score=c.score)
            for c in rag.retrieve(question, top_k, documents=set(documents) or None)]


def translation_result(question: str, route: Route, rag: RagService, compare: CompareService, documents: list[str], allow_download: bool,
                       groq_key: str | None, run_all: bool, marian=None, decoder=decoder_translate, judge=groq_judge, picker=pick_translator) -> ThreeWayAnswer:
    """Translate the located slide/page (or the best-matching passages). Encoder-only models cannot write text, so they sit this out."""
    location = parse_location(question)
    sources = (locate_sources(rag, location, documents, keep_lines=True) if location else []) or _passages(question, rag, documents, 2)
    notes = ["The text translated is the source shown below, not a summary of it."]
    if route.language is None:
        entry = ArchAnswer(architecture="encoder_decoder", model_id="-", method="Translation", error="Tell me which language to translate into, for example: “translate the first slide to Hindi”.")
        return ThreeWayAnswer(question=question, sources=sources, answers=[entry], verdicts=[], notes=notes)
    texts = [(s, s.text[:1500]) for s in sources]

    def render(translate) -> str:
        parts = [(f"[{s.rank}] {s.document.split('/')[-1]} · {s.location}\n" if len(texts) > 1 else "") + dedupe_repeats(translate(t)) for s, t in texts]
        return "\n\n".join(parts)

    answers: list[ArchAnswer] = []
    wanted: list[Arch] = ["decoder_only", "encoder_decoder"] if run_all else list(route.architectures)
    for arch in ("encoder_decoder", "decoder_only"):
        if arch not in wanted:
            continue
        if arch == "encoder_decoder":
            model_id = picker(route.language_code, allow_download)
            entry = ArchAnswer(architecture=arch, model_id=model_id, method=f"Dedicated translation model: an encoder reads each sentence, a decoder writes the {route.language}")
        else:
            model_id = compare.available_models()["defaults"]["decoder_only"]
            entry = ArchAnswer(architecture=arch, model_id=model_id, method=f"General instruction-tuned language model asked to translate into {route.language}")
        started = time.perf_counter()
        try:
            loaded = compare.get_model(arch, model_id, allow_download)
            if arch == "encoder_decoder":
                encdec = marian or (lambda l, t: translate_with(l, model_id, t, route.language_code))
                entry.answer = render(lambda t: encdec(loaded, t))
            else:
                entry.answer = render(lambda t: decoder(loaded, t, route.language))
        except CompareUnavailable as exc:
            size = " (about 2.5 GB)" if arch == "encoder_decoder" and model_id == NLLB else ""
            entry.error = f"{exc} Tick “Allow downloads” under the input box to fetch it once{size}."
        entry.seconds = round(time.perf_counter() - started, 2)
        answers.append(entry)
    if run_all:
        notes.append("Encoder-only models cannot generate text, so only the decoder-only and encoder-decoder models took part.")
    verdicts = []
    if len([a for a in answers if a.answer]) > 1:
        if groq_key:
            try:
                verdicts.append(judge(question, sources, answers, groq_key, kind="translation"))
            except JudgeError as exc:
                notes.append(f"Groq judge failed: {exc}.")
        else:
            notes.append("Translation quality cannot be checked automatically. Add GROQ_API_KEY to .env to have Groq judge which translation is better.")
    return ThreeWayAnswer(question=question, sources=sources, answers=answers, verdicts=verdicts, notes=notes)


# --- one chat turn -----------------------------------------------------------------------
def best_answer_text(turn: ChatTurn) -> str:
    result = turn.result
    winner = result.verdicts[0].winner if result.verdicts else None
    pick = next((a for a in result.answers if a.architecture == winner and a.answer), None) or next((a for a in result.answers if a.answer), None)
    return pick.answer if pick and pick.answer else ""


def run_turn(request: ChatRequest, chat: Chat | None, rag: RagService, compare: CompareService, groq_key: str | None,
             answer_fn: Callable[..., ThreeWayAnswer] = answer_three_ways, rewrite_fn: Callable[..., str] = rewrite_followup,
             translate_fn: Callable[..., ThreeWayAnswer] = translation_result) -> ChatTurn:
    history = [(t.question, best_answer_text(t)) for t in chat.turns] if chat else []
    standalone = rewrite_fn(history, request.question, groq_key)
    detected = route_question(standalone)
    auto = request.mode == "auto"
    route = detected if auto else None
    top_k = 6 if detected.intent == "summarize" else request.top_k
    if detected.intent == "translate":
        result = translate_fn(standalone, detected, rag, compare, request.documents, request.allow_model_download, groq_key, run_all=not auto)
    elif auto and detected.sentiment:
        result = sentiment_result(standalone, rag, request.documents, top_k, request.allow_model_download)
    else:
        location = parse_location(standalone)
        fixed = locate_sources(rag, location, request.documents) if location else None
        extra = {"fixed_sources": fixed} if fixed else {}
        result = answer_fn(AnswersRequest(question=standalone, top_k=top_k, documents=request.documents, architectures=detected.architectures if auto else None,
                                          use_groq=bool(groq_key), allow_model_download=request.allow_model_download), rag, compare, groq_key, **extra)
        if location and not fixed:
            result.notes.append(f"Could not find a {location.unit} {location.position} in the selected files, so the closest passages were used instead.")
    return ChatTurn(id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(), question=request.question, standalone_question=standalone,
                    mode=request.mode, route=route, result=result)


def new_chat(first: ChatRequest) -> Chat:
    now = datetime.now(timezone.utc).isoformat()
    return Chat(id=str(uuid4()), title=first.question.strip()[:70], created_at=now, updated_at=now, documents=first.documents)
