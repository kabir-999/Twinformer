"""Answer one question three ways (encoder-only, decoder-only, encoder-decoder) from retrieved passages, then judge them."""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from typing import Any, Callable, Literal

import httpx
from pydantic import BaseModel, Field

from app.compare.service import DEFAULT_DOWNLOADS as DEFAULTS, CompareService, CompareUnavailable
from app.rag.service import RagService

Arch = Literal["encoder_only", "decoder_only", "encoder_decoder"]
ORDER: list[Arch] = ["encoder_only", "decoder_only", "encoder_decoder"]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
STOPWORDS = set("that this with from have were been which their there what when where will would about into than then them they your also such these those other some more most only over under each both while being does done make made very much many".split())
SYSTEM = ("Answer the question using only the numbered sources. Be concise (1-3 sentences). "
          "If the sources do not contain the answer, say so. Cite sources like [1].")
DECLINE = re.compile(r"(do(es)? not|don't|cannot|can't|no) (contain|mention|provide|say|include|have)|not (mentioned|provided|found)|unable to (find|answer)", re.I)


class AnswersRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=8)
    encoder_model_id: str | None = None
    decoder_model_id: str | None = None
    seq2seq_model_id: str | None = None
    documents: list[str] = Field(default_factory=list)          # restrict retrieval to these files (empty = all)
    architectures: list[Arch] | None = None                       # run only these families (None = all three)
    use_groq: bool = False
    allow_model_download: bool = False


class SourcePassage(BaseModel):
    rank: int
    document: str
    location: str
    text: str
    score: float


class Metrics(BaseModel):
    grounding: float | None = None    # share of the answer's content words found in the sources
    relevance: float | None = None    # cosine similarity between question and answer (encoder embeddings)
    repetition: float | None = None   # share of repeated 3-word phrases
    words: int = 0


class ArchAnswer(BaseModel):
    architecture: Arch
    model_id: str
    method: str
    answer: str | None = None
    seconds: float | None = None
    error: str | None = None
    metrics: Metrics | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cold_start: bool = False                 # the model was loaded for this answer, so its time is not representative
    predicted_seconds: float | None = None   # what the digital twin expected, before this run was learned from


class AnswerVerdict(BaseModel):
    architecture: Arch
    score: float            # 0-100
    comment: str


class Verdict(BaseModel):
    source: Literal["heuristic", "groq"]
    judge_model: str | None = None
    winner: Arch | None = None
    summary: str
    per_answer: list[AnswerVerdict]
    caveat: str


class ThreeWayAnswer(BaseModel):
    question: str
    sources: list[SourcePassage]
    answers: list[ArchAnswer]
    verdicts: list[Verdict]
    notes: list[str] = Field(default_factory=list)


class JudgeError(RuntimeError):
    pass


# --- text cleanup ------------------------------------------------------------------------
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


def clean_text(text: str, keep_lines: bool = False) -> str:
    """PDF/slide text arrives with one word per line and ligature glyphs; make it readable.
    keep_lines=True only fixes glyphs and blank lines, preserving the line structure (needed for translation)."""
    for glyph, plain in LIGATURES.items():
        text = text.replace(glyph, plain)
    if keep_lines:
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())
    paragraphs = (re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text))
    return re.sub(r"[ \t]{2,}", " ", "\n\n".join(p for p in paragraphs if p))


def _wrap(sentence: str, limit: int = 260) -> list[str]:
    """List-style slide/PDF text has no punctuation; break run-ons at word boundaries so answers stay readable."""
    pieces, current = [], ""
    for word in sentence.split():
        if len(current) + len(word) + 1 > limit and current:
            pieces.append(current)
            current = ""
        current = f"{current} {word}".strip()
    return pieces + [current] if current else pieces


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n\n|\s•\s", text)
    return [piece for p in parts if len(p.strip()) >= 30 and sum(c.isalpha() for c in p) >= 20 for piece in _wrap(p.strip())]


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 3 and w not in STOPWORDS]


def grounding(answer: str, sources_text: str) -> float | None:
    words = content_words(answer)
    if not words:
        return None
    known = set(content_words(sources_text))
    return round(sum(w in known for w in words) / len(words), 3)


def repetition(answer: str) -> float:
    tokens = answer.lower().split()
    grams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    return round(1 - len(set(grams)) / len(grams), 3) if grams else 0.0


def dedupe_repeats(text: str) -> str:
    """Small models sometimes loop ("This technique can reduce energy use. This technique can reduce energy use."): keep each sentence once."""
    seen, kept = set(), []
    for piece in re.split(r"(?<=[.!?।])\s+|\n+", text):
        key = re.sub(r"\W+", " ", piece.lower()).strip()
        if piece.strip() and key not in seen:
            seen.add(key)
            kept.append(piece.strip())
    return "\n".join(kept) if "\n" in text else " ".join(kept)


# --- the three answerers -----------------------------------------------------------------
def _cosine_rank(embed: Callable[[list[str]], Any], query: str, texts: list[str]) -> list[float]:
    import numpy as np
    vectors = embed([query] + texts)
    return (vectors[1:] @ vectors[0]).astype(float).tolist() if len(np.shape(vectors)) == 2 else []


def make_embedder(loaded) -> Callable[[list[str]], Any]:
    import torch

    def embed(texts: list[str]):
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 32):
                enc = loaded.tokenizer(texts[i:i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt")
                hidden = loaded.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                out.append(torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1).numpy())
        import numpy as np
        return np.concatenate(out)
    return embed


def extractive_answer(embed, question: str, passages: list[SourcePassage], n: int = 2) -> str:
    """Encoder-only models cannot write text; they can rank it. Pick the sentences closest to the question."""
    sentences = [(p.rank, s) for p in passages for s in split_sentences(p.text)]
    if not sentences:
        return ""
    scores = _cosine_rank(embed, question, [s for _, s in sentences])
    best = sorted(range(len(sentences)), key=lambda i: -scores[i])[:n]
    return " ".join(f"{sentences[i][1]} [{sentences[i][0]}]" for i in sorted(best))


def decoder_answer(loaded, question: str, context: str) -> tuple[str, int, int]:
    import torch
    tok = loaded.tokenizer
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}]
    chat = bool(getattr(tok, "chat_template", None))
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if chat else f"{SYSTEM}\n\nSources:\n{context}\n\nQuestion: {question}\nAnswer:"
    enc = tok(prompt, return_tensors="pt")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.inference_mode():
        out = loaded.model.generate(**enc, max_new_tokens=160, do_sample=False, repetition_penalty=1.1, pad_token_id=pad)
    n_in, n_out = int(enc["input_ids"].shape[-1]), int(out.shape[-1] - enc["input_ids"].shape[-1])
    text = tok.decode(out[0, n_in:], skip_special_tokens=True).strip()
    text = text if chat else re.split(r"\n\s*\n|\nQuestion:", text)[0].strip()  # base models keep inventing further turns
    return dedupe_repeats(text), n_in, n_out


def seq2seq_answer(loaded, question: str, context: str) -> tuple[str, int, int]:
    import torch
    tok = loaded.tokenizer
    ids = tok(context, add_special_tokens=False)["input_ids"][:380]  # keep the question inside the 512-token window
    prompt = f"Read the context and answer the question in one or two complete sentences.\n\nContext: {tok.decode(ids)}\n\nQuestion: {question}"
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.inference_mode():
        out = loaded.model.generate(**enc, max_new_tokens=96, do_sample=False, num_beams=4, min_new_tokens=10, no_repeat_ngram_size=3)
    return tok.decode(out[0], skip_special_tokens=True).strip(), int(enc["input_ids"].shape[-1]), int(out.shape[-1] - 1)


def pick_defaults(local: dict[str, list[str]]) -> dict[str, str]:
    return {kind: CompareService._default(kind, {k: local.get(k, []) for k in DEFAULTS}) for kind in DEFAULTS}


# --- judging -----------------------------------------------------------------------------
NAMES = {"encoder_only": "Encoder-only", "decoder_only": "Decoder-only", "encoder_decoder": "Encoder-decoder"}


def heuristic_judge(answers: list[ArchAnswer]) -> Verdict:
    """Transparent automatic checks. They measure support and relevance, not whether the answer is true."""
    scored: list[tuple[ArchAnswer, float, str]] = []
    for a in answers:
        m = a.metrics
        if a.error or not a.answer or m is None:
            scored.append((a, 0.0, "gave no usable answer" + (f" ({a.error})" if a.error else "")))
            continue
        if m.words < 3:
            scored.append((a, 20.0, f"gave a very short answer ({m.words} word{'s' if m.words != 1 else ''})"))
            continue
        if DECLINE.search(a.answer):
            scored.append((a, 30.0, "said the sources do not contain the answer"))
            continue
        g, r, rep = m.grounding or 0.0, max(m.relevance or 0.0, 0.0), m.repetition or 0.0
        score = 100 * (0.45 * g + 0.45 * min(r / 0.7, 1.0) + 0.10 * (1 - rep))
        parts = [f"{round(g * 100)}% of its key words appear in the sources"]
        if m.relevance is not None:
            parts.append(f"relevance to the question {r:.2f}")
        if rep > 0.25:
            parts.append("it repeats itself")
        scored.append((a, round(score, 1), ", ".join(parts)))
    per = [AnswerVerdict(architecture=a.architecture, score=s, comment=c[0].upper() + c[1:] + ".") for a, s, c in scored]
    ranked = sorted(scored, key=lambda t: -t[1])
    winner = ranked[0][0].architecture if ranked and ranked[0][1] > 0 else None
    if winner is None:
        summary = "None of the three produced a usable answer."
    elif len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 5:
        summary = f"{NAMES[winner]} and {NAMES[ranked[1][0].architecture]} are effectively tied on these checks, so there is no clear winner."
        winner = None
    else:
        summary = f"{NAMES[winner]} scores highest: {ranked[0][2]}."
    return Verdict(source="heuristic", winner=winner, summary=summary, per_answer=per, caveat=(
        "Automatic checks only. The encoder-only answer quotes sentences from the sources, so its support score is 100% by construction. "
        "None of this verifies that an answer is correct; switch on the Groq judge for a reasoned verdict."))


def groq_judge(question: str, sources: list[SourcePassage], answers: list[ArchAnswer], api_key: str, client: httpx.Client | None = None, model: str = GROQ_MODEL,
               kind: str = "qa") -> Verdict:
    """Blind comparison: the judge sees shuffled, anonymous answers and the sources, never the architecture names."""
    usable = [a for a in answers if a.answer]
    if len(usable) < 2:
        raise JudgeError("Need at least two answers to compare.")
    rng = random.Random(int(hashlib.sha256(question.encode()).hexdigest(), 16))
    shuffled = usable[:]
    rng.shuffle(shuffled)
    source_text = "\n\n".join(f"[{s.rank}] ({s.document}, {s.location}) {s.text[:900]}" for s in sources)
    answer_text = "\n\n".join(f"Answer {i + 1}: {a.answer}" for i, a in enumerate(shuffled))
    rubric = ("You are a strict, fair evaluator of question answering. Judge ONLY against the provided sources. "
              "For each answer give integer scores 1-5 for correctness, grounding (claims supported by the sources) and completeness, and a one-sentence comment. ") if kind == "qa" else (
              "You are a strict, fair evaluator of translation. The sources are the original text. For each translation give integer scores 1-5 for correctness "
              "(meaning preserved), grounding (nothing added that is not in the original) and completeness (everything translated), and a one-sentence comment. ")
    system = (rubric +
              "Then pick the best answer and explain why in 2-3 sentences. Reply with JSON only: "
              '{"answers":[{"id":1,"correctness":0,"grounding":0,"completeness":0,"comment":""}],"best_id":1,"reasoning":""}')
    body = {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": f"Question: {question}\n\nSources:\n{source_text}\n\n{answer_text}"}]}
    owns = client is None
    client = client or httpx.Client(timeout=45)
    try:
        response = client.post(GROQ_URL, headers={"Authorization": f"Bearer {api_key}"}, json=body)
    except httpx.HTTPError as exc:
        raise JudgeError(f"Could not reach Groq: {type(exc).__name__}") from exc
    finally:
        if owns:
            client.close()
    if response.status_code != 200:
        raise JudgeError(f"Groq returned {response.status_code}" + (" (check the API key)" if response.status_code in (401, 403) else ""))
    try:
        data = json.loads(response.json()["choices"][0]["message"]["content"])
        by_id = {int(x["id"]): x for x in data["answers"]}
        per = []
        for i, a in enumerate(shuffled, start=1):
            x = by_id[i]
            avg = (int(x["correctness"]) + int(x["grounding"]) + int(x["completeness"])) / 3
            per.append(AnswerVerdict(architecture=a.architecture, score=round(avg / 5 * 100, 1), comment=str(x["comment"])))
        winner = shuffled[int(data["best_id"]) - 1].architecture
        summary = str(data["reasoning"])
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        raise JudgeError("Groq returned an answer in an unexpected format.") from exc
    return Verdict(source="groq", judge_model=model, winner=winner, summary=summary, per_answer=per, caveat=(
        "Judged by a large language model that saw the question, the sources and the answers (with architecture names hidden and order shuffled). "
        "It can still be wrong or biased."))


# --- orchestration -----------------------------------------------------------------------
def answer_three_ways(request: AnswersRequest, rag: RagService, compare: CompareService, groq_key: str | None = None,
                      judge: Callable[..., Verdict] = groq_judge, fixed_sources: list[SourcePassage] | None = None) -> ThreeWayAnswer:
    if fixed_sources is not None:   # e.g. "the first slide": the exact text was located, no search needed
        sources = fixed_sources
    else:
        citations = rag.retrieve(request.question, request.top_k, documents=set(request.documents) or None)
        sources = [SourcePassage(rank=c.rank, document=c.document, location=c.location, text=clean_text(c.text), score=c.score) for c in citations]
    sources_text = "\n\n".join(s.text for s in sources)
    context = "\n\n".join(f"[{s.rank}] {s.text}" for s in sources)
    defaults = pick_defaults(compare.available_models()["local"])
    ids = {"encoder_only": request.encoder_model_id or defaults["encoder_only"], "decoder_only": request.decoder_model_id or defaults["decoder_only"],
           "encoder_decoder": request.seq2seq_model_id or defaults["encoder_decoder"]}
    methods = {"encoder_only": "Picks the sentences most similar to the question (an encoder can rank text but not write it)",
               "decoder_only": "Writes an answer one token at a time from the sources",
               "encoder_decoder": "Reads the sources with an encoder, then a decoder writes the answer"}
    answers: list[ArchAnswer] = []
    params: dict[str, int] = {}
    embed = None
    for arch in [a for a in ORDER if not request.architectures or a in request.architectures]:
        entry = ArchAnswer(architecture=arch, model_id=ids[arch], method=methods[arch])
        started = time.perf_counter()
        try:
            loaded = compare.get_model(arch, ids[arch], request.allow_model_download)
            entry.cold_start = loaded.load_ms > 0
            params[arch] = round(loaded.params_millions)
            if arch == "encoder_only":
                embed = make_embedder(loaded)
                entry.answer = extractive_answer(embed, request.question, sources)
            elif arch == "decoder_only":
                entry.answer, entry.input_tokens, entry.output_tokens = decoder_answer(loaded, request.question, context)
            else:
                entry.answer, entry.input_tokens, entry.output_tokens = seq2seq_answer(loaded, request.question, context)
        except CompareUnavailable as exc:
            entry.error = str(exc)
        entry.seconds = round(time.perf_counter() - started, 2)
        answers.append(entry)
    if embed is None and any(a.answer for a in answers):  # encoder not selected: still needed to score relevance
        try:
            embed = make_embedder(compare.get_model("encoder_only", ids["encoder_only"], request.allow_model_download))
        except CompareUnavailable:
            embed = None
    for a in answers:
        if a.answer:
            relevance = _cosine_rank(embed, request.question, [a.answer])[0] if embed else None
            a.metrics = Metrics(grounding=grounding(a.answer, sources_text), relevance=round(relevance, 3) if relevance is not None else None,
                                repetition=repetition(a.answer), words=len(a.answer.split()))
    verdicts = [heuristic_judge(answers)] if len(answers) > 1 else []
    notes = ["All three answers are built from the same retrieved sources, so differences come from the model, not from what it was shown."]
    if len(params) > 1:
        sizes = " vs ".join(f"{NAMES[a]} {params[a]}M" for a in ORDER if a in params)
        notes.append(f"This compares three specific models ({sizes} parameters) that also differ in training data, so the result reflects model size and training as well as architecture. It is not a controlled test of the architectures themselves.")
    if request.use_groq and len(answers) > 1:
        if not groq_key:
            notes.append("Groq judge skipped: no API key is configured.")
        else:
            try:
                verdicts.insert(0, judge(request.question, sources, answers, groq_key))
            except JudgeError as exc:
                notes.append(f"Groq judge failed: {exc}. Showing the automatic checks only.")
    return ThreeWayAnswer(question=request.question, sources=sources, answers=answers, verdicts=verdicts, notes=notes)
