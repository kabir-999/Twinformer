from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.models.local import list_local_models

Task = Literal["summarize", "answer", "continue", "translate_fr"]
MAX_INPUT_TOKENS = 384
MAX_HEATMAP_TOKENS = 32
DEFAULT_DOWNLOADS = {"encoder_only": "sentence-transformers/all-MiniLM-L6-v2", "decoder_only": "Qwen/Qwen2.5-0.5B-Instruct", "encoder_decoder": "google/flan-t5-base"}
PREFERRED = {"encoder_only": ["sentence-transformers/all-MiniLM-L6-v2"], "decoder_only": [], "encoder_decoder": ["google/flan-t5-large", "google/flan-t5-base", "google/flan-t5-small"]}


class CompareRequest(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    question: str | None = Field(default=None, max_length=500)
    task: Task = "summarize"
    encoder_model_id: str | None = None
    decoder_model_id: str | None = None
    seq2seq_model_id: str | None = None
    max_new_tokens: int = Field(default=60, ge=4, le=128)
    allow_model_download: bool = False


class Timings(BaseModel):
    tokenize_ms: float
    encode_ms: float | None = None     # encoder forward, or decoder prefill
    decode_ms: float | None = None     # generation after the encode/prefill step
    total_ms: float


class Heatmap(BaseModel):
    kind: Literal["self_bidirectional", "self_causal", "cross"]
    rows: list[str]
    cols: list[str]
    row_index: list[int] = Field(default_factory=list)   # position of each shown row/column in the full token list
    col_index: list[int] = Field(default_factory=list)
    values: list[list[float]]
    layer: int
    truncated: bool = False


class TokenStep(BaseModel):
    step: int
    chosen: str
    chosen_prob: float
    alternatives: list[dict[str, float | str]]


class RelatedPassage(BaseModel):
    document: str
    location: str
    text: str
    score: float


class ModelRun(BaseModel):
    architecture: Literal["encoder_only", "decoder_only", "encoder_decoder"]
    status: Literal["ok", "unavailable"] = "ok"
    error: str | None = None
    model_id: str
    params_millions: float | None = None
    weights_mb: float | None = None
    load_ms: float | None = None
    prompt: str | None = None
    input_tokens: list[str] = Field(default_factory=list)
    input_token_count: int = 0
    input_truncated: bool = False
    output_text: str | None = None
    output_token_count: int = 0
    tokens_per_second: float | None = None
    timings: Timings | None = None
    embedding_dim: int | None = None
    embedding_preview: list[float] = Field(default_factory=list)
    question_similarity: float | None = None
    attention: Heatmap | None = None
    token_steps: list[TokenStep] = Field(default_factory=list)
    related_passages: list[RelatedPassage] = Field(default_factory=list)


class CompareResult(BaseModel):
    id: str
    created_at: str
    kind: Literal["compare"] = "compare"
    request: CompareRequest
    runs: list[ModelRun]
    notes: list[str]


class CompareUnavailable(RuntimeError):
    pass


# --- pure helpers (unit-tested) ----------------------------------------------------------
def build_prompt(architecture: str, task: Task, text: str, question: str | None = None) -> str:
    """Each family gets the prompt style it was trained on; the encoder just reads the raw text."""
    text = text.strip()
    if architecture == "encoder_only":
        return text
    if task == "answer" and question:
        if architecture == "encoder_decoder":
            return f"Read the context and answer the question in one or two complete sentences.\n\nContext: {text}\n\nQuestion: {question}"
        return f"Context: {text}\n\nQuestion: {question}\nAnswer in one or two sentences using only the context:"
    if architecture == "encoder_decoder":
        return {"summarize": f"Summarize: {text}", "answer": f"Answer the question: {text}", "continue": f"Continue the text: {text}", "translate_fr": f"Translate English to French: {text}"}[task]
    return {"summarize": f"Text: {text}\n\nOne-sentence summary of the text:", "answer": f"Question: {text}\nAnswer:", "continue": text,
            "translate_fr": f"English: {text}\nFrench:"}[task]


def pretty_token(token: str) -> str:
    return token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\\n").replace("\n", "\\n")


def tidy_output(text: str, task: Task) -> str:
    """Base decoder models keep going after the answer (often inventing the next question); keep the first line unless continuing."""
    text = text.strip()
    return text if task == "continue" else text.split("\n", 1)[0].strip()


def heatmap(kind: str, rows: list[str], cols: list[str], matrix, layer: int) -> Heatmap:
    """matrix: (rows x cols) attention weights averaged over heads. Long inputs are cut to the tokens that receive the most
    attention (kept in original order), so the picture shows what matters rather than just the first words."""
    import numpy as np
    m = np.asarray(matrix, dtype=float)
    r_idx, c_idx = list(range(len(rows))), list(range(len(cols)))
    truncated = False
    if kind == "cross":
        if len(cols) > MAX_HEATMAP_TOKENS:
            c_idx, truncated = sorted(np.argsort(-m.sum(0))[:MAX_HEATMAP_TOKENS].tolist()), True
        if len(rows) > MAX_HEATMAP_TOKENS:
            r_idx, truncated = r_idx[:MAX_HEATMAP_TOKENS], True
    elif len(cols) > MAX_HEATMAP_TOKENS:
        c_idx = sorted(np.argsort(-m.sum(0))[:MAX_HEATMAP_TOKENS].tolist())
        r_idx, truncated = c_idx, True
    values = [[round(float(v), 4) for v in row] for row in m[np.ix_(r_idx, c_idx)]]
    return Heatmap(kind=kind, rows=[rows[i] for i in r_idx], cols=[cols[i] for i in c_idx], row_index=r_idx, col_index=c_idx, values=values, layer=layer, truncated=truncated)


def top_alternatives(probs, tokenizer, chosen_id: int, k: int = 5) -> TokenStep:
    import torch
    values, ids = torch.topk(probs, k)
    return TokenStep(step=0, chosen=pretty_token(tokenizer.convert_ids_to_tokens(int(chosen_id))), chosen_prob=round(float(probs[chosen_id]), 4),
                     alternatives=[{"token": pretty_token(tokenizer.convert_ids_to_tokens(int(i))), "prob": round(float(v), 4)} for v, i in zip(values, ids)])


# --- model handling ----------------------------------------------------------------------
@dataclass
class Loaded:
    model: Any
    tokenizer: Any
    load_ms: float
    params_millions: float
    weights_mb: float
    warmed: bool = False


class CompareService:
    def __init__(self, retriever: Callable[[str, int], list[RelatedPassage]] | None = None, retriever_model: Callable[[], str | None] | None = None) -> None:
        self._cache: dict[tuple[str, str], Loaded] = {}
        self._retriever, self._retriever_model = retriever, retriever_model

    def available_models(self) -> dict[str, Any]:
        local = list_local_models()
        grouped = {kind: [m["id"] for m in local if m["kind"] == kind] for kind in DEFAULT_DOWNLOADS}
        missing = {kind: model for kind, model in DEFAULT_DOWNLOADS.items() if model not in grouped[kind]}
        return {"local": grouped, "defaults": {kind: self._default(kind, grouped) for kind in DEFAULT_DOWNLOADS}, "missing_defaults": missing}

    @staticmethod
    def _default(kind: str, grouped: dict[str, list[str]]) -> str:
        ids = grouped[kind]
        if kind == "decoder_only":
            chats = [m for m in ids if re.search(r"instruct|chat", m, re.I)]
            if chats:
                def size(m: str) -> float:
                    found = re.search(r"(\d+(?:\.\d+)?)\s*b\b", m, re.I)
                    return float(found.group(1)) if found else 0.0
                usable = [m for m in chats if size(m) <= 4] or chats   # larger is better, but keep it runnable on a laptop CPU
                return max(usable, key=size)
        preferred = next((m for m in PREFERRED[kind] if m in ids), None)
        return preferred or (ids[0] if ids else DEFAULT_DOWNLOADS[kind])

    def _load(self, architecture: str, model_id: str, allow_download: bool) -> Loaded:
        key = (architecture, model_id)
        if key in self._cache:
            cached = self._cache[key]
            return Loaded(cached.model, cached.tokenizer, 0.0, cached.params_millions, cached.weights_mb, cached.warmed)
        try:
            from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise CompareUnavailable("Install backend/requirements.txt (torch, transformers).") from exc
        auto = {"encoder_only": AutoModel, "decoder_only": AutoModelForCausalLM, "encoder_decoder": AutoModelForSeq2SeqLM}[architecture]
        started = time.perf_counter()
        local_only = not allow_download
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
            # eager attention is required to read attention weights back out
            model = auto.from_pretrained(model_id, local_files_only=local_only, attn_implementation="eager").eval()
        except Exception as exc:
            hint = "Tick “Allow downloading” to fetch it." if local_only else "Check the model name and your connection."
            raise CompareUnavailable(f"{model_id} is not available locally. {hint} ({type(exc).__name__})") from exc
        params = sum(p.numel() for p in model.parameters())
        weights = sum(p.numel() * p.element_size() for p in model.parameters())
        loaded = Loaded(model, tokenizer, (time.perf_counter() - started) * 1000, params / 1e6, weights / 1024 ** 2)
        self._cache[key] = loaded
        return loaded

    # --- running ---------------------------------------------------------------------
    def run(self, request: CompareRequest) -> CompareResult:
        models = self.available_models()["defaults"]
        plan = [("encoder_only", request.encoder_model_id or models["encoder_only"], self._run_encoder),
                ("decoder_only", request.decoder_model_id or models["decoder_only"], self._run_decoder),
                ("encoder_decoder", request.seq2seq_model_id or models["encoder_decoder"], self._run_seq2seq)]
        runs = []
        for architecture, model_id, runner in plan:
            try:
                loaded = self._load(architecture, model_id, request.allow_model_download)
                runs.append(runner(loaded, model_id, request))
                self._cache[(architecture, model_id)].warmed = True
            except CompareUnavailable as exc:
                runs.append(ModelRun(architecture=architecture, model_id=model_id, status="unavailable", error=str(exc)))
        notes = [
            "The three models differ in size and training data as well as architecture, so speed and quality differences reflect both. Parameter counts are shown for context.",
            "Attention weights are the last layer averaged over heads. They show where the model looks, not a complete explanation of its output.",
            "Models run on the CPU with eager attention so weights can be read; timings are the median of 3 runs after a warm-up, except generation, which is one timed run.",
            "Generation is greedy (always the most likely next token), so the same input always gives the same output.",
        ]
        return CompareResult(id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(), request=request, runs=runs, notes=notes)

    @staticmethod
    def _encode(loaded: Loaded, prompt: str, device_tensors: bool = True):
        started = time.perf_counter()
        enc = loaded.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
        return enc, (time.perf_counter() - started) * 1000

    @staticmethod
    def _median_ms(fn: Callable[[], Any], repeats: int = 3) -> float:
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            fn()
            times.append((time.perf_counter() - started) * 1000)
        return statistics.median(times)

    def _base(self, architecture, model_id, loaded: Loaded, prompt: str, enc, raw_prompt_tokens: int) -> dict:
        ids = enc["input_ids"][0]
        return dict(architecture=architecture, model_id=model_id, params_millions=round(loaded.params_millions, 1), weights_mb=round(loaded.weights_mb, 1),
                    load_ms=round(loaded.load_ms, 1) if loaded.load_ms else None, prompt=prompt,
                    input_tokens=[pretty_token(t) for t in loaded.tokenizer.convert_ids_to_tokens(ids)], input_token_count=int(ids.shape[0]),
                    input_truncated=raw_prompt_tokens > MAX_INPUT_TOKENS)

    def _raw_token_count(self, loaded: Loaded, prompt: str) -> int:
        return len(loaded.tokenizer(prompt, add_special_tokens=True)["input_ids"])

    def _run_encoder(self, loaded: Loaded, model_id: str, request: CompareRequest) -> ModelRun:
        import torch
        prompt = build_prompt("encoder_only", request.task, request.text, request.question)
        enc, tokenize_ms = self._encode(loaded, prompt)
        with torch.inference_mode():
            loaded.model(**enc)  # warm-up
            encode_ms = self._median_ms(lambda: loaded.model(**enc))
            out = loaded.model(**enc, output_attentions=True)
        mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
        vector = torch.nn.functional.normalize((out.last_hidden_state * mask).sum(1) / mask.sum(1), dim=-1)[0]
        tokens = [pretty_token(t) for t in loaded.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])]
        similarity = None
        if request.question:
            q_enc = loaded.tokenizer(request.question, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
            with torch.inference_mode():
                q_out = loaded.model(**q_enc)
            q_mask = q_enc["attention_mask"].unsqueeze(-1).to(q_out.last_hidden_state.dtype)
            q_vec = torch.nn.functional.normalize((q_out.last_hidden_state * q_mask).sum(1) / q_mask.sum(1), dim=-1)[0]
            similarity = round(float(q_vec @ vector), 3)
        related: list[RelatedPassage] = []
        if self._retriever and self._retriever_model and self._retriever_model() == model_id:
            related = self._retriever(prompt, 3)
        return ModelRun(**self._base("encoder_only", model_id, loaded, prompt, enc, self._raw_token_count(loaded, prompt)),
                        timings=Timings(tokenize_ms=round(tokenize_ms, 2), encode_ms=round(encode_ms, 2), total_ms=round(tokenize_ms + encode_ms, 2)),
                        question_similarity=similarity, embedding_dim=int(vector.shape[0]), embedding_preview=[round(float(v), 4) for v in vector[:48]],
                        attention=heatmap("self_bidirectional", tokens, tokens, out.attentions[-1][0].mean(0).tolist(), len(out.attentions) - 1), related_passages=related)

    def _steps(self, scores, sequences, tokenizer, start: int) -> list[TokenStep]:
        import torch
        steps = []
        for i, score in enumerate(scores):
            step = top_alternatives(torch.softmax(score[0].float(), dim=-1), tokenizer, int(sequences[0][start + i]))
            step.step = i + 1
            steps.append(step)
        return steps

    def _run_decoder(self, loaded: Loaded, model_id: str, request: CompareRequest) -> ModelRun:
        import torch
        prompt = build_prompt("decoder_only", request.task, request.text, request.question)
        enc, tokenize_ms = self._encode(loaded, prompt)
        pad = loaded.tokenizer.pad_token_id if loaded.tokenizer.pad_token_id is not None else loaded.tokenizer.eos_token_id
        generate = lambda **kw: loaded.model.generate(**enc, max_new_tokens=request.max_new_tokens, do_sample=False, pad_token_id=pad, **kw)
        with torch.inference_mode():
            if not loaded.warmed:
                loaded.model.generate(**enc, max_new_tokens=2, do_sample=False, pad_token_id=pad)
            prefill_ms = self._median_ms(lambda: loaded.model(**enc))
            started = time.perf_counter()
            sequence = generate()
            total_ms = (time.perf_counter() - started) * 1000
            prompt_len = enc["input_ids"].shape[1]
            attn = loaded.model(**enc, output_attentions=True).attentions
            detail = generate(return_dict_in_generate=True, output_scores=True)
        new_ids = sequence[0, prompt_len:]
        text = tidy_output(loaded.tokenizer.decode(new_ids, skip_special_tokens=True), request.task)
        n = int(new_ids.shape[0])
        tokens = [pretty_token(t) for t in loaded.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])]
        return ModelRun(**self._base("decoder_only", model_id, loaded, prompt, enc, self._raw_token_count(loaded, prompt)),
                        output_text=text, output_token_count=n, tokens_per_second=round(n / (total_ms / 1000), 1) if n and total_ms else None,
                        timings=Timings(tokenize_ms=round(tokenize_ms, 2), encode_ms=round(prefill_ms, 2), decode_ms=round(max(total_ms - prefill_ms, 0), 2), total_ms=round(tokenize_ms + total_ms, 2)),
                        attention=heatmap("self_causal", tokens, tokens, attn[-1][0].mean(0).tolist(), len(attn) - 1),
                        token_steps=self._steps(detail.scores, detail.sequences, loaded.tokenizer, prompt_len)[:16])

    def _run_seq2seq(self, loaded: Loaded, model_id: str, request: CompareRequest) -> ModelRun:
        import torch
        prompt = build_prompt("encoder_decoder", request.task, request.text, request.question)
        enc, tokenize_ms = self._encode(loaded, prompt)
        generate = lambda **kw: loaded.model.generate(**enc, max_new_tokens=request.max_new_tokens, do_sample=False, **kw)
        with torch.inference_mode():
            if not loaded.warmed:
                loaded.model.generate(**enc, max_new_tokens=2, do_sample=False)
            encoder = loaded.model.get_encoder()
            encode_ms = self._median_ms(lambda: encoder(**enc))
            started = time.perf_counter()
            sequence = generate()
            total_ms = (time.perf_counter() - started) * 1000
            detail = generate(return_dict_in_generate=True, output_scores=True, output_attentions=True)
        new_ids = sequence[0, 1:]  # drop the decoder start token
        text = loaded.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        n = int(new_ids.shape[0])
        src = [pretty_token(t) for t in loaded.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])]
        gen = [pretty_token(t) for t in loaded.tokenizer.convert_ids_to_tokens(new_ids)]
        cross = [step[-1][0].mean(0)[0].tolist() for step in detail.cross_attentions]  # last layer, mean over heads, one row per generated token
        return ModelRun(**self._base("encoder_decoder", model_id, loaded, prompt, enc, self._raw_token_count(loaded, prompt)),
                        output_text=text, output_token_count=n, tokens_per_second=round(n / (total_ms / 1000), 1) if n and total_ms else None,
                        timings=Timings(tokenize_ms=round(tokenize_ms, 2), encode_ms=round(encode_ms, 2), decode_ms=round(max(total_ms - encode_ms, 0), 2), total_ms=round(tokenize_ms + total_ms, 2)),
                        attention=heatmap("cross", gen, src, cross, len(detail.cross_attentions[0]) - 1),
                        token_steps=self._steps(detail.scores, detail.sequences, loaded.tokenizer, 1)[:16])


def _get_model(self: CompareService, architecture: str, model_id: str, allow_download: bool) -> Loaded:
    """Public access to the cached model loader (used by the document Q&A comparison)."""
    return self._load(architecture, model_id, allow_download)


CompareService.get_model = _get_model  # type: ignore[attr-defined]
