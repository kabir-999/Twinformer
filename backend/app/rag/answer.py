from __future__ import annotations

import re
import time

from app.benchmark.engine import BenchmarkUnavailable, engine
from app.schemas.workloads import Architecture, BenchmarkRequest, Citation, WorkloadType

SYSTEM = ("You answer questions using only the numbered passages provided. Cite the passages you use like [1] or [2]. "
          "If the passages do not contain the answer, say that they do not. Be concise.")


def generate_answer(question: str, citations: list[Citation], model_id: str | None, allow_download: bool, max_new_tokens: int = 220) -> tuple[str, str, float]:
    import torch
    probe = BenchmarkRequest(architecture=Architecture.DECODER, workload_type=WorkloadType.AUTOREGRESSIVE_GENERATION, model_id=model_id,
                             input_tokens=1, allow_model_download=allow_download)
    loaded = engine.registry.load(probe)
    context = "\n\n".join(f"[{c.rank}] ({c.document}, {c.location})\n{c.text}" for c in citations)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"}]
    tokenizer = loaded.tokenizer
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        raise BenchmarkUnavailable(f"{probe.resolved_model_id} has no chat template; pick an instruct/chat model for written answers.")
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000).to(loaded.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = loaded.model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=6, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True)
    text = re.split(r"\n\s*(?:Human|User|Assistant|Question)\s*:", text)[0].strip()  # small base models sometimes invent further turns
    return text, probe.resolved_model_id, round(time.perf_counter() - started, 1)
