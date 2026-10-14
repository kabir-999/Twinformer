"""Translation with dedicated encoder-decoder models (MarianMT), plus a decoder-only fallback prompt."""
from __future__ import annotations

import re

NLLB = "facebook/nllb-200-distilled-600M"
# language name -> (short code, NLLB code). NLLB is one encoder-decoder model that translates English into all of these.
LANGUAGES = {
    "hindi": ("hi", "hin_Deva"), "french": ("fr", "fra_Latn"), "german": ("de", "deu_Latn"), "spanish": ("es", "spa_Latn"), "romanian": ("ro", "ron_Latn"),
    "italian": ("it", "ita_Latn"), "russian": ("ru", "rus_Cyrl"), "chinese": ("zh", "zho_Hans"), "arabic": ("ar", "arb_Arab"), "tamil": ("ta", "tam_Taml"),
    "telugu": ("te", "tel_Telu"), "bengali": ("bn", "ben_Beng"), "marathi": ("mr", "mar_Deva"), "gujarati": ("gu", "guj_Gujr"), "urdu": ("ur", "urd_Arab"),
    "japanese": ("ja", "jpn_Jpan"), "korean": ("ko", "kor_Hang"), "portuguese": ("pt", "por_Latn"), "dutch": ("nl", "nld_Latn"), "turkish": ("tr", "tur_Latn"),
    "punjabi": ("pa", "pan_Guru"),
}
NLLB_CODES = {short: nllb for short, nllb in LANGUAGES.values()}
MARIAN = {short: f"Helsinki-NLP/opus-mt-en-{short}" for short in ("hi", "fr", "de", "es", "ro", "it", "ru", "zh", "ar")}   # smaller per-language fallback


def target_language(question: str) -> tuple[str, str] | None:
    """(short code, language name) for the language the user asked for, or None if no known language was named."""
    q = question.lower()
    for name, (short, _) in LANGUAGES.items():
        if re.search(rf"\b{name}\b", q):
            return short, name.capitalize()
    return None


def pick_translator(code: str, allow_download: bool, cached=None) -> str:
    """NLLB when it is on this computer (or may be downloaded); otherwise the small per-language Marian model if it is cached."""
    from app.models.local import is_cached
    cached = cached or is_cached
    if cached(NLLB) or allow_download:
        return NLLB
    if code in MARIAN and cached(MARIAN[code]):
        return MARIAN[code]
    return NLLB


def normalize_caps(piece: str) -> str:
    """Translation models handle SHOUTING badly: turn long ALL-CAPS words into normal case, leave codes and short acronyms alone."""
    return " ".join(w.capitalize() if w.isalpha() and w.isupper() and len(w) > 3 else w for w in piece.split())


def split_for_translation(text: str, limit: int = 300) -> list[str]:
    """One translatable piece per line / sentence / '·' bullet, so slide fragments are not glued into one confusing run."""
    pieces: list[str] = []
    for line in text.split("\n"):
        for sentence in re.split(r"(?<=[.!?])\s+|\s[·•|]\s", line):
            sentence = normalize_caps(sentence.strip(" ·•|"))
            if not sentence:
                continue
            while len(sentence) > limit:   # very long sentence: break at a comma/space near the limit
                cut = max(sentence.rfind(", ", 0, limit), sentence.rfind(" ", 0, limit))
                cut = cut if cut > 40 else limit
                pieces.append(sentence[:cut].strip(" ,"))
                sentence = sentence[cut:].strip(" ,")
            pieces.append(sentence)
    return pieces


def _generate_batches(loaded, pieces: list[str], **kwargs) -> list[str]:
    import torch
    out: list[str] = []
    for i in range(0, len(pieces), 8):
        enc = loaded.tokenizer(pieces[i:i + 8], return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.inference_mode():
            ids = loaded.model.generate(**enc, num_beams=4, max_new_tokens=256, **kwargs)
        out += [loaded.tokenizer.decode(row, skip_special_tokens=True) for row in ids]
    return out


def marian_translate(loaded, text: str) -> str:
    return "\n".join(_generate_batches(loaded, split_for_translation(text)))


def nllb_translate(loaded, text: str, code: str) -> str:
    loaded.tokenizer.src_lang = "eng_Latn"
    forced = loaded.tokenizer.convert_tokens_to_ids(NLLB_CODES[code])
    return "\n".join(_generate_batches(loaded, split_for_translation(text), forced_bos_token_id=forced))


def translate_with(loaded, model_id: str, text: str, code: str) -> str:
    return nllb_translate(loaded, text, code) if "nllb" in model_id.lower() else marian_translate(loaded, text)


def decoder_translate(loaded, text: str, language: str) -> str:
    import torch
    tok = loaded.tokenizer
    messages = [{"role": "system", "content": f"You are a professional translator. Translate the user's text into {language}. Output only the translation."},
                {"role": "user", "content": text}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if getattr(tok, "chat_template", None) else f"Translate to {language}:\n{text}\n{language}:"
    enc = tok(prompt, return_tensors="pt")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.inference_mode():
        out = loaded.model.generate(**enc, max_new_tokens=400, do_sample=False, pad_token_id=pad)
    return tok.decode(out[0, enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
