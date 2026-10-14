from __future__ import annotations

import json
import os
from pathlib import Path


def _hub_dir() -> Path:
    if os.getenv("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _kind(architectures: list[str]) -> str | None:
    name = (architectures or [""])[0]
    if any(t in name for t in ("Vision", "VL", "Llava", "Blip", "Whisper", "Vit", "DPT", "Sam")):
        return None
    if "CausalLM" in name or "LMHeadModel" in name:
        return "decoder_only"
    if name.endswith("ForConditionalGeneration") and any(t in name for t in ("T5", "Bart", "Pegasus", "Marian")):
        return "encoder_decoder"
    if name in {"BertModel", "DistilBertModel", "RobertaModel", "BertForMaskedLM", "DistilBertForMaskedLM", "RobertaForMaskedLM", "XLMRobertaModel"}:
        return "encoder_only"
    return None


def _has_weights(snapshot: Path) -> bool:
    """True when the snapshot holds loadable weights (a partial download only has config/tokenizer files)."""
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        try:
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        except (OSError, ValueError):
            return False
        return bool(shards) and all((snapshot / shard).exists() for shard in shards)  # exists() follows the symlink into blobs/
    return any((snapshot / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))


def list_local_models() -> list[dict[str, str]]:
    """Models already in the Hugging Face cache that the benchmark engine can load, without downloading anything."""
    found = []
    for entry in sorted(_hub_dir().glob("models--*")):
        configs = [c for c in sorted(entry.glob("snapshots/*/config.json")) if _has_weights(c.parent)]
        if not configs:
            continue
        if any(t in entry.name.lower() for t in ("-vl", "vision", "llava", "blip", "sam-", "midas", "ocr")):
            continue
        try:
            kind = _kind(json.loads(configs[-1].read_text()).get("architectures") or [])
        except (OSError, ValueError):
            continue
        if kind:
            found.append({"id": entry.name[len("models--"):].replace("--", "/"), "kind": kind})
    return found


def is_cached(model_id: str) -> bool:
    """True when the model's weights are already in the local Hugging Face cache."""
    entry = _hub_dir() / ("models--" + model_id.replace("/", "--"))
    return any(_has_weights(config.parent) for config in entry.glob("snapshots/*/config.json"))
