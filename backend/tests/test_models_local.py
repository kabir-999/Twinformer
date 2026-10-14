import json

from app.models.local import list_local_models


def test_lists_only_loadable_architectures(tmp_path, monkeypatch):
    def model(name, arch, weights=True):
        d = tmp_path / f"models--{name}" / "snapshots" / "abc"
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"architectures": [arch]}))
        if weights:
            (d / "model.safetensors").write_bytes(b"x")
    model("Qwen--Qwen1.5-0.5B", "Qwen2ForCausalLM")
    model("google--flan-t5-small", "T5ForConditionalGeneration")
    model("distilbert--distilbert-base-uncased", "DistilBertForMaskedLM")
    model("Qwen--Partial", "Qwen2ForCausalLM", weights=False)  # config only: interrupted download
    model("Salesforce--blip-base", "BlipForConditionalGeneration")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    kinds = {m["id"]: m["kind"] for m in list_local_models()}
    assert kinds == {"Qwen/Qwen1.5-0.5B": "decoder_only", "google/flan-t5-small": "encoder_decoder", "distilbert/distilbert-base-uncased": "encoder_only"}
