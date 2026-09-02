from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model_and_tokenizer(cfg: dict, adapter_path: str | None = None):
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["id"],
        revision=model_cfg.get("revision", "main"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "revision": model_cfg.get("revision", "main"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "torch_dtype": resolve_dtype(model_cfg.get("dtype", "bfloat16")),
        "device_map": "auto",
    }
    attn = model_cfg.get("attn_implementation")
    if attn:
        kwargs["attn_implementation"] = attn

    model = AutoModelForCausalLM.from_pretrained(model_cfg["id"], **kwargs)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def get_transformer_layers(model):
    base = getattr(model, "base_model", model)
    # PEFT wraps the model more than once. Walk through common paths.
    candidates = [
        getattr(getattr(base, "model", None), "layers", None),
        getattr(getattr(getattr(base, "model", None), "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    raise AttributeError("Could not locate transformer layers on this model architecture.")


def render_instruction(tokenizer, system_prompt: str, instruction: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@contextmanager
def layer_intervention(model, layer_index: int, transform: Callable[[torch.Tensor], torch.Tensor] | None):
    if transform is None:
        yield
        return

    layers = get_transformer_layers(model)
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(f"layer_index={layer_index} but model has {len(layers)} layers")

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            changed = transform(hidden)
            return (changed, *output[1:])
        return transform(output)

    handle = layers[layer_index].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
