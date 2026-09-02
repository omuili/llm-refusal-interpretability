from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np
import torch

from .modeling import layer_intervention, render_instruction
from .refusal import is_refusal


def generate_refusal_labels(
    model,
    tokenizer,
    rows: list[dict],
    system_prompt: str,
    max_input_length: int,
    max_new_tokens: int,
    batch_size: int,
    layer_index: int | None = None,
    transform: Callable | None = None,
):
    tokenizer.padding_side = "left"
    outputs = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [render_instruction(tokenizer, system_prompt, r["instruction"]) for r in batch]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        prompt_len = encoded["input_ids"].shape[1]

        with layer_intervention(model, layer_index or 0, transform if layer_index is not None else None):
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        for i, row in enumerate(batch):
            continuation_ids = generated[i, prompt_len:]
            text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
            outputs.append({
                "source_id": row["source_id"],
                "prompt_hash": row["prompt_hash"],
                "harmfulness": row["harmfulness"],
                "style": row["style"],
                "label_4way": row["label_4way"],
                "refusal": bool(is_refusal(text)),
                "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "generated_tokens": int(continuation_ids.numel()),
            })
    return outputs


def select_balanced_harmfulness_rows(ds, per_class: int, seed: int):
    rows = [dict(r) for r in ds]
    rng = np.random.default_rng(seed)
    selected = []
    for label in ("benign", "harmful"):
        idx = [i for i, r in enumerate(rows) if r["harmfulness"] == label]
        if len(idx) < per_class:
            raise RuntimeError(f"Need {per_class} {label} examples, found {len(idx)}")
        chosen = rng.choice(idx, size=per_class, replace=False)
        selected.extend(rows[int(i)] for i in chosen)
    selected.sort(key=lambda r: hashlib.sha256(f"{seed}|{r['source_id']}|{r['prompt_hash']}".encode()).hexdigest())
    return selected
