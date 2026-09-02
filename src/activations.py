from __future__ import annotations

import numpy as np
import torch

from .modeling import render_instruction


def extract_last_prompt_activations(model, tokenizer, rows: list[dict], system_prompt: str, max_length: int, batch_size: int):
    tokenizer.padding_side = "right"
    all_acts = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [render_instruction(tokenizer, system_prompt, row["instruction"]) for row in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        device = next(model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            outputs = model(**enc, output_hidden_states=True, use_cache=False)

        hidden_states = outputs.hidden_states[1:]  # block outputs, excluding embedding state
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        batch_acts = []
        for layer_h in hidden_states:
            picked = layer_h[torch.arange(layer_h.shape[0], device=layer_h.device), last_idx]
            batch_acts.append(picked.float().cpu().numpy())
        # [batch, layers, hidden]
        all_acts.append(np.stack(batch_acts, axis=1))

    return np.concatenate(all_acts, axis=0)
