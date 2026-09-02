from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from datasets import load_from_disk

from .activations import extract_last_prompt_activations
from .config import load_config, project_path
from .modeling import load_model_and_tokenizer
from .utils import set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", required=True, choices=["discovery", "validation", "test"])
    parser.add_argument("--adapter-path", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    ds = load_from_disk(str(project_path(cfg["dataset"]["processed_dir"])))[args.split]
    rows = [dict(row) for row in ds]

    model, tokenizer = load_model_and_tokenizer(cfg, adapter_path=args.adapter_path)
    acts = extract_last_prompt_activations(
        model,
        tokenizer,
        rows,
        system_prompt=cfg["model"]["system_prompt"],
        max_length=int(cfg["model"]["max_input_length"]),
        batch_size=int(cfg["activations"]["batch_size"]),
    )

    dtype = np.float16 if cfg["activations"].get("storage_dtype", "float16") == "float16" else np.float32
    acts = acts.astype(dtype)
    outdir = project_path(cfg["activations"]["directory"])
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "base" if not args.adapter_path else "adapter"
    out = outdir / f"{args.split}_{suffix}.npz"
    np.savez_compressed(
        out,
        activations=acts,
        harmfulness=np.asarray(ds["harmfulness"], dtype="U16"),
        style=np.asarray(ds["style"], dtype="U16"),
        label_4way=np.asarray(ds["label_4way"], dtype="U32"),
        source_id=np.asarray(ds["source_id"], dtype="U64"),
        leakage_group_id=np.asarray(ds["leakage_group_id"], dtype="U64"),
        prompt_hash=np.asarray(ds["prompt_hash"], dtype="U64"),
    )
    meta = {
        "split": args.split,
        "rows": len(ds),
        "shape": list(acts.shape),
        "dtype": str(acts.dtype),
        "adapter_path": args.adapter_path,
        "raw_prompts_saved": False,
    }
    write_json(out.with_suffix(".json"), meta)
    print(f"Saved {acts.shape} activations to {out}")
    print("Raw prompt text saved: NO")


if __name__ == "__main__":
    main()
