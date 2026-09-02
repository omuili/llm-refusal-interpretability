from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from datasets import load_from_disk

from .causal import generate_refusal_labels, select_balanced_harmfulness_rows
from .config import load_config, project_path
from .interventions import center_projection_transform, zero_projection_transform, addition_transform
from .modeling import load_model_and_tokenizer
from .utils import bootstrap_rate_difference_ci, read_json, set_seed, write_json


def refusal_rate(rows):
    return float(np.mean([r["refusal"] for r in rows])) if rows else float("nan")


def only(rows, harmfulness: str):
    return [r for r in rows if r["harmfulness"] == harmfulness]


def add_condition(rows, condition: str):
    return [{**row, "condition": condition} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--suffix", default="base", choices=["base", "adapter"])
    parser.add_argument("--adapter-path", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    result_dir = project_path(cfg["results"]["dir"])
    direction_meta = read_json(result_dir / f"direction_summary_{args.suffix}.json")
    direction_npz = np.load(
        project_path("outputs/directions") / f"direction_{args.suffix}.npz",
        allow_pickle=False,
    )
    layer = int(direction_meta["selected_layer"])
    direction = direction_npz["directions"][layer].astype(np.float32)
    center = float(direction_npz["centers"][layer])
    gap = float(direction_npz["gaps"][layer])

    ds = load_from_disk(str(project_path(cfg["dataset"]["processed_dir"]))) ["test"]
    per_class = int(cfg["causal"]["examples_per_harmfulness_class"])
    rows = select_balanced_harmfulness_rows(ds, per_class=per_class, seed=seed + 101)
    harmful_rows = only(rows, "harmful")
    benign_rows = only(rows, "benign")

    model, tokenizer = load_model_and_tokenizer(cfg, adapter_path=args.adapter_path)
    common = dict(
        model=model,
        tokenizer=tokenizer,
        system_prompt=cfg["model"]["system_prompt"],
        max_input_length=int(cfg["model"]["max_input_length"]),
        max_new_tokens=int(cfg["causal"]["max_new_tokens"]),
        batch_size=int(cfg["causal"]["batch_size"]),
    )

    baseline_h = generate_refusal_labels(rows=harmful_rows, **common)
    baseline_b = generate_refusal_labels(rows=benign_rows, **common)

    mode = cfg["causal"].get("intervention_mode", "center_projection")
    ablation = center_projection_transform(direction, center) if mode == "center_projection" else zero_projection_transform(direction)
    ablated_h = generate_refusal_labels(rows=harmful_rows, layer_index=layer, transform=ablation, **common)

    alpha = float(cfg["causal"]["addition_alpha"])
    added_b = generate_refusal_labels(rows=benign_rows, layer_index=layer, transform=addition_transform(direction, alpha * gap), **common)

    # Matched random directions give a causal null using the same intervention magnitude.
    rng = np.random.default_rng(seed + 303)
    null_records = []
    null_deltas_ablate = []
    null_deltas_add = []
    discovery_acts = np.load(
        project_path(cfg["activations"]["directory"]) / f"discovery_{args.suffix}.npz",
        allow_pickle=False,
    )
    discovery_layer = discovery_acts["activations"][:, layer].astype(np.float32)
    discovery_harm = discovery_acts["harmfulness"] == "harmful"

    for null_i in range(int(cfg["causal"]["random_direction_nulls"])):
        rd = rng.normal(size=direction.shape[0]).astype(np.float32)
        rd /= np.linalg.norm(rd)
        rd_proj = discovery_layer @ rd
        rd_center = float(
            (rd_proj[discovery_harm].mean() + rd_proj[~discovery_harm].mean()) / 2.0
        )
        null_ab = generate_refusal_labels(
            rows=harmful_rows,
            layer_index=layer,
            transform=center_projection_transform(rd, rd_center),
            **common,
        )
        null_ad = generate_refusal_labels(
            rows=benign_rows,
            layer_index=layer,
            transform=addition_transform(rd, alpha * abs(gap)),
            **common,
        )
        da = refusal_rate(null_ab) - refusal_rate(baseline_h)
        dd = refusal_rate(null_ad) - refusal_rate(baseline_b)
        null_deltas_ablate.append(da)
        null_deltas_add.append(dd)
        null_records.append({"null": null_i, "harmful_ablation_delta": da, "benign_addition_delta": dd})

    base_h_vec = np.asarray([r["refusal"] for r in baseline_h], dtype=float)
    abl_h_vec = np.asarray([r["refusal"] for r in ablated_h], dtype=float)
    add_b_vec = np.asarray([r["refusal"] for r in added_b], dtype=float)
    base_b_vec = np.asarray([r["refusal"] for r in baseline_b], dtype=float)
    ablation_delta = refusal_rate(ablated_h) - refusal_rate(baseline_h)
    addition_delta = refusal_rate(added_b) - refusal_rate(baseline_b)
    ablation_ci = bootstrap_rate_difference_ci(abl_h_vec, base_h_vec, samples=int(cfg["causal"]["bootstrap_samples"]), seed=seed + 501)
    addition_ci = bootstrap_rate_difference_ci(add_b_vec, base_b_vec, samples=int(cfg["causal"]["bootstrap_samples"]), seed=seed + 502)

    public_rows = []
    public_rows += add_condition(baseline_h, "harmful_baseline")
    public_rows += add_condition(ablated_h, "harmful_direction_ablation")
    public_rows += add_condition(baseline_b, "benign_baseline")
    public_rows += add_condition(added_b, "benign_direction_addition")
    private_behavior_dir = project_path("outputs/behavior")
    private_behavior_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(public_rows).to_csv(
        private_behavior_dir / f"causal_behavior_{args.suffix}.csv",
        index=False,
    )
    pd.DataFrame(null_records).to_csv(
        result_dir / f"causal_random_null_{args.suffix}.csv",
        index=False,
    )

    summary = {
        "selected_layer": layer,
        "direction_source": "ground-truth harmfulness mean difference on discovery split",
        "intervention_mode": mode,
        "harmful_baseline_refusal_rate": refusal_rate(baseline_h),
        "harmful_ablation_refusal_rate": refusal_rate(ablated_h),
        "harmful_ablation_delta": ablation_delta,
        "harmful_ablation_delta_95ci": list(ablation_ci),
        "benign_baseline_refusal_rate": refusal_rate(baseline_b),
        "benign_addition_refusal_rate": refusal_rate(added_b),
        "benign_addition_delta": addition_delta,
        "benign_addition_delta_95ci": list(addition_ci),
        "random_direction_mean_harmful_ablation_delta": float(np.mean(null_deltas_ablate)),
        "random_direction_mean_benign_addition_delta": float(np.mean(null_deltas_add)),
        "raw_generations_saved": False,
        "interpretation_rule": "Call the direction refusal-linked only if held-out causal effects are directionally consistent and materially exceed matched random-direction controls. Predictive separation alone is insufficient.",
    }
    write_json(result_dir / f"causal_summary_{args.suffix}.json", summary)
    print(summary)
    print("Raw generated text saved: NO")


if __name__ == "__main__":
    main()
