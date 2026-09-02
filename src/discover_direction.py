from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .config import load_config, project_path
from .directions import binary_labels, mean_difference_direction, projection_scores, auc, bootstrap_auc_ci, random_direction_null
from .utils import write_json


def load_npz(path: Path):
    return np.load(path, allow_pickle=False)


def cross_style(train, test, train_style: str, test_style: str, layer: int):
    tr_mask = train["style"] == train_style
    te_mask = test["style"] == test_style
    y_tr = binary_labels(train["harmfulness"][tr_mask], "harmful")
    y_te = binary_labels(test["harmfulness"][te_mask], "harmful")
    direction, _, _ = mean_difference_direction(train["activations"][tr_mask, layer].astype(np.float32), y_tr)
    return auc(y_te, projection_scores(test["activations"][te_mask, layer].astype(np.float32), direction))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--suffix", default="base", choices=["base", "adapter"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    actdir = project_path(cfg["activations"]["directory"])
    discovery = load_npz(actdir / f"discovery_{args.suffix}.npz")
    validation = load_npz(actdir / f"validation_{args.suffix}.npz")
    test = load_npz(actdir / f"test_{args.suffix}.npz")

    y_train = binary_labels(discovery["harmfulness"], "harmful")
    y_val = binary_labels(validation["harmfulness"], "harmful")
    y_test = binary_labels(test["harmfulness"], "harmful")
    n_layers = discovery["activations"].shape[1]

    rows = []
    directions = []
    centers = []
    gaps = []
    for layer in range(n_layers):
        direction, center, gap = mean_difference_direction(discovery["activations"][:, layer].astype(np.float32), y_train)
        val_scores = projection_scores(validation["activations"][:, layer].astype(np.float32), direction)
        test_scores = projection_scores(test["activations"][:, layer].astype(np.float32), direction)
        rows.append({
            "layer": layer,
            "validation_auroc": auc(y_val, val_scores),
            "test_auroc_secondary": auc(y_test, test_scores),
            "projection_center": center,
            "projection_gap": gap,
        })
        directions.append(direction)
        centers.append(center)
        gaps.append(gap)

    frame = pd.DataFrame(rows)
    selected_layer = int(frame.loc[frame["validation_auroc"].idxmax(), "layer"])
    selected_direction = directions[selected_layer]
    selected_scores = projection_scores(test["activations"][:, selected_layer].astype(np.float32), selected_direction)
    test_auc = auc(y_test, selected_scores)
    ci_low, ci_high = bootstrap_auc_ci(
        y_test,
        selected_scores,
        samples=int(cfg["direction"]["bootstrap_samples"]),
        seed=int(cfg["project"]["seed"]),
    )

    null_values = random_direction_null(
        validation["activations"][:, selected_layer].astype(np.float32),
        y_val,
        n=int(cfg["direction"]["random_null_directions"]),
        seed=int(cfg["project"]["seed"]) + 17,
    )
    observed_val = float(frame.loc[frame["layer"] == selected_layer, "validation_auroc"].iloc[0])
    p_emp = float((1 + np.sum(null_values >= observed_val)) / (len(null_values) + 1))

    transfer = {
        "vanilla_to_adversarial_test_auroc": cross_style(discovery, test, "vanilla", "adversarial", selected_layer),
        "adversarial_to_vanilla_test_auroc": cross_style(discovery, test, "adversarial", "vanilla", selected_layer),
    }

    results = project_path(cfg["results"]["dir"])
    results.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results / f"direction_layers_{args.suffix}.csv", index=False)
    private_direction_dir = project_path("outputs/directions")
    private_direction_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        private_direction_dir / f"direction_{args.suffix}.npz",
        directions=np.asarray(directions, dtype=np.float32),
        centers=np.asarray(centers, dtype=np.float32),
        gaps=np.asarray(gaps, dtype=np.float32),
        selected_layer=np.asarray(selected_layer),
    )
    pd.DataFrame({"random_validation_auroc": null_values}).to_csv(results / f"direction_random_null_{args.suffix}.csv", index=False)
    summary = {
        "selected_layer": selected_layer,
        "selection_metric": "validation AUROC",
        "validation_auroc": observed_val,
        "held_out_test_auroc": test_auc,
        "held_out_test_auroc_95ci": [ci_low, ci_high],
        "random_direction_validation_mean_auroc": float(null_values.mean()),
        "random_direction_empirical_p": p_emp,
        "cross_style_transfer": transfer,
        "interpretation": "This is a harmfulness-associated residual-stream direction. It should be called refusal-linked only if causal intervention changes refusal behavior on held-out prompts.",
    }
    write_json(results / f"direction_summary_{args.suffix}.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
