from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import load_config, project_path
from .directions import binary_labels, bootstrap_auc_ci
from .probes import fit_probe, probe_auc, shuffled_label_null
from .utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--suffix", default="base", choices=["base", "adapter"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    actdir = project_path(cfg["activations"]["directory"])
    discovery = np.load(actdir / f"discovery_{args.suffix}.npz", allow_pickle=False)
    validation = np.load(actdir / f"validation_{args.suffix}.npz", allow_pickle=False)
    test = np.load(actdir / f"test_{args.suffix}.npz", allow_pickle=False)
    y_tr = binary_labels(discovery["harmfulness"], "harmful")
    y_val = binary_labels(validation["harmfulness"], "harmful")
    y_te = binary_labels(test["harmfulness"], "harmful")
    c = float(cfg["probe"]["c"])
    max_iter = int(cfg["probe"]["max_iter"])

    rows = []
    models = []
    for layer in range(discovery["activations"].shape[1]):
        model = fit_probe(discovery["activations"][:, layer].astype(np.float32), y_tr, c=c, max_iter=max_iter)
        val_auc = probe_auc(model, validation["activations"][:, layer].astype(np.float32), y_val)
        rows.append({"layer": layer, "validation_auroc": val_auc})
        models.append(model)

    frame = pd.DataFrame(rows)
    selected = int(frame.loc[frame["validation_auroc"].idxmax(), "layer"])
    model = models[selected]
    x_te = test["activations"][:, selected].astype(np.float32)
    test_scores = model.predict_proba(x_te)[:, 1]
    test_auc = float(roc_auc_score(y_te, test_scores))
    ci = bootstrap_auc_ci(y_te, test_scores, samples=int(cfg["probe"]["bootstrap_samples"]), seed=int(cfg["project"]["seed"]) + 9)
    nulls = shuffled_label_null(
        discovery["activations"][:, selected].astype(np.float32), y_tr,
        x_te, y_te, c=c, max_iter=max_iter,
        n=int(cfg["probe"]["shuffled_label_nulls"]), seed=int(cfg["project"]["seed"]) + 23,
    )
    p_emp = float((1 + np.sum(nulls >= test_auc)) / (len(nulls) + 1))

    results = project_path(cfg["results"]["dir"])
    frame.to_csv(results / f"probe_layers_{args.suffix}.csv", index=False)
    pd.DataFrame({"shuffled_label_test_auroc": nulls}).to_csv(results / f"probe_shuffled_null_{args.suffix}.csv", index=False)
    summary = {
        "selected_layer": selected,
        "selection_metric": "validation AUROC",
        "held_out_test_auroc": test_auc,
        "held_out_test_auroc_95ci": list(ci),
        "shuffled_label_null_mean_auroc": float(nulls.mean()),
        "shuffled_label_empirical_p": p_emp,
        "interpretation": "Probe accuracy is predictive evidence, not causal evidence. Causal intervention is evaluated separately.",
    }
    write_json(results / f"probe_summary_{args.suffix}.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
