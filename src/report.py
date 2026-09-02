from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

from .config import load_config, project_path


def maybe_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--suffix", default="base", choices=["base", "adapter"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = project_path(cfg["results"]["dir"])
    suffix = args.suffix

    direction = maybe_json(results / f"direction_summary_{suffix}.json")
    probe = maybe_json(results / f"probe_summary_{suffix}.json")
    causal = maybe_json(results / f"causal_summary_{suffix}.json")
    if not direction or not probe:
        raise FileNotFoundError("Run direction discovery and probe training before generating the report.")

    layers = pd.read_csv(results / f"direction_layers_{suffix}.csv")
    probes = pd.read_csv(results / f"probe_layers_{suffix}.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers["layer"], layers["validation_auroc"], marker="o", label="Direction projection")
    ax.plot(probes["layer"], probes["validation_auroc"], marker="o", label="Linear probe")
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xlabel("Transformer block")
    ax.set_ylabel("Validation AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Where Safety-Relevant Information Is Linearly Accessible")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(results / f"layer_auroc_{suffix}.png", dpi=200)
    plt.close(fig)

    nulls = pd.read_csv(results / f"direction_random_null_{suffix}.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(nulls["random_validation_auroc"], bins=20)
    ax.axvline(direction["validation_auroc"], linewidth=2, label="Observed direction")
    ax.set_xlabel("Validation AUROC")
    ax.set_ylabel("Random directions")
    ax.set_title("Random-Direction Null Control")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / f"direction_null_{suffix}.png", dpi=200)
    plt.close(fig)

    if causal:
        labels = ["Harmful\nbaseline", "Harmful\nablation", "Benign\nbaseline", "Benign\naddition"]
        values = [
            causal["harmful_baseline_refusal_rate"],
            causal["harmful_ablation_refusal_rate"],
            causal["benign_baseline_refusal_rate"],
            causal["benign_addition_refusal_rate"],
        ]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.bar(labels, values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Refusal rate")
        ax.set_title("Held-Out Causal Intervention on Refusal Behavior")
        fig.tight_layout()
        fig.savefig(results / f"causal_refusal_{suffix}.png", dpi=200)
        plt.close(fig)

    lines = [
        "# Mechanistic Interpretability Result Summary",
        "",
        "This file is generated from local experiment outputs. It does not contain raw WildJailbreak prompts or raw model generations.",
        "",
        "## Direction discovery",
        "",
        f"- Selected block: **{direction['selected_layer']}**, chosen on validation AUROC only.",
        f"- Validation AUROC: **{direction['validation_auroc']:.4f}**.",
        f"- Held-out test AUROC: **{direction['held_out_test_auroc']:.4f}**.",
        f"- Test 95% bootstrap CI: **[{direction['held_out_test_auroc_95ci'][0]:.4f}, {direction['held_out_test_auroc_95ci'][1]:.4f}]**.",
        f"- Random-direction empirical p: **{direction['random_direction_empirical_p']:.4f}**.",
        f"- Vanilla to adversarial transfer AUROC: **{direction['cross_style_transfer']['vanilla_to_adversarial_test_auroc']:.4f}**.",
        f"- Adversarial to vanilla transfer AUROC: **{direction['cross_style_transfer']['adversarial_to_vanilla_test_auroc']:.4f}**.",
        "",
        "## Linear probe",
        "",
        f"- Selected block: **{probe['selected_layer']}**, selected on validation only.",
        f"- Held-out test AUROC: **{probe['held_out_test_auroc']:.4f}**.",
        f"- Shuffled-label empirical p: **{probe['shuffled_label_empirical_p']:.4f}**.",
        "",
        "## Interpretation guardrail",
        "",
        "A high projection or probe AUROC shows that harmfulness information is linearly accessible. It does not by itself show that the direction causes refusal. The direction should be described as refusal-linked only when held-out interventions alter refusal behavior and outperform matched random-direction controls.",
    ]
    if causal:
        lines += [
            "",
            "## Causal intervention",
            "",
            f"- Harmful baseline refusal rate: **{causal['harmful_baseline_refusal_rate']:.3f}**.",
            f"- Harmful refusal rate after ablation: **{causal['harmful_ablation_refusal_rate']:.3f}**.",
            f"- Ablation delta: **{causal['harmful_ablation_delta']:+.3f}**.",
            f"- Ablation 95% CI: **[{causal['harmful_ablation_delta_95ci'][0]:+.3f}, {causal['harmful_ablation_delta_95ci'][1]:+.3f}]**.",
            f"- Matched random-direction mean ablation delta: **{causal['random_direction_mean_harmful_ablation_delta']:+.3f}**.",
            f"- Benign baseline refusal rate: **{causal['benign_baseline_refusal_rate']:.3f}**.",
            f"- Benign refusal rate after direction addition: **{causal['benign_addition_refusal_rate']:.3f}**.",
            f"- Addition delta: **{causal['benign_addition_delta']:+.3f}**.",
            f"- Addition 95% CI: **[{causal['benign_addition_delta_95ci'][0]:+.3f}, {causal['benign_addition_delta_95ci'][1]:+.3f}]**.",
            f"- Matched random-direction mean addition delta: **{causal['random_direction_mean_benign_addition_delta']:+.3f}**.",
            "",
            "### Causal conclusion",
            "",
            "The tested intervention does **not** provide convincing evidence that the selected harmfulness-associated direction causally controls refusal. The ablation effect is not directionally consistent with the hypothesis, the addition effect is small, and both confidence intervals include zero.",
            "",
            "Raw generated text was not saved by the causal evaluator.",
        ]
    (results / f"SUMMARY_{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {results / f'SUMMARY_{suffix}.md'}")


if __name__ == "__main__":
    main()
