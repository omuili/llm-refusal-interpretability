# Mechanistic Interpretability Result Summary

This file is generated from local experiment outputs. It does not contain raw WildJailbreak prompts or raw model generations.

## Direction discovery

- Selected block: **7**, chosen on validation AUROC only.
- Validation AUROC: **0.9012**.
- Held-out test AUROC: **0.8753**.
- Test 95% bootstrap CI: **[0.8509, 0.8983]**.
- Random-direction empirical p: **0.0099**.
- Vanilla to adversarial transfer AUROC: **0.7170**.
- Adversarial to vanilla transfer AUROC: **0.8097**.

## Linear probe

- Selected block: **15**, selected on validation only.
- Held-out test AUROC: **0.9684**.
- Shuffled-label empirical p: **0.0385**.

## Interpretation guardrail

A high projection or probe AUROC shows that harmfulness information is linearly accessible. It does not by itself show that the direction causes refusal. The direction should be described as refusal-linked only when held-out interventions alter refusal behavior and outperform matched random-direction controls.

## Causal intervention

- Harmful baseline refusal rate: **0.410**.
- Harmful refusal rate after ablation: **0.430**.
- Ablation delta: **+0.020**.
- Ablation 95% CI: **[-0.120, +0.160]**.
- Matched random-direction mean ablation delta: **+0.003**.
- Benign baseline refusal rate: **0.070**.
- Benign refusal rate after direction addition: **0.080**.
- Addition delta: **+0.010**.
- Addition 95% CI: **[-0.060, +0.090]**.
- Matched random-direction mean addition delta: **-0.000**.

### Causal conclusion

The tested intervention does **not** provide convincing evidence that the selected harmfulness-associated direction causally controls refusal. The ablation effect is not directionally consistent with the hypothesis, the addition effect is small, and both confidence intervals include zero.

Raw generated text was not saved by the causal evaluator.
