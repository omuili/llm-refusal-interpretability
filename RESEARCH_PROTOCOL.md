# Research Protocol

This document freezes the default interpretation rules for the project before looking at held-out test results.

## Research question

Does Qwen2.5-1.5B-Instruct contain a residual-stream direction that is associated with prompt harmfulness, transfers across vanilla and adversarial prompt styles, and causally influences refusal behavior?

## Variables that must remain distinct

The project intentionally separates:

- **ground-truth harmfulness**, which comes from the dataset label;
- **prompt style**, which is vanilla or adversarial;
- **model refusal behavior**, which is measured from the model's generated response.

A harmful prompt does not automatically imply that the model refused it, and a refusal does not prove that the prompt was harmful.

## Data roles

- **Discovery split:** estimate harmful-minus-benign directions and fit linear probes.
- **Validation split:** select the transformer block and compare candidate layers.
- **Test split:** final held-out evaluation after choices are frozen.

No layer may be selected using test performance.

## Primary representation endpoint

For each transformer block, compute the normalized difference between the mean harmful and mean benign last-prompt-token activation on the discovery split.

Select the block with the highest validation AUROC of the one-dimensional projection score.

Report the selected block's held-out test AUROC with a bootstrap confidence interval.

## Representation null

At the selected block, compare the observed validation AUROC with random unit directions sampled from the same hidden dimension.

Report the null distribution and empirical p-value.

## Cross-style transfer

At the selected block:

1. derive a harmful-minus-benign direction from vanilla discovery examples and evaluate it on adversarial test examples;
2. derive a direction from adversarial discovery examples and evaluate it on vanilla test examples.

These are transfer tests, not layer-selection criteria.

## Probe endpoint

Fit one regularized logistic-regression probe per block on the discovery split. Select the block on validation AUROC and evaluate that probe once on the test split.

Compare with probes trained on shuffled discovery labels.

A successful probe supports linear decodability only. It does not establish causal involvement.

## Causal endpoint

Use the representation-selected block and direction.

On held-out harmful prompts, measure the change in refusal rate after centered directional ablation.

On held-out benign prompts, measure the change in refusal rate after adding the direction by one discovery-set harmful-versus-benign projection gap.

Report bootstrap confidence intervals for both rate differences.

## Causal null

Repeat matched interventions using random directions. A causal interpretation is stronger only when the selected direction's effects are materially larger and directionally more coherent than the random controls.

## Claim ladder

### Level 1: linearly accessible

Supported when the selected direction and/or probe separate harmful from benign prompts on held-out data and beat null controls.

### Level 2: cross-style representation

Supported when directions transfer between vanilla and adversarial prompt styles.

### Level 3: refusal-linked

Supported only when manipulating the direction changes refusal behavior on held-out prompts in the expected direction and exceeds matched random-direction effects.

### Claims this experiment cannot establish by itself

The experiment does not establish:

- a universal refusal mechanism;
- a unique causal direction;
- generalization to all jailbreak distributions;
- generalization to other model families;
- that the model is safe;
- that a linearly decodable feature is the representation the model uses internally in all contexts.

## Reporting rule

Unexpected or negative findings should be reported rather than hidden. If the direction separates labels but causal interventions fail, the correct conclusion is that the project found predictive representation evidence without convincing causal evidence.
