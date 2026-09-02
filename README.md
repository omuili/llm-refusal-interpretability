# LLM Refusal Interpretability

I built this project to investigate how harmfulness is represented inside **Qwen2.5-1.5B-Instruct**, how strongly that information can be recovered from internal activations, whether the representation transfers across prompt styles, and whether manipulating the discovered representation changes refusal behavior.

I used the gated **WildJailbreak** dataset and combined residual-stream activation analysis, mean-difference directions, linear probing, cross-style transfer, explicit null controls, bootstrap confidence intervals, and causal interventions.

My central research question was:

> **Does Qwen2.5-1.5B contain an internal representation of harmfulness that is not only decodable, but also causally involved in refusal behavior?**

## Results

I found strong evidence that **harmfulness information is represented and linearly accessible inside Qwen2.5-1.5B**.

I did **not** find convincing evidence that the specific harmfulness-associated mean-difference direction identified in the study causally controls refusal behavior.

| Analysis | Result |
|---|---:|
| Mean-difference direction, selected block | 7 |
| Validation AUROC | 0.9012 |
| Held-out direction AUROC | **0.8753** |
| Direction test 95% CI | [0.8509, 0.8983] |
| Random-direction empirical p | 0.0099 |
| Vanilla → adversarial transfer AUROC | 0.7170 |
| Adversarial → vanilla transfer AUROC | 0.8097 |
| Best linear-probe block | 15 |
| Held-out probe AUROC | **0.9684** |
| Probe test 95% CI | [0.9573, 0.9781] |
| Shuffled-label empirical p | 0.0385 |

![Layer-wise harmfulness separation](results/layer_auroc_base.png)

### Representation versus causality

The strongest simple harmful-minus-benign direction appeared at **block 7**, while the strongest linear probe appeared at **block 15**.

This difference became one of the most interesting findings in the project. The block-7 mean-difference direction captured a strong harmfulness-associated axis, but block 15 contained even more linearly decodable harmfulness information.

I then tested the block-7 direction causally on held-out prompts.

| Causal measurement | Result |
|---|---:|
| Harmful baseline refusal | 41% |
| Harmful refusal after ablation | 43% |
| Ablation delta | +2 pp |
| Ablation 95% CI | [-12 pp, +16 pp] |
| Benign baseline refusal | 7% |
| Benign refusal after addition | 8% |
| Addition delta | +1 pp |
| Addition 95% CI | [-6 pp, +9 pp] |

![Causal refusal intervention](results/causal_refusal_base.png)

The ablation effect moved in the opposite direction from the simple causal hypothesis, the addition effect was small, and both confidence intervals included zero.

I therefore do **not** describe the discovered vector as a causal refusal direction.

The result I support is narrower:

> **Qwen2.5-1.5B contains a strong, partially cross-style, linearly accessible harmfulness representation, but the tested mean-difference direction does not show convincing causal control over refusal behavior.**

### Main research lesson

**Decodability is not causality.**

The linear probe reached **0.9684 held-out AUROC**, while manipulating the simpler harmfulness-associated direction produced little measurable change in refusal behavior.

That gap is the central interpretability finding of the study.

![Direction versus random nulls](results/direction_null_base.png)

## Experimental Design

### Model

I studied:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The analysis covered residual-stream representations across all **28 transformer blocks**.

### Data

I built a controlled WildJailbreak subset with three independent partitions:

```text
discovery:   800 prompts
validation:  400 prompts
test:        800 prompts
```

Each partition was balanced across the four WildJailbreak categories:

```text
adversarial_benign
adversarial_harmful
vanilla_benign
vanilla_harmful
```

The discovery partition contained:

```text
800 total prompts
400 harmful
400 benign
400 adversarial
400 vanilla
```

The held-out test partition used the same class balance.

I used the three partitions for different purposes:

- **Discovery** for constructing candidate harmfulness directions and training probes.
- **Validation** for selecting transformer blocks and analysis choices.
- **Test** only for final held-out evaluation after those choices were fixed.

This prevented me from selecting a layer because it happened to perform best on the final evaluation examples.

## Leakage Control

WildJailbreak contains related source prompts and transformed adversarial variants, so a naive row-level random split could place semantically linked examples in multiple partitions.

I grouped examples using:

- a stable hash of the source vanilla prompt,
- a stable hash of the normalized evaluated prompt,
- connected leakage groups that merge source-related and exact-prompt-related examples.

I assigned those groups to discovery, validation, or test as units and sampled the balanced research subsets only after the group-level split had been established.

The final preprocessing audit returned:

```text
Leakage check: PASS
```

I did not redistribute the gated WildJailbreak rows in the repository.

## Residual-Stream Activation Analysis

I ran Qwen2.5-1.5B-Instruct with hidden-state output enabled and extracted the residual-stream activation at the **final prompt token** from every transformer block.

For a prompt \(x_i\), the representation at layer \(\ell\) was:

```text
h_i^(ell) = residual-stream activation at the final prompt position
```

For the discovery partition, the resulting activation tensor had shape:

```text
(800, 28, 1536)
```

corresponding to:

```text
800 prompts
28 transformer blocks
1536 hidden dimensions per block
```

I extracted separate activation tensors for discovery, validation, and held-out test examples.

The activation tensors remained local experimental artifacts. Raw prompt text was not copied into the public activation outputs.

## Mean-Difference Direction

At each transformer block, I divided the discovery activations by ground-truth harmfulness and constructed a harmful-minus-benign mean-difference direction:

```text
d_l = normalize(mean(harmful_l) - mean(benign_l))
```

I scored examples by projecting their activations onto each candidate direction.

I used **validation AUROC only** to select the block. The strongest direction occurred at **block 7**.

```text
Validation AUROC:       0.9012
Held-out test AUROC:    0.8753
95% bootstrap CI:       [0.8509, 0.8983]
```

The held-out performance remained strong even though the test examples were not used to construct the direction or select the layer.

## Random-Direction Null Control

I compared the discovered harmfulness direction with randomly generated unit directions in the same representation space.

The random directions produced mean validation performance close to chance:

```text
Random-direction mean AUROC: 0.5067
```

The discovered block-7 direction substantially exceeded that null distribution:

```text
Empirical p: 0.0099
```

This gave me evidence that the observed separation was not simply the result of projecting high-dimensional activations onto an arbitrary vector.

## Linear Probe Analysis

I separately trained regularized logistic-regression probes on each layer's residual-stream activations.

The probe analysis measured how linearly accessible harmfulness information was at each layer without constraining the classifier to the simple class-mean direction.

I selected the best probe layer using validation data only.

The strongest probe appeared at **block 15**:

```text
Held-out test AUROC: 0.9684
95% bootstrap CI:    [0.9573, 0.9781]
```

I also repeated the probe analysis with permuted discovery labels.

The shuffled-label null remained close to chance:

```text
Mean shuffled-label AUROC: 0.4989
Empirical p:               0.0385
```

This provided strong evidence that harmfulness information was linearly decodable from the model's internal representations.

## Block 7 Versus Block 15

The mean-difference analysis and the probe analysis selected different layers:

```text
Mean-difference direction: block 7
Best linear probe:         block 15
```

I interpret these as measuring different aspects of the representation.

The mean-difference method looks for a specific geometric axis connecting the average harmful and benign representations.

The linear probe can learn a more flexible linear separating boundary.

The result therefore suggests that harmfulness becomes highly linearly accessible deeper in the model even though the strongest simple class-mean direction appears earlier.

## Cross-Style Transfer

I tested whether the harmfulness signal generalized across prompt styles rather than simply reflecting the difference between vanilla and adversarial prompt forms.

A direction estimated from vanilla prompts and evaluated on adversarial test prompts achieved:

```text
Vanilla → adversarial AUROC: 0.7170
```

A direction estimated from adversarial prompts and evaluated on vanilla test prompts achieved:

```text
Adversarial → vanilla AUROC: 0.8097
```

Both transfer results remained above chance.

The drop relative to the overall held-out direction AUROC of **0.8753** suggests that the representation contains transferable harmfulness information while retaining some dependence on prompt style.

## Causal Intervention

The final analysis tested whether the selected block-7 direction actually influenced refusal behavior.

### Harmful-prompt ablation

For held-out harmful prompts, I removed the prompt representation's displacement along the selected harmfulness direction by projecting it toward the discovery-set center.

The measured refusal rates were:

```text
Baseline refusal rate:        41%
Refusal rate after ablation:  43%
Change:                       +2 percentage points
95% CI:                       [-12, +16]
```

Under the simple causal hypothesis, removing the harmfulness-associated component should have reduced refusal.

That did not happen.

The observed effect was small, moved in the opposite direction, and had a confidence interval that included zero.

### Benign-prompt addition

For held-out benign prompts, I added the selected harmfulness direction by an amount calibrated to the discovery-set harmful-versus-benign projection gap.

The measured refusal rates were:

```text
Baseline refusal rate:        7%
Refusal rate after addition:  8%
Change:                       +1 percentage point
95% CI:                       [-6, +9]
```

The effect was directionally consistent with increased refusal but very small, and its confidence interval also included zero.

### Matched causal nulls

I repeated matched interventions using random directions.

The average random-direction effects were approximately zero:

```text
Random mean harmful-ablation delta: +0.0033
Random mean benign-addition delta:  ~0.0000
```

The causal evidence therefore did not support the stronger claim that the block-7 mean-difference direction directly controls refusal behavior.

## Refusal Measurement

For the causal analysis, I generated short deterministic continuations and measured refusal using a conservative refusal-pattern detector.

I kept row-level non-text behavior metadata under the ignored local `outputs/behavior/` directory and published only aggregate refusal statistics and null-control summaries.

The causal evaluator recorded:

```text
Raw generated text saved: NO
```

This kept raw harmful continuations and row-level gated-dataset evaluation records out of the public release.

I treat the heuristic refusal detector as a limitation of the current experiment rather than as a perfect behavioral oracle.

## What I Conclude

The study supports four increasingly specific conclusions.

### 1. Harmfulness information is internally represented

A simple mean-difference direction separated harmful and benign prompts on held-out data:

```text
AUROC = 0.8753
```

### 2. Harmfulness is strongly linearly decodable

A linear probe achieved:

```text
AUROC = 0.9684
```

on held-out activations.

### 3. The representation partially transfers across prompting styles

Cross-style transfer remained above chance:

```text
Vanilla → adversarial: 0.7170
Adversarial → vanilla: 0.8097
```

### 4. The tested mean-difference direction does not show convincing causal control over refusal

The causal interventions produced only small behavioral changes, and both confidence intervals included zero.

My final conclusion is:

> **Qwen2.5-1.5B contains a strong, partially cross-style, linearly accessible representation of harmfulness. However, the simple mean-difference direction identified in this study does not show convincing causal control over refusal behavior under the intervention tested here.**

## Experimental Controls

I used several controls to reduce the chance of interpreting artifacts as meaningful internal structure:

```text
leakage-controlled discovery, validation, and test partitions
balanced harmful and benign examples
balanced vanilla and adversarial examples
validation-only layer selection
held-out final evaluation
bootstrap confidence intervals
random-direction controls
shuffled-label controls
cross-style transfer tests
matched random-direction causal interventions
```

The empirical null tests used finite numbers of randomizations, so I treat their p-values as coarse empirical estimates rather than highly precise significance measurements.

## Privacy and Responsible Release

I kept sensitive and heavyweight experimental artifacts outside the public repository.

The public release excludes:

```text
WildJailbreak raw rows
raw prompts
raw generated continuations
activation tensors
learned direction tensors
model checkpoints
adapter weights
Hugging Face credentials
```

The `.gitignore` excludes the primary local data and output directories.

I also included an automated pre-publication check that verifies that private paths, model artifacts, activation files, and Hugging Face credentials are not staged for release.

## Repository Structure

```text
configs/
    default.yaml

src/
    prepare_data.py
    extract_activations.py
    discover_direction.py
    train_probe.py
    causal_eval.py
    report.py
    data.py
    activations.py
    directions.py
    probes.py
    interventions.py
    refusal.py
    modeling.py
    ...

scripts/
    run_pipeline.sh
    pre_publish_check.sh

tests/

results/
    README.md
    SUMMARY_base.md
    direction_summary_base.json
    probe_summary_base.json
    causal_summary_base.json
    direction_layers_base.csv
    probe_layers_base.csv
    direction_random_null_base.csv
    probe_shuffled_null_base.csv
    causal_random_null_base.csv
    layer_auroc_base.png
    direction_null_base.png
    causal_refusal_base.png
```

## Reproducibility Record

I ran the experiment in Python 3.11 using the configuration recorded in:

```text
configs/default.yaml
```

The environment setup used:

```bash
PYTHON_BIN="$(uv python find 3.11)" bash setup.sh
source .venv/bin/activate
```

I authenticated locally with Hugging Face because WildJailbreak is gated. No authentication token was committed to the repository.

The experiment sequence I ran was:

```bash
python -m src.preflight --config configs/default.yaml

python -m src.prepare_data --config configs/default.yaml

python -m src.extract_activations --config configs/default.yaml --split discovery
python -m src.extract_activations --config configs/default.yaml --split validation
python -m src.extract_activations --config configs/default.yaml --split test

python -m src.discover_direction --config configs/default.yaml

python -m src.train_probe --config configs/default.yaml

python -m src.causal_eval --config configs/default.yaml

python -m src.report --config configs/default.yaml
```

The repository also records the complete sequence in:

```text
scripts/run_pipeline.sh
```

The pre-specified research design and interpretation criteria are documented in:

```text
RESEARCH_PROTOCOL.md
```

## Generated Research Artifacts

The completed study produced the following public aggregate artifacts:

```text
data_summary.json

direction_layers_base.csv
direction_summary_base.json
direction_random_null_base.csv

probe_layers_base.csv
probe_summary_base.json
probe_shuffled_null_base.csv

causal_summary_base.json
causal_random_null_base.csv

layer_auroc_base.png
direction_null_base.png
causal_refusal_base.png
SUMMARY_base.md
```

The public summary was generated from the measured experiment outputs rather than from hard-coded headline values.

## Tests

I included unit tests covering data-label parsing, direction construction, intervention mathematics, refusal detection, privacy-sensitive output schemas, and utility functions.

The final test run completed with:

```text
.......... [100%]
```

The pre-publication check also passed its staged-file privacy scan, token scan, Python compilation check, and test suite.

## Limitations

I interpret the findings within the scope of this experiment rather than as a universal statement about refusal mechanisms.

The main limitations are:

- I studied one model, Qwen2.5-1.5B-Instruct.
- I used one controlled WildJailbreak split and one project seed.
- I analyzed the residual-stream representation at the final prompt token.
- The mean-difference direction is a simple linear approximation to what may be a distributed nonlinear mechanism.
- The causal experiment tested one intervention family.
- The refusal detector is heuristic.
- WildJailbreak may contain dataset-specific cues that do not transfer to other jailbreak distributions.
- A probe can exploit information that may not itself be used by downstream model computation.
- Failure of the tested intervention does not establish that harmfulness has no causal role elsewhere in the network.
- Harmfulness may be represented through distributed, nonlinear, multi-layer, or circuit-level mechanisms that are not captured by a single residual-stream direction.
- Activation interventions can create off-distribution hidden states, which is why I included matched random-direction controls.

These limitations constrain the strength of the causal claim, but they do not undermine the central empirical result that harmfulness information is strongly and linearly accessible inside the model.

## Research Takeaway

The most important result from this study was not the highest AUROC.

It was the difference between what I could **decode** and what I could **causally manipulate**.

I could predict harmfulness from held-out internal activations with **0.9684 AUROC**, yet intervening on the strongest simple harmfulness-associated direction produced almost no measurable change in refusal behavior.

That gap is the central interpretability finding of the project.

## License

MIT. See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).
