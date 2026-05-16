# HAWQ-V2

Standalone PyTorch implementation of the HAWQ-V2 mixed-precision initializer from:

`HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks`

This package implements the HAWQ-V2 selection procedure as a self-contained repository:

- Hutchinson-based average Hessian trace estimation
- trace-weighted quantization perturbation scoring
- mixed-precision configuration search
- Pareto-style frontier export over Omega vs bit-complexity
- structured result export for downstream quantization workflows

The implementation is designed to be usable as a general standalone HAWQ-V2 initializer for models built from
`nn.Conv1d`, `nn.Conv2d`, and `nn.Linear`, including wrapped modules that expose the underlying weight-bearing layer
through `.op`.

## Scope

This repository focuses on the HAWQ-V2 precision-initialization stage:

- it estimates layer sensitivity using average Hessian trace
- it evaluates candidate weight precisions using trace-weighted perturbation
- it can export the candidate frontier directly, or optionally select a single assignment

This repository is not a full quantization framework. It does not try to own model rewriting, quantizer insertion,
training schedules, deployment export, or hardware-specific execution stacks. Those belong around the initializer, not
inside it.

Activation precision is configurable:

- `fixed`: keep all activations at a chosen bitwidth, typically INT8
- `inherit`: assign each activation the adjacent selected weight precision
- `disabled`: return no activation assignment

That means weight-only mixed-precision workflows are supported as a configuration choice, not as a hard-coded
limitation of the implementation.

## Method

For each target layer, the initializer computes:

1. Average Hessian trace using Hutchinson iterations.
2. Quantization perturbation for each candidate bitwidth.
3. The HAWQ-V2 sensitivity term:

```text
trace(layer) * ||Q(W_layer) - W_layer||_2^2
```

For a full configuration, the repository sums the layer sensitivities into the HAWQ-V2 objective:

```text
Omega(b) = sum_i trace(layer_i) * ||Q_{b_i}(W_i) - W_i||_2^2
```

The result frontier is then built over `Omega` vs bit-complexity across the candidate set. The JSON also keeps a
backward-compatible `metric` field equal to `omega`.

Selection modes:

- `pareto`: default, exports the Pareto-style frontier and does not force a single chosen config
- `compression_ratio`: NNCF-style selection of the lowest-Omega config among configs meeting a requested ratio target
- `min_metric`: choose the minimum-Omega config directly

Two search modes are supported:

- `monotonic`: uses the standard HAWQ restriction that less sensitive layers receive lower precision
- `all`: evaluates the full Cartesian product of candidate bitwidths

## Repository Layout

```text
hawqv2/
├── examples/
│   ├── indoor_alpha_sweep_manifest.json
│   └── sample_hawqv2_result.json
├── src/hawqv2/
│   ├── __init__.py
│   ├── bitwidth_export.py
│   ├── compression_ratio.py
│   ├── config.py
│   ├── examples/
│   │   └── toy_factory.py
│   ├── hessian_trace.py
│   ├── initializer.py
│   ├── model.py
│   ├── perturbations.py
│   ├── quantization.py
│   ├── selector.py
│   └── traces_order.py
└── tools/
    ├── extract_bitwidths.py
    ├── make_ai8x_qat_policy.py
    ├── print_hawq_frontier.py
    ├── run_hawqv2.py
    ├── run_hawqv2_indoor.py
    ├── run_hawqv2_indoor_sweep.py
    └── select_ace_from_hawq.py
```

Core modules:

- [initializer.py](/Users/hamza/Documents/GitHub/Indoor-Environment-Quantization/hawqv2/src/hawqv2/initializer.py): standalone HAWQ-V2 initializer
- [hessian_trace.py](/Users/hamza/Documents/GitHub/Indoor-Environment-Quantization/hawqv2/src/hawqv2/hessian_trace.py): Hutchinson trace estimator
- [compression_ratio.py](/Users/hamza/Documents/GitHub/Indoor-Environment-Quantization/hawqv2/src/hawqv2/compression_ratio.py): bit-complexity and compression-ratio accounting
- [quantization.py](/Users/hamza/Documents/GitHub/Indoor-Environment-Quantization/hawqv2/src/hawqv2/quantization.py): quantization and perturbation computation
- [model.py](/Users/hamza/Documents/GitHub/Indoor-Environment-Quantization/hawqv2/src/hawqv2/model.py): target-layer discovery and layer complexity profiling

## Programmatic Use

```python
import torch
from hawqv2 import ActivationBitwidthConfig
from hawqv2 import HAWQConfig
from hawqv2 import StandaloneHAWQPrecisionInitializer
from hawqv2 import save_hawq_result

config = HAWQConfig(
    candidate_bits=[2, 4, 8],
    selection="pareto",
    compression_ratio=None,
    num_data_points=100,
    max_trace_iters=200,
    tolerance=1e-4,
    device="cuda",
    quantization_mode="asymmetric",
    per_channel=False,
    search="monotonic",
    activation=ActivationBitwidthConfig(mode="fixed", bits=8),
)

initializer = StandaloneHAWQPrecisionInitializer(
    model=model,
    data_loader=train_loader,
    criterion=torch.nn.CrossEntropyLoss(),
    config=config,
    criterion_fn=None,
    layer_names=None,
)

result = initializer.apply_init()
save_hawq_result(result, "hawqv2/hawqv2_runs/result.json")
```

For simpler use, the package also exposes `run_hawqv2(...)` through [selector.py](/Users/hamza/Documents/GitHub/Indoor-Environment-Quantization/hawqv2/src/hawqv2/selector.py).

## Generic CLI

Use [run_hawqv2.py](/Users/hamza/Documents/GitHub/Indoor-Environment-Quantization/hawqv2/tools/run_hawqv2.py) with a
factory function that returns a model, a dataloader, a criterion, and optionally a custom criterion function.

```bash
python hawqv2/tools/run_hawqv2.py \
  --factory my_project.hawq_factory:build \
  --factory-kwargs '{"checkpoint": "path/to/model.pth"}' \
  --search all \
  --output hawqv2/hawqv2_runs/result.json
```

Factory return conventions:

- `(model, data_loader, criterion)`
- `(model, data_loader, criterion, criterion_fn)`
- `{"model": ..., "data_loader": ..., "criterion": ..., "criterion_fn": ...}`

The initializer auto-discovers supported weight layers by default. Use `--layers ...` when you want to restrict the
search to specific modules.

Checkpoint guidance:

- the intended input is a floating-point or pre-QAT model checkpoint
- this is the standard use of HAWQ-V2 as a precision initializer before final mixed-precision quantization/training
- a randomly initialized model is only useful for smoke testing
- an already quantized or QAT-finished checkpoint can still be analyzed, but it is not the preferred starting point
  for selecting the bit configuration

CLI smoke-test with the included toy factory:

```bash
python hawqv2/tools/run_hawqv2.py \
  --factory hawqv2.examples.toy_factory:build \
  --output hawqv2/hawqv2_runs/toy_result.json \
  --device cpu \
  --num-data-points 16 \
  --max-trace-iters 2
```

## Result Format

The output JSON contains:

- repository method/configuration metadata
- target layer names
- trace order
- per-layer traces
- per-layer perturbations for each candidate bitwidth
- per-layer Omega terms, `trace(layer) * perturbation(layer, bitwidth)`
- per-layer complexity estimates
- Pareto-style frontier entries
- all evaluated configurations
- optionally, one selected configuration

Example:

```json
{
  "pareto_frontier": [
    {
      "layer_bits": {
        "features.0": 2,
        "features.2": 2,
        "classifier.1": 2,
        "classifier.3": 2
      },
      "omega": 0.456,
      "metric": 0.456,
      "omega_by_layer": {
        "features.0": 0.12,
        "features.2": 0.22,
        "classifier.1": 0.11,
        "classifier.3": 0.006
      },
      "compression_ratio": 4.0,
      "bit_complexity": 1234.0
    }
  ],
  "selected_config": null
}
```

To inspect a saved result as a table:

```bash
python hawqv2/tools/print_hawq_frontier.py \
  --input hawqv2/hawqv2_runs/result.json \
  --show frontier
```

## HAWQ + ACE Deployment Selection

HAWQ-V2 does not observe final QAT accuracy and therefore should not be treated as the final deployment selector when
the deployment objective is accuracy-constrained. In this paper repo, the intended workflow is:

1. HAWQ-V2 scores candidate bit assignments from the FP32 checkpoint using `Omega`.
2. The HAWQ size-Omega frontier is retained as a reduced candidate set.
3. The retained candidates are QAT-trained/evaluated using the normal ai8x flow.
4. ACE selects the final deployment configuration from the HAWQ-pruned set.

This matches the paper's use of HAWQ-V2 as a complementary structure-aware pruning method. The exhaustive sweep defines
the oracle; HAWQ is used to test whether a much smaller Omega-guided candidate set preserves the ACE-optimal region.

The ACE rule used by the paper is:

```text
ACE(A, S) =
  beta1 * (1 - S / S_base) + beta2 * ((A - A_tgt) / (100 - A_tgt)), if A >= A_tgt
  -(A_tgt - A) / A_tgt, otherwise
```

For the high-accuracy compact operating point, the paper uses:

```text
A_tgt = 99.2
beta1 = 1.0
beta2 = 0.0
```

With `beta1=1.0` and `beta2=0.0`, ACE reduces to choosing the smallest model among candidates whose QAT accuracy meets
the target.

Use [select_ace_from_hawq.py](/Users/hamza/Documents/GitHub/Indoor-Environment-Quantization/hawqv2/tools/select_ace_from_hawq.py)
to join HAWQ candidates with QAT sweep results:

```bash
python hawqv2/tools/select_ace_from_hawq.py \
  --item 91:hawqv2/hawqv2_runs/indoor_L91_float_hawqv2_pareto.json \
  --sweep-csv results/mixed_precision_sweep_summary_sizes.csv \
  --candidate-set frontier \
  --target-acc 99.2 \
  --beta1 1.0 \
  --beta2 0.0
```

Example result for the `alpha=91` FP32 checkpoint:

```text
selected: INT 8-8-2-2 @ alpha=91
acc  = 99.2068%
size = 46.360 kB
ACE  = 0.772141944
omega = 0.000100798817584
```

This means `INT 8-8-2-2` is not chosen by HAWQ alone. It is first retained by the HAWQ size-Omega frontier, then selected
by the paper's ACE rule after joining with QAT accuracy and model-size results.

To apply ACE across multiple alpha values, pass multiple `--item` arguments:

```bash
python hawqv2/tools/select_ace_from_hawq.py \
  --item 91:hawqv2/hawqv2_runs/indoor_L91_float_hawqv2_pareto.json \
  --item 101:hawqv2/hawqv2_runs/indoor_L101_float_hawqv2_pareto.json \
  --sweep-csv results/mixed_precision_sweep_summary_sizes.csv \
  --candidate-set frontier \
  --target-acc 99.2 \
  --beta1 1.0 \
  --beta2 0.0 \
  --output hawqv2/hawqv2_runs/hawq_frontier_ace99p2.json
```

For a full 11-alpha study, first create a manifest with one FP32 checkpoint per input length:

```json
{
  "items": [
    {
      "tag": "L101",
      "alpha": 101,
      "input_length": 101,
      "checkpoint": "/Users/hamza/Desktop/testMax/ai8x-training/hawq_float_runs/indoor_float_L101_seed_42___YYYY.MM.DD-HHMMSS/indoor_float_L101_seed_42_best.pth.tar"
    },
    {
      "tag": "L91",
      "alpha": 91,
      "input_length": 91,
      "checkpoint": "/Users/hamza/Desktop/testMax/ai8x-training/hawq_float_runs/indoor_float_L91_seed_42___YYYY.MM.DD-HHMMSS/indoor_float_L91_seed_42_best.pth.tar"
    }
  ]
}
```

Then run HAWQ-V2 in `pareto` mode for every FP32 checkpoint:

```bash
conda activate max

python hawqv2/tools/run_hawqv2_indoor_sweep.py \
  --ai8x-root /Users/hamza/Desktop/testMax/ai8x-training \
  --data-dir /Users/hamza/Desktop/testMax/ai8x-training/data/indoor_environment \
  --manifest hawqv2/hawqv2_runs/indoor_alpha_sweep_manifest.json \
  --output-dir hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto \
  --seed 42 \
  --device cpu \
  --bits 2 4 8 \
  --search all \
  --selection pareto \
  --num-data-points 100 \
  --max-trace-iters 200 \
  --skip-policy
```

This writes one HAWQ result per alpha under:

```text
hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto/results/
```

Finally, apply ACE to the union of HAWQ frontier candidates. The exact command should include one `--item` per alpha:

```bash
python hawqv2/tools/select_ace_from_hawq.py \
  --item 101:hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto/results/L101.json \
  --item 91:hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto/results/L91.json \
  --item 81:hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto/results/L81.json \
  --item 71:hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto/results/L71.json \
  --item 61:hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto/results/L61.json \
  --item 51:hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto/results/L51.json \
  --item 41:hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto/results/L41.json \
  --item 31:hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto/results/L31.json \
  --item 21:hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto/results/L21.json \
  --item 11:hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto/results/L11.json \
  --item 5:hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto/results/L5.json \
  --sweep-csv results/mixed_precision_sweep_summary_sizes.csv \
  --candidate-set frontier \
  --target-acc 99.2 \
  --beta1 1.0 \
  --beta2 0.0 \
  --limit 0 \
  --output hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto/hawq_frontier_ace99p2.json
```

The validation question is whether the union of HAWQ frontiers still selects the exhaustive ACE-optimal configuration.
For the current paper setting, the expected high-accuracy compact result is `INT 8-8-2-2` at `alpha=91`.

If you request an explicit selection mode, `selected_config` is populated:

```json
{
  "selected_config": {
    "layer_bits": {
      "features.0": 8,
      "features.2": 4,
      "classifier.1": 2,
      "classifier.3": 2
    }
  }
}
```

## AI8X Extras

This repository also includes helpers for the ai8x/MAX78002 workflow used in this paper repo. These are extensions on
top of the standalone initializer, not the core of the implementation.

Available helpers:

- [run_hawqv2_indoor.py](/Users/hamza/Documents/GitHub/Indoor-Environment-Quantization/hawqv2/tools/run_hawqv2_indoor.py):
  convenience runner for the indoor ai8x model in this repository
- [run_hawqv2_indoor_sweep.py](/Users/hamza/Documents/GitHub/Indoor-Environment-Quantization/hawqv2/tools/run_hawqv2_indoor_sweep.py):
  manifest-driven sweep runner for multiple alpha/input-length checkpoints
- [make_ai8x_qat_policy.py](/Users/hamza/Documents/GitHub/Indoor-Environment-Quantization/hawqv2/tools/make_ai8x_qat_policy.py):
  export a selected configuration as an ai8x QAT policy
- [extract_bitwidths.py](/Users/hamza/Documents/GitHub/Indoor-Environment-Quantization/hawqv2/tools/extract_bitwidths.py):
  normalize a result JSON into a simple `layer_bits` JSON

Example indoor workflow:

```bash
conda activate max

python hawqv2/tools/run_hawqv2_indoor.py \
  --data-dir training/data/indoor_environment \
  --checkpoint path/to/checkpoint.pth.tar \
  --search all \
  --output hawqv2/hawqv2_runs/indoor_hawqv2.json \
  --policy-output hawqv2/hawqv2_runs/qat_policy_hawqv2.yaml \
  --selection compression_ratio \
  --compression-ratio 1.5
```

For the indoor workflow, use a floating-point or pre-QAT checkpoint if possible. That is the correct input for HAWQ
precision initialization. Using a final quantized/QAT checkpoint is possible as a diagnostic, but it is not the
preferred path for selecting the mixed-precision configuration.

This convenience runner expects the ai8x-side model/runtime pieces used by the training overlay in this repository.
The standalone HAWQ-V2 initializer itself does not depend on ai8x.

Alpha/input-length sweep:

When `alpha` changes the effective 1D input bandwidth or model input size, each alpha is a separate outer candidate.
Use the sweep runner to apply HAWQ-V2 to each fixed checkpoint independently.

Example manifest:

- [indoor_alpha_sweep_manifest.json](/Users/hamza/Documents/GitHub/Indoor-Environment-Quantization/hawqv2/examples/indoor_alpha_sweep_manifest.json)

Example sweep command:

```bash
conda activate max

python hawqv2/tools/run_hawqv2_indoor_sweep.py \
  --ai8x-root /path/to/ai8x-training \
  --data-dir /path/to/ai8x-training/data/indoor_environment \
  --manifest hawqv2/examples/indoor_alpha_sweep_manifest.json \
  --output-dir hawqv2/hawqv2_runs/indoor_alpha_sweep \
  --seed 42 \
  --device cpu \
  --bits 2 4 8 \
  --compression-ratio 1.5 \
  --num-data-points 100 \
  --max-trace-iters 200
```

The sweep runner writes:

- one HAWQ result JSON per manifest item
- one ai8x YAML policy per manifest item when a single config is selected
- `summary.json` with all selected configurations and metadata
- `summary.csv` for quick spreadsheet/table use

Use a fixed `--seed` for reproducible dataset splitting and Hutchinson trace sampling across the entire sweep.

The aggregate summary preserves one row per alpha/input-length checkpoint. It does not declare a global best model
across different alpha values by itself; that final selection should be made against your actual paper objective
(accuracy, memory, latency, or Pareto frontier) after downstream evaluation.

## Notes

- Candidate bitwidths are configurable and are not restricted to `{2, 4, 8}`.
- For very small models, `--search all` is often practical and may be preferable when you want an exact exhaustive
  HAWQ score over the candidate set.
- For larger models, `monotonic` search is the intended default.
- If you use this repository in a weight-only mixed-precision workflow, fixing activations to INT8 is a usage choice
  made through the activation policy, not a built-in limitation of the implementation.
- If you want an ai8x YAML policy, run with `--selection compression_ratio` or `--selection min_metric`; pure
  `pareto` mode does not choose a single configuration.
