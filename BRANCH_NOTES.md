# Branch notes: optimized experimental pipeline

This file summarizes only the work added on this branch. The original project documentation remains in `README.md`, `RESULTS.md`, `data/README.md`, `scripts/README.md`, and `src/README.md`.

The original README files are intentionally preserved unchanged. Their command names remain valid; on this branch, the stable filenames such as `tune_fusion.py` and `evaluate.py` now contain the optimized implementations described below.

## Overview

This branch keeps the original spectral-stability method and evaluation protocol, but makes the experimental pipeline substantially faster and adds a more complete set of tuning and statistical checks.

Main changes:

- CUDA computation of confidence, agreement, stability, and pairwise Jensen-Shannon divergence.
- Batched CUDA grid search for fusion hyperparameters.
- GPU-backed cross-entropy fitting with analytic gradients.
- Accuracy-based score-scale selection and separate output-temperature calibration.
- Optional mean-zero per-view bias.
- Optional unconstrained-coefficient diagnostic.
- Shared probability/signal computation across hold-out folds and evaluation methods.
- Grouped GPU evaluation so each cached condition is loaded and analyzed once.
- Vectorized paired bootstrap for method-to-method comparisons.
- Reduced terminal verbosity in orchestration scripts, with `--verbose` available for debugging.

## Tests

The focused numerical/parity suite passes 26 tests. It covers:

- NumPy/Torch agreement for profile signals and Jensen-Shannon divergence.
- GPU grid scoring parity.
- CE objective and analytic gradient checks.
- View-bias parameterization.
- Output-temperature calibration.
- Grouped evaluation parity.

`test_models.py` remains a separate model-loading smoke test and requires the full Torch/timm environment and pretrained model downloads.

## Runtime improvements

On the development machine (RTX 5070, 12 GB VRAM), the ResNet-18 fixed-ladder 45-condition grid search changed from roughly 35.9 minutes in the original CPU-heavy implementation to 115 seconds after moving the repeated fusion grid to CUDA, then to 16 seconds after moving JSD/signal computation to CUDA as well.

The canonical 75-condition, all-severity grid run completed in about 27 seconds.

The later CE and evaluation stages were also reworked to avoid repeated cache loads, repeated CPU JSD computation, and repeated work across methods.

## Refined spectral ablation

Protocol: 100 ImageNet classes, 15 ImageNet-C corruptions, all 5 severities, leave-one-corruption-family-out tuning, and disjoint tune/test halves. `mCA` is mean corrupted top-1 accuracy over the 75 corruption/severity conditions. `Neg.` is negative transfer: originally correct predictions broken by fusion.

### ResNet-18

| Ladder | Method | mCA | Gain vs original | NLL | ECE | Neg. | Clean acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed | grid | **33.225%** | +0.663 | 3.6936 | 0.1026 | 8.71% | 67.78% |
| fixed | CE | 32.755% | +0.194 | 3.6316 | **0.0529** | 9.14% | 67.74% |
| fixed | CE + view bias | 33.134% | +0.573 | **3.5866** | 0.0570 | **5.91%** | **67.96%** |
| fixed | CE unconstrained | 33.025% | +0.463 | 3.5869 | 0.0555 | 8.36% | 67.94% |
| geometric | grid | 33.202% | +0.640 | 3.7015 | 0.1034 | 8.02% | 67.42% |
| geometric | CE | 32.951% | +0.389 | 3.6172 | 0.0521 | 8.03% | 67.50% |
| geometric | CE + view bias | 33.133% | +0.571 | 3.5893 | **0.0536** | **6.18%** | 67.72% |
| geometric | CE unconstrained | **33.278%** | +0.716 | **3.5779** | 0.0555 | 8.37% | 67.64% |

The geometric unconstrained diagnostic has the highest ResNet point-estimate mCA. Among the canonical constrained methods, the fixed-grid configuration has the highest mCA. CE + view bias improves calibration and negative transfer but gives up a small amount of top-1 accuracy.

### DeiT-Tiny

| Ladder | Method | mCA | Gain vs original | NLL | ECE | Neg. | Clean acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed | grid | 44.099% | +0.132 | 2.9352 | 0.0943 | 6.37% | 69.18% |
| fixed | CE | 44.065% | +0.098 | 2.8576 | 0.0953 | 5.81% | 69.16% |
| fixed | CE + view bias | **44.593%** | **+0.626** | **2.8236** | 0.0928 | 3.28% | **70.02%** |
| fixed | CE unconstrained | 43.922% | -0.045 | 2.8592 | 0.0929 | 5.98% | 69.16% |
| geometric | grid | 44.326% | +0.359 | 2.9133 | 0.0998 | 5.05% | 69.22% |
| geometric | CE | 44.128% | +0.161 | 2.8547 | 0.0911 | 5.49% | 69.20% |
| geometric | CE + view bias | 44.493% | +0.526 | 2.8249 | 0.0920 | **3.25%** | 69.90% |
| geometric | CE unconstrained | 43.975% | +0.008 | 2.8559 | **0.0905** | 5.72% | 69.20% |

For DeiT-Tiny, fixed-ladder CE + view bias is the strongest canonical spectral configuration by top-1 accuracy, NLL, clean accuracy, and low negative transfer.

## Paired spectral-selection bootstrap

Each comparison uses 5,000 paired bootstrap resamples.

| Model | Comparison: fixed CE+bias minus ... | Delta (pts) | 95% CI | P(delta <= 0) |
|---|---|---:|---:|---:|
| ResNet-18 | fixed grid | -0.090 | [-0.152, -0.030] | 0.9974 |
| ResNet-18 | geometric CE+bias | +0.002 | [-0.037, +0.041] | 0.4640 |
| ResNet-18 | geometric unconstrained diagnostic | -0.144 | [-0.202, -0.089] | 1.0000 |
| DeiT-Tiny | fixed grid | +0.494 | [+0.420, +0.567] | 0.0000 |
| DeiT-Tiny | geometric CE+bias | +0.101 | [+0.069, +0.131] | 0.0000 |
| DeiT-Tiny | geometric grid | +0.267 | [+0.198, +0.335] | 0.0000 |

The spectral configurations carried forward are:

- ResNet-18: fixed ladder + grid fusion.
- DeiT-Tiny: fixed ladder + CE + view bias.

The unconstrained ResNet result is retained as a diagnostic rather than promoted to the primary method.

## Generated outputs

Large caches, result folders, logs, figures, IDE files, and virtual environments should remain local and be ignored by Git. The source branch should contain code, tests, documentation, and small configuration/metadata files only.
