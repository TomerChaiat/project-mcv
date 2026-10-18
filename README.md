# Spectral Stability Profiles for Robust Test-Time Image Classification

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![Models](https://img.shields.io/badge/models-ResNet--18%20%7C%20DeiT--Tiny%20%7C%20ConvNeXt--Tiny-44cc11.svg)
![Benchmark](https://img.shields.io/badge/benchmark-ImageNet--C-f39c12.svg)

> Modern Computer Vision final project - Tomer Chaiat, Omer Yahav, Hadar Etzion, Gil Yashayev, Saar Misk

## Overview

When a photo is corrupted - sensor noise, blur, fog, low contrast - a frozen
image classifier often gets it wrong. This project asks whether a
**spectral stability profile** - the classifier's predictions on
progressively low-pass-filtered copies of the same image - can tell, image
by image, which copy to trust, and so improve robustness at test time:
no retraining, no labels, no knowledge of the corruption.

Each test image is low-pass filtered at several strengths, every copy goes
through the same frozen classifier, and the predictions are fused, weighted
by three per-view trust signals read from the profile: confidence,
agreement, and stability across neighbouring filter strengths. The same
fusion is also run together with standard crop/flip test-time
augmentation (TTA).

## Main findings

- **Low-pass views improve corruption robustness** with a frozen model:
  +1.98 / +1.26 / +0.73 pts mean corrupted accuracy for ResNet-18 / DeiT-Tiny /
  ConvNeXt-Tiny. Tuned the same way and with the same number of views, they
  beat crop/flip TTA for ResNet-18 and match it for DeiT and ConvNeXt;
  combining both kinds of views works best for all three.
- **They help on corruptions that add high-frequency structure** - noise,
  pixelation, JPEG - and not on blur or weather, where the detail is
  already gone. On unseen corruption types the gain holds for noise only.
- **The stability signal carries the per-image benefit, but only for the
  noise-sensitive model.** For ResNet-18, weighting views by stability alone
  adds +0.51 pts over fixed per-view weights, slightly more than all three
  signals together (+0.46). For DeiT and ConvNeXt, no signal subset adds
  more than +0.04 pts over fixed per-view weights.
- **Not CNN vs. Transformer:** ConvNeXt, a CNN, behaves like DeiT, not like
  ResNet-18. The split is noise-sensitive vs. noise-robust, likely due to the
  training recipe.

Numbers, confidence intervals and all comparisons: **[`RESULTS.md`](RESULTS.md)**.

## Method

For a test image `x`, build `K + 1` views:

```
x_0 = x                          the original
x_k = F^-1( F(x) · M_k )          k = 1 ... 7
```

`F` is the 2-D DFT and `M_k` a Gaussian low-pass mask with cutoffs
`(0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.12)`. Every view goes through the same
frozen classifier, and three trust signals are read per view:

| signal | what it measures | formula |
|---|---|---|
| `C_k` confidence | how peaked the prediction is | `max_c p_k(c)` |
| `A_k` agreement | how close this view is to all the others | `-(1/K) Σ_{j≠k} JSD(p_k, p_j)` |
| `S_k` stability | how close this view is to its neighbours on the ladder | `-mean_{j∈N(k)} JSD(p_k, p_j)` |

They are combined into fusion weights:

```
r_k = α·C_k + β·A_k + γ·S_k (+ an optional per-view bias b_k)
w   = softmax(r / τ)
p*  = Σ_k w_k · p_k
```

The coefficients are tuned on one half of the images, leaving out each
corruption family in turn, and everything is evaluated on the other half.
Two view sets are used: **spectral** (the 8 views above) and **combined**
(those plus 7 crop/flip views, 15 in total).

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- GPU
- ~6 GB for the data, ~45 GB for the prediction caches.


### Installation

```bash
python -m venv .venv && source .venv/bin/activate      # or a conda env
python -m pip install -r requirements.txt               # install a CUDA build of torch first
python -m pytest -q test                                # unit tests, no GPU or data needed
```

### Data

A 100-class subset of ImageNet-C plus the matching clean validation images.
Full instructions: [`data/README.md`](data/README.md).

```bash
python scripts/build_subset.py          # the 100 classes (already in the repo)
python scripts/fetch_clean_images.py    # clean images -> data/imagenet/val/
python scripts/fetch_imagenet_c.py      # corrupted images -> data/imagenet-c/ (slow, resumable)
```

## 💻 Usage

Run everything end to end:

```bash
bash scripts/run_full_pipeline.sh                           # results/ and figures/, logs in results/logs/
PHASES="extras figures" bash scripts/run_full_pipeline.sh   # only some phases (see the script header)
PY=/path/to/python bash scripts/run_full_pipeline.sh        # choose the Python env
```

Phases: `cache12` -> `build` -> `tune12+cache3` -> `tune3` -> `geo` ->
`extras` -> `figures`, plus five optional ones: `spatialfit` (the crop/flip
views tuned the same way as the low-pass views, for a like-for-like
comparison), `padctl` (crop-geometry check), `extra` (ImageNet-C's 4
held-out corruptions; fetch them first with `fetch_imagenet_c.py --tarballs
extra`), `views` (other view families and signals; uses most of a 24 GB
GPU, so run it on its own) and `signals` (what each per-image trust signal
contributes).

Or step by step, for one model and view set:

```bash
python scripts/cache_profiles.py --models resnet18 --ladder fixed --cache-dir cache_fixed
python scripts/tune_fusion.py    --models resnet18 --cache-dir cache_fixed --tag cache_fixed_lofo --out results
python scripts/tune_fusion_ce.py --models resnet18 --cache-dir cache_fixed --tag cache_fixed_lofo --out results \
                                 --grid-tuning auto --variants base bias unconstrained bias_only
python scripts/select_method.py  --models resnet18 --tag cache_fixed_lofo --results results
python scripts/evaluate.py       --models resnet18 --cache-dir cache_fixed --out results --tag cache_fixed_lofo_final \
                                 --tuning results/tuning_resnet18_cache_fixed_lofo_final_<time>.json
python scripts/bootstrap_ci.py   --models resnet18 --cache-dir cache_fixed --tag cache_fixed_lofo_final \
                                 --results-dir results --out results
python scripts/make_paper_figures.py
```

Every script and its options: [`scripts/README.md`](scripts/README.md).

## 🏗️ Project Structure

| folder | contents | details |
|---|---|---|
| `src/` | library code: signals, fusion, metrics, caching, models | [src/README.md](src/README.md) |
| `scripts/` | entry points: data, caching, tuning, evaluation, figures | [scripts/README.md](scripts/README.md) |
| `data/` | dataset definition and download instructions | [data/README.md](data/README.md) |
| `test/` | unit tests | [test/README.md](test/README.md) | |

Generated locally (gitignored): `cache_*/` (prediction caches), `results/`
(JSON/CSV outputs and logs), `figures/` (PDF/PNG figures and LaTeX tables).

## ⚠️ Notes

- **Class indices stay 0-999** so the pretrained heads line up;
  `data/subset_classes.json` maps the 100 classes.
- **Filtering happens in `[0, 1]` pixel space, before normalisation.**
  ImageNet-C images are 224x224 and used as-is; clean images get
  resize(256) -> centre-crop(224), the geometry ImageNet-C was made with.
- **All three models see identical pixels**; only the normalisation
  constants differ.
- **Tuning never sees the corruption family or the images it is scored on.**

## 🐛 Troubleshooting

- **"missing cache" / every condition skipped** - build the cache for that
  `--cache-dir` first (or run the `cache12` / `tune12+cache3` phases).
  `select_method.py` stops with this error rather than choosing a method on
  part of the conditions.
- **`missing result: results/...` from `make_paper_figures.py`** - a pipeline
  phase hasn't run yet; the message names the file.
- **`ModuleNotFoundError: numpy` / `torch`** - wrong Python environment;
  activate it or pass `PY=...`.
- **Out of GPU memory** - lower `--batch-size` (caching) or `--config-batch`
  (`tune_fusion.py`).
- **Slow ImageNet-C download** - the server throttles; re-run
  `fetch_imagenet_c.py`, it resumes.

## 📎 Reference

ImageNet-C: Hendrycks and Dietterich, *Benchmarking Neural Network Robustness to
Common Corruptions and Perturbations*, ICLR 2019
([Zenodo 2235448](https://zenodo.org/records/2235448)).
