# Spectral Stability Profiles for Robust Test-Time Image Classification

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![Models](https://img.shields.io/badge/models-ResNet--18%20%7C%20DeiT--Tiny-44cc11.svg)
![Benchmark](https://img.shields.io/badge/benchmark-ImageNet--C-f39c12.svg)

> Modern Computer Vision final project - Tomer Chaiat, Omer Yahav, Hadar Etzion, Gil Yashayev, Saar Misk

## Overview

When a photo is corrupted - sensor noise, motion blur, fog, low contrast - a
frozen classifier often gets it wrong.  
This project asks whether it can recover
at test time using nothing but its own reactions to that one image: no
retraining, no labels, no access to the corruption, no weight updates.

The idea: take the test image, blur it a little, blur it more, blur it a lot,
and run every version through the same classifier. That gives a short
trajectory of predictions - a **spectral stability profile**. Where the
classifier is confident and its answer holds steady as the blur changes, trust
that view; where it flip-flops, don't.  
Then average the views with those trust weights.

### 🗝️ Key Ideas

- **Frozen classifier** - no retraining, no corruption labels, no source data, no parameter updates.
- **Frequency views** - the image low-pass filtered at a ladder of strengths (two ladders compared: per-image energy-adaptive, and fixed absolute cutoffs).
- **Three trust signals** - confidence, agreement across views, and stability along the ladder, combined into softmax fusion weights.
- **Combining view families** - the same fusion mechanism also arbitrates across a second, structurally different view family (spatial-TTA crop/flip views), for free since both are already cached, and this beats either family alone - see [`RESULTS.md`](RESULTS.md).
- **Honest evaluation** - coefficients tuned leave-one-corruption-family-out, on a disjoint half of the images, with bootstrap confidence intervals on every headline delta.
- **Two architectures** - ResNet-18 and DeiT-Tiny, fed identical pixels, to test whether a CNN and a ViT behave differently.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A GPU for the one-time caching step (everything after runs on CPU)
- ~15 GB free disk for the data and the profile cache

### Installation & Setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

PYTHONPATH=. python test/test_models.py      # pull the pretrained weights
PYTHONPATH=. python test/test_analysis.py    # should print "all passed"
```

Then build the dataset - a 100-class slice of ImageNet-C plus the matching
clean validation images. Full instructions and checks are in
[`data/README.md`](data/README.md); the short version:

```bash
python scripts/build_subset.py          # choose the 100 classes
python scripts/fetch_clean_images.py    # clean images
python scripts/fetch_imagenet_c.py      # corrupted images (safe to stop and re-run)
```

## 💻 Usage

The pipeline is split so the GPU work happens once and everything after is
fast NumPy you can re-run while changing parameters:

```bash
python scripts/cache_profiles.py --ladder fixed --cache-dir cache_fixed  # GPU, once: every profile -> cache_fixed/  (~40 min/model)
python scripts/tune_fusion.py --cache-dir cache_fixed                    # grid-search the fusion coefficients, leave-one-family-out
python scripts/evaluate.py --cache-dir cache_fixed --tuning auto         # score every condition -> results/eval_<model>_<time>.csv
```

Every other script - the spatial-TTA baseline, the combined view-family
profile, statistical rigor, the signal-ceiling analysis, single-image
inspection, figure generation - is catalogued with copy-pasteable commands in
[`scripts/README.md`](scripts/README.md).

## 📊 Results

Headline: fusing across **both** a spectral low-pass ladder and a
spatial-TTA ensemble (crop + flip) beats either view family alone, on both
architectures, with 95% confidence:

| | ResNet-18 | DeiT-Tiny |
|---|:--:|:--:|
| original -> combined fusion | 32.6% -> **34.9%** (+2.4 pts) | 44.0% -> **45.9%** (+2.0 pts) |
| vs. spatial-TTA alone (95% CI) | +0.47 [+0.39, +0.56] | +0.39 [+0.31, +0.48] |

The full story - spectral-only vs. spatial-only vs. combined, every baseline
the proposal committed to, the per-view mechanism, and a ceiling analysis
asking whether the remaining oracle gap is a tuning problem or an information
problem - is in **[`RESULTS.md`](RESULTS.md)**, plotted versions included
(`scripts/make_report_figures.py`).

## 🏗️ Architecture

### How it works

For a test image `x` we build `K + 1` views:

```
x_0 = x                          the original, untouched
x_k = F^-1( F(x) · M_k )          k = 1 ... K
```

`F` is the 2-D DFT and each `M_k` is a smooth Gaussian low-pass mask. The masks
get progressively stronger; two cutoff schedules are implemented (see
`RESULTS.md` for why the fixed one is used for the headline numbers):
per-image energy-adaptive (`spectral_views`, keep a target fraction of that
image's own spectral energy) and fixed absolute cutoffs
(`spectral_views_fixed`), both in `src/cache.py`.

Every view goes through the **same frozen classifier**, giving the profile
`S(x) = (p_0, ... , p_K)`. From it we read three trust signals per view:

| signal | what it measures | formula |
|---|---|---|
| `C_k` confidence | how peaked the prediction is | `max_c p_k(c)` |
| `A_k` agreement | how close this view is to all the others | `-(1/K) Σ_{j≠k} JSD(p_k, p_j)` |
| `S_k` stability | how close this view is to its ladder neighbours | `-mean_{j∈N(k)} JSD(p_k, p_j)` |

`JSD` is the base-2 Jensen-Shannon divergence, so `C_k ∈ [0, 1]` and
`A_k, S_k ∈ [-1, 0]`. The three combine into a score and softmax weights, and
the views are averaged:

```
r_k = α·C_k + β·A_k + γ·S_k
w   = softmax(r / τ)
p*  = Σ_k w_k · p_k
```

`p*` is judged against the original view alone, an equal-weight average, a
confidence-only weighting, and a per-image oracle (right if *any* view is - the
ceiling for any weighting scheme). Full results: [`RESULTS.md`](RESULTS.md).

### Project Structure

```
src/       library code - the profile maths, metrics, caching, models     -> src/README.md
scripts/   every entry point - data fetch, GPU caching, tuning, analysis  -> scripts/README.md
data/      the ImageNet-C subset + clean val, and how to (re)fetch it     -> data/README.md
test/      unit tests for the profile maths (no GPU or data needed)
```

Generated locally by the scripts above, gitignored (not in this repo -
regenerate them any time):

```
cache/, cache_fixed/, cache_spatial/, cache_combined/   the profile caches
results/                                                evaluation output
figures/                                                report-ready PNGs
```

## ⚠️ Gotchas

- **Class indices stay 0-999.** The subset keeps original ImageNet indices so
  the pretrained heads line up. `src/imagenet_labels.py` reads
  `data/subset_classes.json`; never re-index to 0-99.
- **Filter in `[0, 1]` space, before normalisation.** Order: resize(256) ->
  centre-crop(224) -> to float -> low-pass -> ImageNet-normalise. ImageNet-C
  images already carry the corruption; we do not re-corrupt.
- **Both architectures see identical pixels** - same geometry, only the
  normalisation constants differ.
- **Standardise the signals before combining.** Confidence is in `[0, 1]`,
  agreement and stability in `[-1, 0]`, so without the per-image z-score
  (`standardize`, on by default) alpha, beta, gamma are not comparable. Whether
  to standardise is itself part of the tuning grid.
- **Tuning never sees the family it is scored on**, and tunes on a disjoint
  image half.
- **GPU env**: `requirements.txt` needs an env with CUDA torch; see
  `scripts/README.md` if you have more than one Python env on the machine.


## 🐛 Troubleshooting

- **`no results/tuning_<model>_*.json found`** - run `scripts/tune_fusion.py`
  before `scripts/evaluate.py --tuning auto`.
- **`evaluate.py` / `tune_fusion.py` skip everything** - they read `cache/`; run
  `scripts/cache_profiles.py` first (or pass the matching `--cache-dir`).
- **ImageNet-C download is slow or drops mid-way** - expected, the source
  server throttles. Re-run `scripts/fetch_imagenet_c.py`; it resumes.
- **macOS `CERTIFICATE_VERIFY_FAILED`** when pulling model weights - run the
  Python cert installer, e.g.
  `open "/Applications/Python 3.12/Install Certificates.command"`.

## 📎 Reference

ImageNet-C: Hendrycks and Dietterich, *Benchmarking Neural Network Robustness to
Common Corruptions and Perturbations*, ICLR 2019
([Zenodo 2235448](https://zenodo.org/records/2235448)).
