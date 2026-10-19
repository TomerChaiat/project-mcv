# `scripts/`

Everything runnable lives here - the main README's three-command quick start
is the short version of this file. Run everything from the repo root
(`python scripts/<name>.py ...`); most write timestamped output to a local
`results/` folder. For what these produced, see
[`../RESULTS.md`](../RESULTS.md); for the library code underneath them,
[`../src/`](../src/README.md).

## 1. Get the data (no GPU needed)

```bash
python scripts/build_subset.py          # choose the fixed 100-class subset -> data/subset_classes.json
python scripts/fetch_clean_images.py    # clean ImageNet-val for those classes -> data/imagenet/val/
python scripts/fetch_imagenet_c.py      # ImageNet-C for those classes -> data/imagenet-c/ (safe to stop/re-run)
```
Full details, including the Zenodo throttling caveat: [`../data/README.md`](../data/README.md).

## 2. Build the profile caches (the GPU step, once each)

```bash
python scripts/cache_profiles.py                                         # spectral, energy ladder -> cache/
python scripts/cache_profiles.py --ladder fixed --cache-dir cache_fixed  # spectral, fixed ladder -> cache_fixed/ (the one the results use)
python scripts/cache_spatial_profiles.py                                 # 8-view crop+flip TTA -> cache_spatial/
python scripts/build_combined_cache.py                                   # merge cache_fixed + cache_spatial -> cache_combined/ (NO GPU - pure numpy concat)
```
~40 min/model on an A10 for either `cache_profiles.py` or
`cache_spatial_profiles.py`; skips conditions already on disk unless
`--overwrite`. `build_combined_cache.py` needs the first and third to exist
already.

Once a cache exists, everything below is plain NumPy - seconds to minutes,
safe to re-run as often as you like while you change parameters.

## 3. Tune + evaluate

```bash
python scripts/tune_fusion.py --cache-dir cache_fixed                                    # grid-search alpha/beta/gamma/tau/standardize/m/agreement, LOFO
python scripts/evaluate.py    --cache-dir cache_fixed --tuning auto                      # score every condition against the tuned configs
python scripts/evaluate.py    --alpha 1 --beta 0.5 --gamma 0.5 --tau 0.1                 # or try one setting by hand (skips tuning entirely)
```
For the combined cache, both need `--ladder-len 8` (see `src/README.md`'s
note on `ladder_len`):
```bash
python scripts/tune_fusion.py --cache-dir cache_combined --ladder-len 8
python scripts/evaluate.py    --cache-dir cache_combined --ladder-len 8 --tuning auto
```
`evaluate.py --tuning auto` picks the newest matching `results/tuning_*.json`;
pass a specific path to pin one. Both default to scoring only the held-out
test half (`--eval-half all` scores everything, e.g. for a sanity check).

## 4. The spatial-TTA baseline

The natural question once fusion beats the plain classifier: does *any*
8-view ensemble do that, or is the reliability weighting earning its keep?
This scores a plain crop+flip ensemble the same way, for comparison.

```bash
python scripts/evaluate_tta.py    # score cache_spatial/ on the same held-out half as evaluate.py -> results/tta_<model>_<time>.csv
```

## 5. Statistical rigor and the ceiling analysis

Two follow-up questions once the headline numbers were in: are these deltas
real, and is the remaining gap to the oracle a tuning problem or an
information problem?

```bash
python scripts/bootstrap_ci.py --cache-dir cache_fixed    --tag cache_fixed              # 95% CIs: fused-vs-orig, fused-vs-spatial-TTA
python scripts/bootstrap_ci.py --cache-dir cache_combined --tag cache_combined --ladder-len 8
python scripts/signal_ceiling.py --no-gbm --agreement origin                             # best any scorer of C/A/S could do, not just our linear grid
python scripts/measure_overhead.py                                                       # wall-clock: 1 forward pass vs. the spectral/combined profile
```
`bootstrap_ci.py` resamples images *within* each held-out condition, paired
across the predictors being compared, so the CI targets the delta directly
(see its docstring). `signal_ceiling.py` fits a classifier on
`(C_k, A_k, S_k) -> "is this view correct"` per LOFO fold and fuses with its
predicted reliability instead of the hand-tuned linear rule; `--no-gbm` skips
the slow gradient-boosted-trees check once it's confirmed to tie logistic
regression (details in `RESULTS.md`).

## 6. Look at one image

For building intuition or debugging a specific case, rather than aggregate numbers:

```bash
python scripts/profile_single_image.py --image data/imagenet-c/gaussian_noise/3/<wnid>/<img>.JPEG
python scripts/show_filter_views.py    --image <same image>
python scripts/verify_filter_energy.py --image <same image>
```

## 7. Figures

```bash
python scripts/make_report_figures.py    # -> local figures/ folder, 300 DPI, not tracked in git
```
Numbers are hard-coded from specific `results/*.json` files (named in each
figure function's docstring) rather than re-derived live - re-run the
upstream scripts and update the hard-coded numbers if the caches change.

## Naming conventions

- `--cache-dir` picks which profile cache a script reads (`cache/`,
  `cache_fixed/`, `cache_spatial/`, `cache_combined/`); defaults to `cache/`.
- `--tag` (or the auto-derived cache-dir name) suffixes output filenames so
  runs against different caches never overwrite each other, and so
  `evaluate.py --tuning auto` doesn't cross-pick a config tuned on a
  different cache.
- Everything downstream of the two caching scripts is deterministic given
  the same `--split-seed`/`--frac-tune` (defaults 0/0.5 everywhere) - the
  tune/test split is a hash of the image filename, not RNG state.
