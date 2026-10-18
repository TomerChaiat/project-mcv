# Scripts

Every entry point, in pipeline order. Run from the repository root
(`python scripts/<name>.py ...`); every script has `--help`. Outputs go to
`results/` (timestamped JSON/CSV) and `figures/`, both gitignored.

**The simplest way to reproduce everything is one command:**

```bash
bash scripts/run_full_pipeline.sh
```

It runs the steps below in order (see its header for phases and options).
The individual commands are listed so any step can be re-run or changed.

## 1. Data

```bash
python scripts/build_subset.py          # choose the fixed 100-class subset -> data/subset_classes.json
python scripts/fetch_clean_images.py    # clean ImageNet-val for those classes -> data/imagenet/val/
python scripts/fetch_imagenet_c.py      # ImageNet-C for those classes -> data/imagenet-c/ (resumable)
python scripts/match_clean_names.py     # clean image <-> ImageNet-C name, by content -> data/clean_name_map.json
python scripts/fetch_imagenet_c.py --tarballs extra   # optional: the 4 held-out extra corruptions
```

`data/clean_name_map.json` is committed; `match_clean_names.py` only needs to
be re-run if the clean images are re-downloaded.

## 2. Profile caches

Every image through every view of the frozen model; softmax outputs stored as
float16 (`cache_<name>/<model>/<corruption>/<severity>.npz`).

```bash
python scripts/cache_profiles.py --models resnet18 deit convnext --ladder fixed --cache-dir cache_fixed   # 8 spectral views
python scripts/cache_spatial_profiles.py --models resnet18 deit convnext --cache-dir cache_spatial        # 8 crop/flip views
python scripts/build_combined_cache.py --virtual --models resnet18 deit convnext                         # 15-view profile, no GPU, no copy
python scripts/cache_profiles.py --models resnet18 deit --ladder geometric --cache-dir cache_geometric    # spacing ablation
python scripts/cache_spatial_profiles.py --models resnet18 deit convnext --corner-mode pad --cache-dir cache_spatial_pad   # padded control
# extra corruptions: the two caching commands above with --corruptions speckle_noise gaussian_blur spatter saturate
```

Geometry: ImageNet-C images are 224x224 and are used as-is; clean images get
resize(256) + centre-crop(224). Spatial views are the input, its flip, four
196-px corner crops resized to 224, and two of their flips. View 0 is
bit-identical in the spectral and spatial caches. The resize slightly smooths
the corner views; `cache_spatial_profiles.py --corner-mode pad` builds a
version without it (224 windows of the input reflect-padded to 256), see
RESULTS.md "Crop geometry check".

## 3. Tune, select, evaluate

`--tag` names the outputs; use `cache_fixed_lofo` / `cache_combined_lofo`
for the two profiles. The combined profile needs `--ladder-len 8` everywhere
(the stability signal is only defined along the 8-view spectral ladder).

```bash
python scripts/tune_fusion.py    --models resnet18 deit convnext --cache-dir cache_fixed --tag cache_fixed_lofo --out results
python scripts/tune_fusion_ce.py --models resnet18 deit convnext --cache-dir cache_fixed --tag cache_fixed_lofo --out results \
                                 --grid-tuning auto --variants base bias unconstrained bias_only
python scripts/select_method.py  --models resnet18 deit convnext --tag cache_fixed_lofo --results results
python scripts/evaluate.py       --models resnet18 --cache-dir cache_fixed --out results --tag cache_fixed_lofo_final \
                                 --tuning results/tuning_resnet18_cache_fixed_lofo_final_<time>.json
python scripts/evaluate_many.py  --model resnet18 --cache-dir cache_fixed --out results \
                                 --method cache_fixed_lofo_grid results/tuning_grid_resnet18_cache_fixed_lofo_<time>.json  # ...one --method per variant
```

- `tune_fusion.py`: grid over alpha/beta/gamma/tau/standardize/m/agreement,
  leave-one-family-out (LOFO) on the tune half.
- `tune_fusion_ce.py`: fits the same weighting by cross-entropy; variants
  `base`, `bias` (+ a per-view bias), `unconstrained` (signed coefficients),
  `bias_only` (per-view bias, no per-image signal). `--signals C S` fits
  only the listed signal coefficients and holds the others at 0
  (`--signals` alone: none); the method name gets a `_sig<subset>` suffix.
- `select_method.py`: picks the method per model from **tune-half** accuracy
  on each fold's *held-out* family (every condition scored with the config
  tuned without its family, as evaluate.py does on the test half) and freezes
  it as `tuning_<model>_<tag>_final_<time>.json`. The in-family fit accuracy
  is printed alongside for reference only. It stops with an error if any
  condition is missing from the cache.
- `evaluate.py`: scores the frozen method on the test half (each corruption
  with the config tuned without its family; clean with the global config),
  plus every baseline and a calibrated-vanilla reference.
- `evaluate_many.py`: same, for several tuning files at once (the ablation table).
- `evaluate_tta.py`: the spatial-TTA baseline alone.

The crop/flip views can be tuned the same way, for a like-for-like
comparison with the low-pass views (pipeline phase `spatialfit`, ~45 min):
the same commands on `--cache-dir cache_spatial --tag cache_spatial_lofo
--ladder-len 1` (the spatial views have no filter ladder, so the stability
signal is 0 and the fit uses confidence, agreement and a per-view weight),
then `bootstrap_pairwise.py` with the tuned spectral and tuned spatial
configs (`--b-ladder-len 1`).

## 4. Statistics

Paired cluster bootstrap over images (`src/bootstrap.py`): one draw of test
images per replicate, applied to all 75 conditions.

```bash
python scripts/bootstrap_ci.py --models resnet18 deit convnext --cache-dir cache_fixed --tag cache_fixed_lofo_final \
                               --results-dir results --spatial-cache cache_spatial --out results   # fused vs original / spatial TTA, every baseline vs original
python scripts/bootstrap_pairwise.py --model resnet18 \
    --a-mode fused --a-cache cache_fixed --a-tuning <tuning A> \
    --b-mode fused --b-cache cache_fixed --b-tuning <tuning B> --out results/<name>.json             # any two predictors
```

`bootstrap_pairwise.py --b-mode spatial` is the equal-weight average over all
views of `--b-cache` (used for "selected vs equal average").

## 5. Analyses

```bash
python scripts/spectral_energy.py --n-per-condition 500     # three spectral measures per view (energy removed / changed / added) vs that view's gain
python scripts/profile_shape.py                             # prediction drift down the ladder, per model and family
python scripts/signal_ceiling.py --models resnet18 deit convnext --cache-dir cache_fixed --no-gbm   # learned per-image scorer of C/A/S
python scripts/measure_overhead.py --models resnet18 deit convnext                               # inference cost vs 1 forward pass
python scripts/spatial_crop_control.py                      # resize-196 vs reflect-padded corner crops (needs cache_spatial_pad/, phase padctl)
python scripts/evaluate_extra.py                            # the 4 held-out extra corruptions with the frozen configs (phase extra)
```

`evaluate_extra.py` never tunes: each extra corruption (speckle noise,
Gaussian blur, spatter, saturate) is scored with the selected method's
leave-one-family-out config of its family (`src.splits.EXTRA_FAMILY`, kept
out of `CORRUPTION_FAMILY` so no tuner sees them). `spectral_energy.py
--corruptions ...` runs the spectral-energy measurement on them.

## 6. Figures and tables

```bash
python scripts/make_paper_figures.py    # results/ -> figures/ (PDF + 300 DPI PNG + LaTeX tables)
python scripts/make_pipeline_figure.py  # method schematic -> figures/pipeline.pdf + .png (optional out dir argument)
```

`make_paper_figures.py` reads the newest result file per model and profile,
so re-running any upstream step and then this script keeps figures and
numbers in sync. `make_pipeline_figure.py` needs no results or data: it
draws the method overview (spectral and spatial views, fusion) on a stock
sample image with synthetic noise, for illustration only.

## 7. Single-image inspection

```bash
python scripts/profile_single_image.py --image data/imagenet-c/gaussian_noise/3/<wnid>/<img>.JPEG
python scripts/show_filter_views.py    --image <same image>
python scripts/verify_filter_energy.py --image <same image>
```

## 8. Other view families and signals

Alternatives to the method's design choices, each tested against it
(pipeline phase `views`, ~2.5 GPU-h; RESULTS.md "Other view families and
signals"):

```bash
python scripts/view_family_comparison.py                    # low-pass vs band-pass vs notch views, single views, 3 models
python scripts/view_family_comparison.py --set ladders \
    --out-dir results/ladder_compare --out results/ladder_comparison.json   # fixed vs per-image energy-adaptive ladder
python scripts/decorrelation_diagnostic.py --method ce_bias_only   # what the learned per-view weights track (also --method ce_bias)
python scripts/experiment_ambiguity.py                      # a 4th trust signal (disagreement with the view average)
```

- `view_family_comparison.py` filters the images on the fly (no cache
  needed): 15 corruptions at severity 3 plus clean, all images. The best
  setting of each view family is chosen on the tune half and reported on the
  test half. Per-condition results are saved in `--out-dir` as they finish,
  so an interrupted run resumes where it stopped.
- `decorrelation_diagnostic.py` and `experiment_ambiguity.py` read
  `cache_combined/` and the CE tuning JSONs in `results/`, so they need the
  `tune12+cache3` / `tune3` phases first.
- **GPU memory:** `view_family_comparison.py` (all three models loaded) and
  `experiment_ambiguity.py` each use ~10-12 GB. Run them one at a time, not
  next to another GPU job, on a 24 GB card.


## 9. Accuracy vs. number of views

```bash
python scripts/view_budget_ablation.py               # budgets 2/4/8/15, 3 models -> results/view_budget/, figures/view_budget.*
python scripts/view_budget_ablation.py --plot-only   # redraw the figure from results/view_budget/view_budget.json
```

- Reads only `cache_combined/` and the frozen combined configurations
  (`results/tuning_<model>_cache_combined_lofo_final_*.json`); no model
  inference. Needs the `tune12+cache3` / `tune3` phases first. ~10 min per
  model on the A10 (~6 GB of GPU memory).
- Subset rule (per leave-one-family-out fold, tune half only): start from the
  original and greedily add the view that most improves the plain-average
  accuracy, averaged over the fold's tuning corruptions x 5 severities; ties
  go to the lower view index. The fold's subsets are the first 2/4/8 views of
  that order; 15 is all views.
- Fusion: each model's selected combined method, re-fit per fold on the
  subset with the `tune_fusion_ce.py` fit/scale/temperature search, warm
  started from the fold's 15-view CE direction. 15 views uses the frozen
  configuration as is and must reproduce `evaluate.py`'s combined accuracy
  (the script stops otherwise); re-fitting all 15 views is also run as a
  check of the re-fit (`--skip-refit-check` to skip).

## 10. What each reliability signal contributes

Pipeline phase `signals` (~30 min GPU, after the tuning phases):

```bash
# CE + bias refit with one signal subset (repeated for every subset and model by the phase)
python scripts/tune_fusion_ce.py --models resnet18 --cache-dir cache_fixed --out results/signal_ablation \
    --tag cache_fixed_lofo --grid-tuning results/tuning_grid_resnet18_cache_fixed_lofo_<time>.json \
    --variants bias --agreement-modes all --signals C S
python scripts/signal_ablation.py                 # AUROC per signal + test-half accuracy per subset
python scripts/signal_ablation.py --plot-only     # tables + figure from the saved JSONs
```

- Part A (no fitting): AUROC of each raw signal (C, A to all views, A to the
  original, S with m=1) for "this low-pass view is correct", and of the
  signal minus the original view's for "this view is correct and the
  original is wrong". Within-condition AUROC, averaged over
  conditions and the 7 low-pass views; cluster bootstrap over images.
- Part B: the subset fits scored on the test half like `evaluate.py`, with
  paired bootstraps against {C, A, S} and against bias only. The script also
  re-scores the pipeline's own CE + bias and bias-only fits and stops if they
  do not reproduce `evaluate_many.py`'s numbers.
- Outputs: `results/signal_ablation/signal_auroc.json`, `signal_subsets.json`,
  the subset fits; `figures/table_signal_auroc.tex`,
  `figures/table_signal_subsets.tex`, `figures/signal_auroc.pdf/.png`.

## Conventions

- `--cache-dir` picks the profile cache; `--tag` suffixes output filenames so
  runs on different caches never collide.
- The tune/test split is a hash of the image id (`src/splits.py`), identical
  for every corruption, severity and the clean set, and independent of RNG
  state. Defaults `--split-seed 0 --frac-tune 0.5` everywhere.
- Output files are timestamped; scripts always use the newest matching file.
