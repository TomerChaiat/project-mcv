# Comprehensive evaluation: CE + view bias for both models

This folder repeats the previous leakage-safe comprehensive evaluation, but the
**final fusion method is CE + learned view bias for both ResNet-18 and
DeiT-Tiny**.

The grid search is run only to provide the same controlled initialization used
by the CE tuner. Grid parameters are **not** used for final evaluation.

## Protocol

1. Validate that `cache_combined_recheck` and `cache_spatial` are complete: 75 ImageNet-C corruption/severity records plus clean (`76/76`) for each model, and 15 views in the combined profile.
2. Use the established deterministic 50/50 tune/test split (`split seed = 0`).
3. Tune over the **entire tune/train half across all 15 corruptions x all 5 severities**.
4. Run a global grid search for each model only as the CE optimizer warm start.
5. Fit **CE + mean-zero learned view bias** separately for ResNet-18 and DeiT-Tiny, including accuracy-based score-scale selection and output-temperature calibration.
6. Remove all leave-one-family-out configurations before the final evaluation, forcing one global CE+bias configuration per architecture.
7. Evaluate the untouched test half on every corruption and every severity against the vanilla original-view classifier.
8. Run paired bootstrap statistics for final CE+bias vs vanilla; spatial TTA is also recorded as a reference.
9. Benchmark actual compute from an already-loaded image tensor: vanilla 1-view inference versus the full 15-view CE+bias approach.
10. Generate the same exhaustive CSV and Markdown outputs as the previous comprehensive folder.

## Run

Extract/place this `final_comprehensive_ce_bias_both` folder directly inside the repository root. From the repository root run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\final_comprehensive_ce_bias_both\run_all.ps1
```

Defaults use:

```text
cache_combined_recheck
cache_spatial
CUDA
5000 bootstrap resamples
benchmark batch size 32
10 benchmark repetitions
```

## Output layout

A unique result directory is created, so it cannot overwrite the previous comprehensive experiment:

```text
results/final_comprehensive_ce_bias_both_YYYYMMDD_HHMMSS/
  input_validation.json
  tuning_raw/
    tuning_grid_resnet18_...json       # warm start only
    tuning_grid_deit_...json           # warm start only
    tuning_ce_bias_resnet18_...json
    tuning_ce_bias_deit_...json
  tuning_global/
    resnet18_ce_bias_global.json
    deit_ce_bias_global.json
  evaluation/
    eval_resnet18_...csv/json
    eval_deit_...csv/json
  statistics/
    bootstrap_resnet18.json
    bootstrap_deit.json
  timing/
    inference_timing.csv
    inference_timing.json
    stage_times.json
  report/
    FINAL_REPORT.md
    ALL_CONDITIONS.md
    all_conditions.csv
    by_corruption.csv
    by_family_severity.csv
    by_severity.csv
    overall_summary.csv
    report_manifest.json
```

`all_conditions.csv` contains exactly 150 held-out rows: 2 models x 15 corruptions x 5 severities. Each row gives vanilla accuracy, CE+bias final accuracy, gain/loss, NLL, ECE, negative transfer and positive transfer.

## Important leakage rule

"Tune on the entire train data" means all 75 corruption/severity cells of the
**pre-existing tune/train half**. The held-out test half is never used for
parameter selection. Fitting CE+bias on the test half would invalidate the
comparison.

## Optional switches

Quicker diagnostic run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\final_comprehensive_ce_bias_both\run_all.ps1 `
    -BootstrapResamples 2000 `
    -BenchmarkReps 5
```

If the timing benchmark does not fit VRAM:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\final_comprehensive_ce_bias_both\run_all.ps1 `
    -BenchmarkBatchSize 16
```

The scientific cache-based evaluation is unaffected by benchmark batch size.
