# Final comprehensive evaluation

This folder runs the final, leakage-safe comparison requested for the report.
It does **not** reuse leave-one-family-out parameters at test time.

## What it does

1. Validates that `cache_combined_recheck` and `cache_spatial` contain all 76 records per model (75 ImageNet-C conditions + clean), and that the combined profile has 15 views.
2. Tunes on the **entire established training/tuning split**: all 15 corruptions x all 5 severities, using the tune half only (`frac_tune=0.5`, split seed 0).
3. Uses one global final configuration per architecture:
   - ResNet-18: global grid fusion.
   - DeiT-Tiny: global CE + learned view bias, initialized from the global grid result.
4. Strips all LOFO/held-out configs from the final tuning JSON before evaluation. This guarantees the held-out test evaluation uses only the globally trained parameters.
5. Evaluates the held-out test half for **every corruption and every severity** against the vanilla classifier (`orig_acc`).
6. Runs paired bootstrap statistics for final vs vanilla (and also records spatial-TTA as an extra reference).
7. Benchmarks actual inference compute for vanilla 1-view vs the final 15-view method. Disk/JPEG I/O is excluded from both sides; both start from the same already-loaded image tensor.
8. Creates a report folder with exhaustive CSV and Markdown tables.

## Run

Place/extract this `final_comprehensive_evaluation` folder directly inside the repository root, then from the repository root run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\final_comprehensive_evaluation\run_all.ps1
```

Defaults assume the complete caches already created in this project:

```text
cache_combined_recheck
cache_spatial
```

The run creates a unique ignored result directory:

```text
results/final_comprehensive_YYYYMMDD_HHMMSS/
```

## Output layout

```text
results/final_comprehensive_.../
  input_validation.json
  tuning_raw/
  tuning_global/
    resnet18_grid_global.json
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

`all_conditions.csv` is the direct answer to the main comparison request: 2 models x 15 corruptions x 5 strengths = 150 rows, each with vanilla accuracy, final-method accuracy, and the gain/loss in percentage points.

## Important protocol note

"Tune on the entire train data" here means the entire **pre-existing tune/train half** across all 75 corruption/severity cells. The held-out test half is intentionally not included in tuning. Using the test half for parameter fitting would invalidate the final comparison.

## Optional switches

For a quicker diagnostic run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\final_comprehensive_evaluation\run_all.ps1 `
    -BootstrapResamples 2000 `
    -BenchmarkReps 5
```

If the benchmark batch size does not fit VRAM, reduce it:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\final_comprehensive_evaluation\run_all.ps1 `
    -BenchmarkBatchSize 16
```

The scientific evaluation itself is cache-based and is unaffected by the benchmark batch size.
