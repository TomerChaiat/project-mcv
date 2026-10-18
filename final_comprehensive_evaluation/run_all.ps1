param(
    [string]$CombinedCache = "cache_combined_recheck",
    [string]$SpatialCache = "cache_spatial",
    [string]$Device = "cuda",
    [int]$BenchmarkBatchSize = 32,
    [int]$BenchmarkReps = 10,
    [int]$BootstrapResamples = 5000,
    [switch]$SkipBootstrap,
    [switch]$SkipBenchmark
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Python virtual environment not found: $Py"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Repo "results\final_comprehensive_$Stamp"
$RawTune = Join-Path $RunRoot "tuning_raw"
$GlobalTune = Join-Path $RunRoot "tuning_global"
$EvalDir = Join-Path $RunRoot "evaluation"
$StatsDir = Join-Path $RunRoot "statistics"
$TimingDir = Join-Path $RunRoot "timing"
$ReportDir = Join-Path $RunRoot "report"

@($RawTune, $GlobalTune, $EvalDir, $StatsDir, $TimingDir, $ReportDir) | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

$Stage = [ordered]@{}

Write-Host "`n=== 1/7 Validate complete caches ==="
& $Py "$PSScriptRoot\tools\validate_inputs.py" `
    --combined-cache $CombinedCache `
    --spatial-cache $SpatialCache `
    --out (Join-Path $RunRoot "input_validation.json")
if ($LASTEXITCODE -ne 0) { throw "input validation failed" }

Write-Host "`n=== 2/7 Tune global grid on ALL training conditions (both models) ==="
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $Py scripts\tune_fusion.py `
    --models resnet18 deit `
    --cache-dir $CombinedCache `
    --tag final_global_train `
    --out $RawTune `
    --tune-severities 1 2 3 4 5 `
    --frac-tune 0.5 `
    --split-seed 0 `
    --ladder-len 8 `
    --device $Device `
    --signal-backend torch `
    --config-batch 64 `
    --signal-chunk 256 `
    --signal-pair-batch 32
if ($LASTEXITCODE -ne 0) { throw "grid tuning failed" }
$sw.Stop(); $Stage["grid_tuning_seconds"] = $sw.Elapsed.TotalSeconds

$ResGrid = (Get-ChildItem "$RawTune\tuning_grid_resnet18_final_global_train_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$DeitGrid = (Get-ChildItem "$RawTune\tuning_grid_deit_final_global_train_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
if (-not $ResGrid -or -not $DeitGrid) { throw "could not locate grid tuning outputs" }

Write-Host "`n=== 3/7 Tune final DeiT CE+view-bias on ALL training conditions ==="
$sw.Restart()
& $Py scripts\tune_fusion_ce.py `
    --models deit `
    --cache-dir $CombinedCache `
    --tag final_global_train `
    --out $RawTune `
    --tune-severities 1 2 3 4 5 `
    --frac-tune 0.5 `
    --split-seed 0 `
    --ladder-len 8 `
    --grid-tuning $DeitGrid `
    --variants bias `
    --optimizer-backend torch `
    --device $Device `
    --resident-probs auto `
    --signal-chunk 256 `
    --signal-pair-batch 32
if ($LASTEXITCODE -ne 0) { throw "CE+bias tuning failed" }
$sw.Stop(); $Stage["deit_ce_bias_tuning_seconds"] = $sw.Elapsed.TotalSeconds

$DeitCE = (Get-ChildItem "$RawTune\tuning_ce_bias_deit_final_global_train_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
if (-not $DeitCE) { throw "could not locate DeiT CE+bias tuning output" }

# Remove LOFO configs from the final records.  This is essential: evaluation
# must use one global configuration trained on all TRAIN/TUNE conditions.
$ResGlobal = Join-Path $GlobalTune "resnet18_grid_global.json"
$DeitGlobal = Join-Path $GlobalTune "deit_ce_bias_global.json"
& $Py "$PSScriptRoot\tools\extract_global_tuning.py" --input $ResGrid --output $ResGlobal --label "ResNet18 global grid"
& $Py "$PSScriptRoot\tools\extract_global_tuning.py" --input $DeitCE --output $DeitGlobal --label "DeiT global CE+bias"

Write-Host "`n=== 4/7 Held-out test evaluation: final vs VANILLA, all 75 conditions ==="
$sw.Restart()
& $Py scripts\evaluate.py `
    --models resnet18 `
    --cache-dir $CombinedCache `
    --tuning $ResGlobal `
    --tag final_global_grid `
    --out $EvalDir `
    --severities 1 2 3 4 5 `
    --eval-half test `
    --frac-tune 0.5 `
    --split-seed 0 `
    --ladder-len 8 `
    --device $Device `
    --quiet-progress
if ($LASTEXITCODE -ne 0) { throw "ResNet evaluation failed" }

& $Py scripts\evaluate.py `
    --models deit `
    --cache-dir $CombinedCache `
    --tuning $DeitGlobal `
    --tag final_global_ce_bias `
    --out $EvalDir `
    --severities 1 2 3 4 5 `
    --eval-half test `
    --frac-tune 0.5 `
    --split-seed 0 `
    --ladder-len 8 `
    --device $Device `
    --quiet-progress
if ($LASTEXITCODE -ne 0) { throw "DeiT evaluation failed" }
$sw.Stop(); $Stage["heldout_evaluation_seconds"] = $sw.Elapsed.TotalSeconds

$ResCsv = (Get-ChildItem "$EvalDir\eval_resnet18_final_global_grid_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$ResJson = (Get-ChildItem "$EvalDir\eval_resnet18_final_global_grid_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$DeitCsv = (Get-ChildItem "$EvalDir\eval_deit_final_global_ce_bias_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$DeitJson = (Get-ChildItem "$EvalDir\eval_deit_final_global_ce_bias_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
if (-not $ResCsv -or -not $ResJson -or -not $DeitCsv -or -not $DeitJson) { throw "could not locate evaluation outputs" }

$BootRes = ""
$BootDeit = ""
if (-not $SkipBootstrap) {
    Write-Host "`n=== 5/7 Paired bootstrap: final vs vanilla (plus spatial reference) ==="
    $sw.Restart()
    $BootRes = Join-Path $StatsDir "bootstrap_resnet18.json"
    $BootDeit = Join-Path $StatsDir "bootstrap_deit.json"
    & $Py scripts\bootstrap_final.py `
        --model resnet18 `
        --combined-cache $CombinedCache `
        --spatial-cache $SpatialCache `
        --tuning $ResGlobal `
        --ladder-len 8 `
        --severities 1 2 3 4 5 `
        --frac-tune 0.5 `
        --split-seed 0 `
        --n-boot $BootstrapResamples `
        --device $Device `
        --quiet-progress `
        --out $BootRes
    if ($LASTEXITCODE -ne 0) { throw "ResNet bootstrap failed" }

    & $Py scripts\bootstrap_final.py `
        --model deit `
        --combined-cache $CombinedCache `
        --spatial-cache $SpatialCache `
        --tuning $DeitGlobal `
        --ladder-len 8 `
        --severities 1 2 3 4 5 `
        --frac-tune 0.5 `
        --split-seed 0 `
        --n-boot $BootstrapResamples `
        --device $Device `
        --quiet-progress `
        --out $BootDeit
    if ($LASTEXITCODE -ne 0) { throw "DeiT bootstrap failed" }
    $sw.Stop(); $Stage["bootstrap_seconds"] = $sw.Elapsed.TotalSeconds
} else {
    $Stage["bootstrap_seconds"] = $null
}

$TimingCsv = ""
if (-not $SkipBenchmark) {
    Write-Host "`n=== 6/7 Real inference timing: vanilla 1-view vs final 15-view ==="
    $sw.Restart()
    & $Py "$PSScriptRoot\tools\benchmark_inference.py" `
        --resnet-tuning $ResGlobal `
        --deit-tuning $DeitGlobal `
        --out-dir $TimingDir `
        --batch-size $BenchmarkBatchSize `
        --reps $BenchmarkReps `
        --warmup 3 `
        --device $Device
    if ($LASTEXITCODE -ne 0) { throw "inference benchmark failed" }
    $sw.Stop(); $Stage["benchmark_seconds"] = $sw.Elapsed.TotalSeconds
    $TimingCsv = Join-Path $TimingDir "inference_timing.csv"
} else {
    $Stage["benchmark_seconds"] = $null
}

$StagePath = Join-Path $TimingDir "stage_times.json"
$Stage | ConvertTo-Json | Set-Content -Encoding Ascii $StagePath

Write-Host "`n=== 7/7 Build comprehensive report folder ==="
$ReportArgs = @(
    "$PSScriptRoot\tools\build_final_report.py",
    "--resnet-csv", $ResCsv,
    "--deit-csv", $DeitCsv,
    "--resnet-json", $ResJson,
    "--deit-json", $DeitJson,
    "--resnet-tuning", $ResGlobal,
    "--deit-tuning", $DeitGlobal,
    "--stage-times", $StagePath,
    "--out-dir", $ReportDir
)
if ($TimingCsv) { $ReportArgs += @("--timing-csv", $TimingCsv) }
if ($BootRes) { $ReportArgs += @("--bootstrap-resnet", $BootRes) }
if ($BootDeit) { $ReportArgs += @("--bootstrap-deit", $BootDeit) }
& $Py @ReportArgs
if ($LASTEXITCODE -ne 0) { throw "report generation failed" }

Write-Host "`nDONE"
Write-Host "Complete result folder: $RunRoot"
Write-Host "Main report:           $(Join-Path $ReportDir 'FINAL_REPORT.md')"
Write-Host "All 150 condition rows: $(Join-Path $ReportDir 'all_conditions.csv')"
