#!/bin/bash
# Full pipeline: caches -> grid + CE tuning (LOFO, tune half) -> tune-half method selection ->
# test-half evaluation -> paired cluster bootstraps. Everything lands in $R.
#
#   bash scripts/run_full_pipeline.sh                  # all phases
#   PHASES="extras figures" bash scripts/run_full_pipeline.sh
#   PY=/path/to/python bash scripts/run_full_pipeline.sh
# Phases: cache12 build tune12+cache3 tune3 geo extras figures. Optional (not
# in the default list): padctl - the padded corner-crop control (builds
# cache_spatial_pad/, ~2 h GPU; see RESULTS.md "Crop geometry check"); extra - the
# 4 held-out ImageNet-C extra corruptions with the frozen configs (needs
# `fetch_imagenet_c.py --tarballs extra` first; RESULTS.md "Unseen corruption types");
# views - other view families and signals (~2.5 h GPU, after the tuning phases;
# RESULTS.md "Other view families and signals"); spatialfit - the 8 crop/flip
# views tuned exactly like the spectral ones, and tuned spectral vs tuned
# spatial (~45 min GPU, after the tuning phases); signals - what each reliability
# signal contributes: AUROC of C/A/S per view and the CE + bias fit refit with
# every signal subset (~30 min GPU, after the tuning phases). Caches that
# already exist are skipped; tuning/evaluation outputs are timestamped and the
# newest file is always the one used.
set -euo pipefail
cd "$(dirname "$0")/.."
# Python with the requirements installed: $PY if set, else the authors' conda
# env when present, else whatever 'python' is on PATH.
if [ -z "${PY:-}" ]; then
  PY=python
  [ -x /home/student/anaconda3/envs/stud_prj/bin/python ] && PY=/home/student/anaconda3/envs/stud_prj/bin/python
fi
PY="$PY -u"
C=${C:-cache_fixed}; SP=${SP:-cache_spatial}; CB=${CB:-cache_combined}; G=${G:-cache_geometric}
R=${R:-results}
PHASES=${PHASES:-"cache12 build tune12+cache3 tune3 geo extras figures"}
mkdir -p "$R" "$R/logs"
Q=--quiet-progress

log() { echo "[$(date +%H:%M:%S)] $*"; }

# tune_eval <cache_dir> <tag> <ladder_len or ''> <models...>
tune_eval() {
  local cdir=$1 tag=$2 ll=$3; shift 3
  local LL=(); [ -n "$ll" ] && LL=(--ladder-len "$ll")
  log "tune grid  $cdir $*"
  $PY scripts/tune_fusion.py --models "$@" --cache-dir "$cdir" --out "$R" --tag "$tag" "${LL[@]}"
  log "tune CE    $cdir $*"
  $PY scripts/tune_fusion_ce.py --models "$@" --cache-dir "$cdir" --out "$R" --tag "$tag" "${LL[@]}" \
      --grid-tuning auto --variants base bias unconstrained bias_only
  log "select     $cdir $*"
  $PY scripts/select_method.py --models "$@" --tag "$tag" --results "$R"
  for m in "$@"; do
    local frozen; frozen=$(ls -t "$R"/tuning_${m}_${tag}_final_*.json | head -1)
    log "evaluate   $m $cdir ($frozen)"
    $PY scripts/evaluate.py --models "$m" --cache-dir "$cdir" "${LL[@]}" --tuning "$frozen" \
        --out "$R" --tag "${tag}_final" $Q
    local M=()
    for meth in grid ce ce_bias ce_unconstrained ce_bias_only; do
      local f; f=$(ls "$R"/tuning_${meth}_${m}_${tag}_2*.json | sort | tail -1)
      M+=(--method "${tag}_${meth}" "$f")
    done
    log "ablation   $m $cdir"
    $PY scripts/evaluate_many.py --model "$m" --cache-dir "$cdir" "${LL[@]}" --out "$R" $Q "${M[@]}"
    log "bootstrap  selected vs equal average  $m $cdir"
    # 'spatial' mode = plain equal-weight average over every view of the cache
    $PY scripts/bootstrap_pairwise.py --model "$m" \
        --a-mode fused --a-cache "$cdir" --a-tuning "$frozen" ${ll:+--a-ladder-len $ll} \
        --b-mode spatial --b-cache "$cdir" \
        $Q --out "$R/bootstrap_selected_vs_equal_${m}_${tag}.json"
    log "bootstrap  selected vs bias-only  $m $cdir"
    $PY scripts/bootstrap_pairwise.py --model "$m" \
        --a-mode fused --a-cache "$cdir" --a-tuning "$frozen" ${ll:+--a-ladder-len $ll} \
        --b-mode fused --b-cache "$cdir" --b-tuning "$(ls "$R"/tuning_ce_bias_only_${m}_${tag}_2*.json | sort | tail -1)" ${ll:+--b-ladder-len $ll} \
        $Q --out "$R/bootstrap_selected_vs_biasonly_${m}_${tag}.json"
  done
  log "bootstrap  fused vs orig / spatial  $cdir $*"
  $PY scripts/bootstrap_ci.py --models "$@" --cache-dir "$cdir" --tag "${tag}_final" "${LL[@]}" \
      --results-dir "$R" --spatial-cache "$SP" --out "$R"
}

# direct call, for smoke tests:  bash scripts/run_full_pipeline.sh fn tune_eval <cache> <tag> <ll> <models...>
if [ "${1:-}" = fn ]; then shift; "$@"; exit $?; fi

# clean images were renamed by their re-host; the content-matched name map puts
# each clean image in the same tune/test half as its corrupted versions
[ -f data/clean_name_map.json ] || $PY scripts/match_clean_names.py

for phase in $PHASES; do
  case $phase in
    cache12)
      log "phase cache12: spectral + spatial caches, resnet18 + deit (parallel)"
      $PY scripts/cache_profiles.py --models resnet18 deit --ladder fixed --cache-dir $C > "$R/logs/cache_fixed_12.log" 2>&1 &
      p1=$!
      $PY scripts/cache_spatial_profiles.py --models resnet18 deit --cache-dir $SP > "$R/logs/cache_spatial_12.log" 2>&1 &
      p2=$!
      wait $p1; wait $p2 ;;
    build)
      log "phase build: virtual combined cache"
      $PY scripts/build_combined_cache.py --virtual --out-dir $CB --spectral-dir $C --spatial-dir $SP ;;
    tune12+cache3)
      log "phase tune12+cache3: convnext caches || resnet18+deit tuning/eval"
      ( $PY scripts/cache_profiles.py --models convnext --ladder fixed --cache-dir $C &&
        $PY scripts/cache_spatial_profiles.py --models convnext --cache-dir $SP ) > "$R/logs/cache_convnext.log" 2>&1 &
      p1=$!
      ( tune_eval $C cache_fixed_lofo "" resnet18 deit &&
        tune_eval $CB cache_combined_lofo 8 resnet18 deit ) > "$R/logs/tune12.log" 2>&1 &
      p2=$!
      wait $p2; wait $p1 ;;
    tune3)
      log "phase tune3: convnext tuning/eval"
      ( tune_eval $C cache_fixed_lofo "" convnext &&
        tune_eval $CB cache_combined_lofo 8 convnext ) > "$R/logs/tune3.log" 2>&1 ;;
    geo)
      log "phase geo: geometric-ladder ablation, resnet18 + deit"
      {
        $PY scripts/cache_profiles.py --models resnet18 deit --ladder geometric --cache-dir $G
        $PY scripts/tune_fusion.py --models resnet18 deit --cache-dir $G --out "$R" --tag cache_geometric_lofo
        $PY scripts/tune_fusion_ce.py --models resnet18 deit --cache-dir $G --out "$R" --tag cache_geometric_lofo \
            --grid-tuning auto --variants bias
        for m in resnet18 deit; do
          $PY scripts/bootstrap_pairwise.py --model $m \
              --a-mode fused --a-cache $C --a-tuning "$(ls "$R"/tuning_ce_bias_${m}_cache_fixed_lofo_2*.json | sort | tail -1)" \
              --b-mode fused --b-cache $G --b-tuning "$(ls "$R"/tuning_ce_bias_${m}_cache_geometric_lofo_2*.json | sort | tail -1)" \
              $Q --out "$R/bootstrap_fixed_vs_geo_$m.json"
        done
      } > "$R/logs/geo.log" 2>&1 ;;
    extras)
      log "phase extras: overhead, spectral-energy measurement, profile shape, signal ceiling"
      {
        $PY scripts/measure_overhead.py --models resnet18 deit convnext --out "$R/inference_overhead.json"
        $PY scripts/spectral_energy.py --n-per-condition 500 --out "$R/spectral_energy.json"
        $PY scripts/profile_shape.py --out "$R/profile_shape.json"
        $PY scripts/signal_ceiling.py --models resnet18 deit convnext --cache-dir $C --no-gbm --out "$R"
        $PY scripts/signal_ceiling.py --models resnet18 deit convnext --cache-dir $CB --ladder-len 8 --no-gbm --out "$R"
      } > "$R/logs/extras.log" 2>&1 ;;
    extra)
      log "phase extra: held-out extra corruptions, frozen configs"
      X="speckle_noise gaussian_blur spatter saturate"
      {
        $PY scripts/cache_profiles.py --models resnet18 deit convnext --ladder fixed --cache-dir $C --corruptions $X
        $PY scripts/cache_spatial_profiles.py --models resnet18 deit convnext --cache-dir $SP --corruptions $X
        $PY scripts/evaluate_extra.py --results "$R" --out "$R/extra_corruptions.json"
        $PY scripts/spectral_energy.py --corruptions $X --n-per-condition 500 --out "$R/spectral_energy_extra.json"
      } > "$R/logs/extra.log" 2>&1 ;;
    spatialfit)
      log "phase spatialfit: tune the spatial-TTA views like the spectral ones"
      {
        # ladder length 1: only the original is on a 'ladder', so stability is
        # neutral and the fit uses confidence, agreement and the per-view bias
        tune_eval $SP cache_spatial_lofo 1 resnet18 deit convnext
        for m in resnet18 deit convnext; do
          $PY scripts/bootstrap_pairwise.py --model $m \
              --a-mode fused --a-cache $C --a-tuning "$(ls -t "$R"/tuning_${m}_cache_fixed_lofo_final_*.json | head -1)" \
              --b-mode fused --b-cache $SP --b-tuning "$(ls -t "$R"/tuning_${m}_cache_spatial_lofo_final_*.json | head -1)" \
              --b-ladder-len 1 $Q --out "$R/bootstrap_spectral_vs_spatial_tuned_$m.json"
          $PY scripts/bootstrap_pairwise.py --model $m \
              --a-mode fused --a-cache $CB --a-tuning "$(ls -t "$R"/tuning_${m}_cache_combined_lofo_final_*.json | head -1)" \
              --a-ladder-len 8 \
              --b-mode fused --b-cache $SP --b-tuning "$(ls -t "$R"/tuning_${m}_cache_spatial_lofo_final_*.json | head -1)" \
              --b-ladder-len 1 $Q --out "$R/bootstrap_combined_vs_spatial_tuned_$m.json"
        done
      } > "$R/logs/spatialfit.log" 2>&1 ;;
    views)
      log "phase views: other view families and signals"
      {
        # run one at a time: each needs ~10-12 GB of GPU memory
        $PY scripts/view_family_comparison.py --out-dir "$R/view_family" --out "$R/view_family_comparison.json"
        $PY scripts/view_family_comparison.py --set ladders --out-dir "$R/ladder_compare" \
            --out "$R/ladder_comparison.json"
        for meth in ce_bias_only ce_bias; do
          $PY scripts/decorrelation_diagnostic.py --cache-dir $CB --results-dir "$R" --method $meth
        done
        $PY scripts/experiment_ambiguity.py --cache-dir $CB --results-dir "$R" --out "$R"
      } > "$R/logs/views.log" 2>&1 ;;
    signals)
      log "phase signals: per-signal AUROC and CE + bias refit per signal subset"
      SA="$R/signal_ablation"
      mkdir -p "$SA"
      {
        for m in resnet18 deit convnext; do
          grid=$(ls "$R"/tuning_grid_${m}_cache_fixed_lofo_2*.json | sort | tail -1)
          # subset (signals to fit) : agreement mode used by A
          for spec in ":all" "C:all" "A:all" "S:all" "C A:all" "C S:all" "A S:all" \
                      "C A S:all" "C A S:origin"; do
            sigs=${spec%%:*}; agree=${spec##*:}
            $PY scripts/tune_fusion_ce.py --models $m --cache-dir $C --out "$SA" \
                --tag cache_fixed_lofo --grid-tuning "$grid" --variants bias \
                --agreement-modes $agree --signals $sigs
          done
        done
        $PY scripts/signal_ablation.py --cache-dir $C --results "$R" --fits "$SA" --out "$SA"
      } > "$R/logs/signals.log" 2>&1 ;;
    padctl)
      log "phase padctl: reflect-padded corner crops (control for the resize-196 baseline)"
      {
        $PY scripts/cache_spatial_profiles.py --models resnet18 deit convnext --corner-mode pad \
            --cache-dir cache_spatial_pad
        $PY scripts/spatial_crop_control.py --resize-cache $SP --pad-cache cache_spatial_pad \
            --out "$R/spatial_crop_control.json"
        for m in resnet18 deit convnext; do
          $PY scripts/bootstrap_pairwise.py --model $m --a-mode spatial --a-cache cache_spatial_pad \
              --b-mode spatial --b-cache $SP $Q --out "$R/bootstrap_spatial_pad_vs_resize196_$m.json"
        done
        $PY scripts/bootstrap_ci.py --models resnet18 deit convnext --cache-dir $C \
            --tag cache_fixed_lofo_final --results-dir "$R" --spatial-cache cache_spatial_pad \
            --suffix _spatial_pad --out "$R"
      } > "$R/logs/padctl.log" 2>&1 ;;
    figures)
      log "phase figures"
      RESULTS_DIR="$R" FIGURES_DIR=figures $PY scripts/make_paper_figures.py > "$R/logs/figures.log" 2>&1 ;;
    *) echo "unknown phase $phase"; exit 1 ;;
  esac
done
log "ALL_DONE"
