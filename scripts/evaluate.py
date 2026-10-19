"""Evaluate the fused predictor against the baselines over the whole cache and
write one results table.

Config source:
  --tuning results/tuning_<model>_*.json   use the leave-one-family-out configs
                                           (each corruption scored with a config
                                           tuned without its family); clean uses
                                           the global config
  --tuning auto                            pick the newest tuning file per model
  (no --tuning)                            one manual config from the flags below

By default only the held-out image half is scored (the half not used for
tuning); pass --eval-half all to score everything.

    python scripts/evaluate.py --tuning auto
    python scripts/evaluate.py --alpha 1 --beta 0.5 --gamma 0.5 --tau 0.1

Writes results/eval_<model>_<timestamp>.csv (one row per corruption/severity)
and results/eval_<model>_<timestamp>.json (the aggregate summary).
"""
import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import cache
from src.metrics import FusionConfig
from src.evaluation import device_from_arg, score_many_torch
from src.splits import CORRUPTION_FAMILY, FAMILIES, tune_test_masks

#: columns of the per-condition CSV
ROW_FIELDS = ["model", "corruption", "family", "severity", "n",
              "orig_acc", "equal_acc", "conf_acc", "agree_acc", "view1_acc",
              "fused_acc", "oracle_acc",
              "orig_ece", "fused_ece", "orig_nll", "fused_nll",
              "negative_transfer", "positive_transfer"]


def latest_tuning(model_name, tag=None, results_dir="results"):
    """Return the newest tuning JSON for this model/cache tag.

    Final tuning files are named ``tuning_grid_*`` or ``tuning_ce*``.  The
    broader glob also accepts development-era tuning files so old local result
    directories remain usable after the repository cleanup.
    """
    root = Path(results_dir)
    if tag:
        patterns = [root / f"tuning_*_{model_name}_{tag}_*.json"]
    else:
        patterns = [root / f"tuning_*_{model_name}_*.json"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(str(pattern)))
    if not files:
        raise SystemExit(f"--tuning auto: no tuning JSON found for {model_name} tag={tag!r} in {root}")
    return max(files, key=lambda x: Path(x).stat().st_mtime)


def config_picker(args, model_name, tag=None):
    """Build the corruption -> FusionConfig mapping for this run.

    With no --tuning, every corruption gets the one manual config from the CLI
    flags. With --tuning, each corruption gets its family's leave-one-family-out
    config and anything unrecognised (e.g. "clean") gets the global config.

    Returns:
        (pick, source_label) where `pick(corruption)` -> FusionConfig and
        `source_label` is a short string for the console header.
    """
    if not args.tuning:
        manual = FusionConfig(args.alpha, args.beta, args.gamma, args.tau,
                              not args.no_standardize, args.m, args.agreement)
        return lambda corruption: manual, "manual"

    path = latest_tuning(model_name, tag) if args.tuning == "auto" else args.tuning
    tune = json.loads(Path(path).read_text())
    held = {fam: FusionConfig.from_dict(tune["held_out"][fam]["config"])
            for fam in tune["held_out"]}
    glob_cfg = FusionConfig.from_dict(tune["global"]["config"])

    def pick(corruption):
        fam = CORRUPTION_FAMILY.get(corruption)
        return held.get(fam, glob_cfg)

    return pick, os.path.basename(path)


def main():
    """Score every cached condition for each model and write the results table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"],
                    choices=["resnet18", "deit"])
    ap.add_argument("--tuning", default=None,
                    help='path to a tuning json, or "auto" for the newest per model')
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--eval-half", choices=["test", "all"], default="test")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--frac-tune", type=float, default=0.5)
    # manual config (used only when --tuning is not given)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--agreement", choices=["all", "origin"], default="all")
    ap.add_argument("--ladder-len", type=int, default=None,
                    help="see tune_fusion.py --ladder-len; must match what the "
                         "tuning config (if any) was produced with")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache-dir", default="cache",
                    help="read profiles from this cache (must match cache_profiles.py --cache-dir)")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output filename, e.g. the ladder name; "
                         "defaults to the cache dir name when it isn't 'cache'")
    ap.add_argument("--device", default="auto",
                    help="torch device for signal/fusion evaluation (default: auto)")
    ap.add_argument("--signal-chunk", type=int, default=256)
    ap.add_argument("--signal-pair-batch", type=int, default=32)
    ap.add_argument("--quiet-progress", action="store_true")
    args = ap.parse_args()

    cache.CACHE_ROOT = Path(args.cache_dir)
    tag = args.tag or (Path(args.cache_dir).name if args.cache_dir != "cache" else None)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_from_arg(args.device)
    for model_name in args.models:
        pick, src = config_picker(args, model_name, tag)
        print(f"\n=== {model_name}  (config: {src}) ===")
        print(f"  evaluation device: {device}")

        rows = []
        for corruption in CORRUPTION_FAMILY:
            for sev in args.severities:
                if not cache.has_condition(model_name, corruption, sev):
                    continue
                d = cache.load_condition(model_name, corruption, sev)
                if args.eval_half == "test":
                    _, mask = tune_test_masks(d["paths"], args.frac_tune, args.split_seed)
                else:
                    mask = np.ones(len(d["labels"]), dtype=bool)
                cfg = pick(corruption)
                r = score_many_torch(
                    d["probs"][mask], d["labels"][mask], {"method": cfg},
                    ladder_len=args.ladder_len, device=device,
                    signal_chunk=args.signal_chunk,
                    signal_pair_batch=args.signal_pair_batch,
                )["method"]
                r.update(model=model_name, corruption=corruption,
                         family=CORRUPTION_FAMILY[corruption], severity=sev)
                rows.append(r)
                if not args.quiet_progress:
                    print(f"  {corruption:18s}/{sev}: fused {100*r['fused_acc']:.2f}%")

        # clean (always all images)
        clean_row = None
        if cache.has_condition(model_name, "clean", 0):
            d = cache.load_condition(model_name, "clean", 0)
            cfg = pick("clean")
            clean_row = score_many_torch(
                d["probs"], d["labels"], {"method": cfg},
                ladder_len=args.ladder_len, device=device,
                signal_chunk=args.signal_chunk,
                signal_pair_batch=args.signal_pair_batch,
            )["method"]

        _report(model_name, rows, clean_row, out_dir, stamp, tag)


def _mean(rows, key):
    """Mean of `rows[i][key]` over all rows (nan if there are none)."""
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def _report(model_name, rows, clean_row, out_dir, stamp, tag=None):
    """Write the per-condition CSV and the aggregate JSON, and print the table."""
    name = f"eval_{model_name}" + (f"_{tag}" if tag else "") + f"_{stamp}"
    # per-condition csv
    csv_path = out_dir / f"{name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # per-corruption means over severities
    per_corr = {}
    for fam, corrs in FAMILIES.items():
        for c in corrs:
            cr = [r for r in rows if r["corruption"] == c]
            if cr:
                per_corr[c] = {k: _mean(cr, k) for k in
                               ("orig_acc", "equal_acc", "conf_acc", "agree_acc",
                                "view1_acc", "fused_acc", "oracle_acc",
                                "negative_transfer", "fused_ece", "orig_nll",
                                "fused_nll")}

    summary = {
        "model": model_name,
        "timestamp": stamp,
        "clean_orig_acc": clean_row["orig_acc"] if clean_row else None,
        "clean_fused_acc": clean_row["fused_acc"] if clean_row else None,
        "mCA_orig": _mean(rows, "orig_acc"),
        "mCA_equal": _mean(rows, "equal_acc"),
        "mCA_conf": _mean(rows, "conf_acc"),
        "mCA_agree": _mean(rows, "agree_acc"),
        "mCA_view1": _mean(rows, "view1_acc"),
        "mCA_fused": _mean(rows, "fused_acc"),
        "mCA_oracle": _mean(rows, "oracle_acc"),
        "mean_negative_transfer": _mean(rows, "negative_transfer"),
        "mean_fused_ece": _mean(rows, "fused_ece"),
        "mean_orig_ece": _mean(rows, "orig_ece"),
        "mean_fused_nll": _mean(rows, "fused_nll"),
        "mean_orig_nll": _mean(rows, "orig_nll"),
        "worst_corruption": min(per_corr, key=lambda c: per_corr[c]["fused_acc"]) if per_corr else None,
        "per_corruption": per_corr,
    }
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # console table
    print(f"{'corruption':>18} {'orig':>7} {'equal':>7} {'conf':>7} {'agree':>7} "
          f"{'view1':>7} {'fused':>7} {'oracle':>7} {'neg-tr':>7}")
    print("-" * 60)
    for fam, corrs in FAMILIES.items():
        for c in corrs:
            if c not in per_corr:
                continue
            p = per_corr[c]
            print(f"{c:>18} {p['orig_acc']*100:>6.1f} {p['equal_acc']*100:>6.1f} "
                  f"{p['conf_acc']*100:>6.1f} {p['agree_acc']*100:>6.1f} "
                  f"{p['view1_acc']*100:>6.1f} "
                  f"{p['fused_acc']*100:>6.1f} {p['oracle_acc']*100:>6.1f} "
                  f"{p['negative_transfer']*100:>6.1f}")
    print("-" * 60)
    if clean_row:
        print(f"{'clean':>18} {clean_row['orig_acc']*100:>6.1f} {'':>7} "
              f"{'':>7} {'':>7} {'':>7} {clean_row['fused_acc']*100:>6.1f}")
    print(f"{'mean corrupted':>18} {summary['mCA_orig']*100:>6.1f} "
          f"{summary['mCA_equal']*100:>6.1f} {summary['mCA_conf']*100:>6.1f} "
          f"{summary['mCA_agree']*100:>6.1f} {summary['mCA_view1']*100:>6.1f} "
          f"{summary['mCA_fused']*100:>6.1f} "
          f"{summary['mCA_oracle']*100:>6.1f} {summary['mean_negative_transfer']*100:>6.1f}")
    print(f"  fused - orig : {(summary['mCA_fused']-summary['mCA_orig'])*100:+.1f} pts")
    print(f"  worst corruption (fused): {summary['worst_corruption']}")
    print(f"  ECE  orig {summary['mean_orig_ece']:.3f} -> fused {summary['mean_fused_ece']:.3f}")
    print(f"  NLL  orig {summary['mean_orig_nll']:.3f} -> fused {summary['mean_fused_nll']:.3f}")
    print(f"  saved {csv_path}  and  {json_path}")
    return csv_path, json_path


if __name__ == "__main__":
    main()
