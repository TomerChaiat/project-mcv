"""Score the spatial-TTA baseline (scripts/cache_spatial_profiles.py) against
the same held-out test half evaluate.py uses, so the numbers are directly
comparable to the spectral fused/oracle/etc. table.

    python scripts/evaluate_tta.py

Writes results/tta_<model>_<timestamp>.{csv,json}.
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import cache
from src.metrics import accuracy, top1, transfer_rates
from src.splits import CORRUPTION_FAMILY, FAMILIES, tune_test_masks

ROW_FIELDS = ["model", "corruption", "family", "severity", "n",
              "orig_acc", "tta_acc", "negative_transfer", "positive_transfer"]


def score_condition(d, mask):
    """orig/TTA accuracy + transfer rates for one loaded spatial condition."""
    probs = d["probs"][mask].astype(np.float32)
    labels = d["labels"][mask]
    orig_pred = top1(probs[:, 0])
    tta_pred = top1(probs.mean(axis=1))
    row = {
        "n": int(len(labels)),
        "orig_acc": accuracy(orig_pred, labels),
        "tta_acc": accuracy(tta_pred, labels),
    }
    row.update(transfer_rates(orig_pred, tta_pred, labels))
    return row


def main():
    """Score the spatial-TTA cache for each model and write the results table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"],
                    choices=["resnet18", "deit"])
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--eval-half", choices=["test", "all"], default="test")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--cache-dir", default="cache_spatial")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cache.CACHE_ROOT = Path(args.cache_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        print(f"\n=== {model_name} (spatial TTA baseline) ===")
        rows = []
        for corruption in CORRUPTION_FAMILY:
            for sev in args.severities:
                if not cache.has_condition(model_name, corruption, sev):
                    continue
                d = cache.load_spatial_condition(model_name, corruption, sev)
                if args.eval_half == "test":
                    _, mask = tune_test_masks(d["paths"], args.frac_tune, args.split_seed)
                else:
                    mask = np.ones(len(d["labels"]), dtype=bool)
                r = score_condition(d, mask)
                r.update(model=model_name, corruption=corruption,
                         family=CORRUPTION_FAMILY[corruption], severity=sev)
                rows.append(r)

        clean_row = None
        if cache.has_condition(model_name, "clean", 0):
            d = cache.load_spatial_condition(model_name, "clean", 0)
            clean_row = score_condition(d, np.ones(len(d["labels"]), dtype=bool))

        _report(model_name, rows, clean_row, out_dir, stamp)


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def _report(model_name, rows, clean_row, out_dir, stamp):
    name = f"tta_{model_name}_{stamp}"
    csv_path = out_dir / f"{name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    per_corr = {}
    for fam, corrs in FAMILIES.items():
        for c in corrs:
            cr = [r for r in rows if r["corruption"] == c]
            if cr:
                per_corr[c] = {k: _mean(cr, k) for k in
                               ("orig_acc", "tta_acc", "negative_transfer")}

    summary = {
        "model": model_name,
        "timestamp": stamp,
        "clean_orig_acc": clean_row["orig_acc"] if clean_row else None,
        "clean_tta_acc": clean_row["tta_acc"] if clean_row else None,
        "mCA_orig": _mean(rows, "orig_acc"),
        "mCA_tta": _mean(rows, "tta_acc"),
        "mean_negative_transfer": _mean(rows, "negative_transfer"),
        "per_corruption": per_corr,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))

    print(f"{'corruption':>18} {'orig':>7} {'tta':>7} {'neg-tr':>7}")
    print("-" * 45)
    for fam, corrs in FAMILIES.items():
        for c in corrs:
            if c not in per_corr:
                continue
            p = per_corr[c]
            print(f"{c:>18} {p['orig_acc']*100:>6.1f} {p['tta_acc']*100:>6.1f} "
                  f"{p['negative_transfer']*100:>6.1f}")
    print("-" * 45)
    if clean_row:
        print(f"{'clean':>18} {clean_row['orig_acc']*100:>6.1f} "
              f"{clean_row['tta_acc']*100:>6.1f}")
    print(f"{'mean corrupted':>18} {summary['mCA_orig']*100:>6.1f} "
          f"{summary['mCA_tta']*100:>6.1f} {summary['mean_negative_transfer']*100:>6.1f}")
    print(f"  tta - orig : {(summary['mCA_tta']-summary['mCA_orig'])*100:+.1f} pts")
    print(f"  saved {csv_path}  and  {out_dir / f'{name}.json'}")


if __name__ == "__main__":
    main()
