"""Crop geometry check for the spatial-TTA corner crops.

The baseline corner crops (cache_spatial/) are 196 px crops resized to 224;
the upsampling slightly smooths them, i.e. adds a mild low-pass. The control
(cache_spatial_pad/, built with ``cache_spatial_profiles.py --corner-mode
pad``) cuts 224 windows from the input reflect-padded to 256 instead: no
resampling, no smoothing. This compares the two on the test half: accuracy
of each single view relative to the original, and of the 8-view equal-weight
spatial TTA, overall and per corruption family. The paired CIs come from
bootstrap_pairwise.py / bootstrap_ci.py --suffix (see run_full_pipeline.sh,
phase padctl).

    python scripts/spatial_crop_control.py --out results/spatial_crop_control.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import cache
from src.splits import CORRUPTION_FAMILY, FAMILIES, tune_test_masks

VIEW_NAMES = ["original", "hflip", "TL", "TR", "BL", "BR", "flip TL", "flip BR"]


def summarize(model, root, severities, frac_tune, seed):
    """Test-half view and TTA accuracies of one spatial cache (percent)."""
    cache.CACHE_ROOT = Path(root)
    views, tta, fam = [], [], {f: [] for f in FAMILIES}
    for corruption, family in CORRUPTION_FAMILY.items():
        for sev in severities:
            d = cache.load_spatial_condition(model, corruption, sev)
            _, mask = tune_test_masks(d["paths"], frac_tune, seed)
            p = d["probs"][mask].astype(np.float32)
            y = d["labels"][mask]
            va = (p.argmax(2) == y[:, None]).mean(0)
            t = (p.mean(1).argmax(1) == y).mean()
            views.append(va)
            tta.append(t)
            fam[family].append(t - va[0])
    views = 100 * np.mean(views, 0)
    out = {"orig_acc": float(views[0]), "tta_acc": float(100 * np.mean(tta)),
           "view_minus_orig": {n: float(v - views[0]) for n, v in zip(VIEW_NAMES, views)},
           "tta_gain_by_family": {f: float(100 * np.mean(v)) for f, v in fam.items()}}
    if cache.has_condition(model, "clean", 0):
        d = cache.load_spatial_condition(model, "clean", 0)
        _, mask = tune_test_masks(d["paths"], frac_tune, seed)
        p = d["probs"][mask].astype(np.float32)
        y = d["labels"][mask]
        out["tta_gain_clean"] = float(100 * ((p.mean(1).argmax(1) == y).mean()
                                             - (p[:, 0].argmax(1) == y).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit", "convnext"])
    ap.add_argument("--resize-cache", default="cache_spatial")
    ap.add_argument("--pad-cache", default="cache_spatial_pad")
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", default="results/spatial_crop_control.json")
    args = ap.parse_args()

    report = {}
    for m in args.models:
        report[m] = {k: summarize(m, root, args.severities, args.frac_tune, args.split_seed)
                     for k, root in (("resize196", args.resize_cache), ("pad", args.pad_cache))}
        r, p = report[m]["resize196"], report[m]["pad"]
        print(f"\n=== {m}: spatial TTA {r['tta_acc']:.2f} (resize-196) vs {p['tta_acc']:.2f} (pad), "
              f"original {r['orig_acc']:.2f}")
        for n in VIEW_NAMES[1:]:
            print(f"  {n:8s} {r['view_minus_orig'][n]:+6.2f} {p['view_minus_orig'][n]:+6.2f}")
        for f in FAMILIES:
            print(f"  TTA gain {f:8s} {r['tta_gain_by_family'][f]:+6.2f} {p['tta_gain_by_family'][f]:+6.2f}")
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
