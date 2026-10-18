"""One-pass bootstrap for final fused vs original and spatial TTA.

The final runner used to execute two independent bootstrap jobs.  Both jobs
reloaded the combined cache and recomputed the final fused predictor.  This
script computes the fused correctness once per condition, reuses it for both
comparisons, and uses the vectorized exact paired bootstrap from
``bootstrap_pairwise.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import cache
from src.evaluation import device_from_arg, fused_correct_many_torch
from src.metrics import FusionConfig
from src.splits import CORRUPTION_FAMILY, tune_test_masks
from scripts.bootstrap_pairwise import bootstrap


def config_picker(path):
    rec = json.loads(Path(path).read_text())
    held = {
        fam: FusionConfig.from_dict(v["config"])
        for fam, v in rec.get("held_out", {}).items()
    }
    glob = FusionConfig.from_dict(rec["global"]["config"])

    def pick(corruption):
        return held.get(CORRUPTION_FAMILY.get(corruption), glob)
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet18", "deit"])
    ap.add_argument("--combined-cache", required=True)
    ap.add_argument("--spatial-cache", required=True)
    ap.add_argument("--tuning", required=True)
    ap.add_argument("--ladder-len", type=int, default=8)
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--signal-chunk", type=int, default=256)
    ap.add_argument("--signal-pair-batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-orig", default=None, help="optional old-style pairwise JSON")
    ap.add_argument("--out-spatial", default=None, help="optional old-style pairwise JSON")
    ap.add_argument("--quiet-progress", action="store_true")
    args = ap.parse_args()

    device = device_from_arg(args.device)
    pick = config_picker(args.tuning)
    vs_orig = []
    vs_spatial = []
    t0 = time.time()

    for corruption in CORRUPTION_FAMILY:
        for sev in args.severities:
            tc = time.time()
            cache.CACHE_ROOT = Path(args.combined_cache)
            if not cache.has_condition(args.model, corruption, sev):
                continue
            dc = cache.load_condition(args.model, corruption, sev)
            _, mask_c = tune_test_masks(dc["paths"], args.frac_tune, args.split_seed)
            paths_c = np.asarray(dc["paths"])[mask_c]
            probs_c = np.ascontiguousarray(dc["probs"][mask_c])
            labels_c = np.asarray(dc["labels"])[mask_c]

            corr, orig = fused_correct_many_torch(
                probs_c, labels_c, {"fused": pick(corruption)},
                ladder_len=args.ladder_len,
                device=device,
                signal_chunk=args.signal_chunk,
                signal_pair_batch=args.signal_pair_batch,
            )
            fused = corr["fused"]
            vs_orig.append({
                "corruption": corruption,
                "severity": sev,
                "a": fused,
                "b": orig,
                "n": len(fused),
            })

            cache.CACHE_ROOT = Path(args.spatial_cache)
            if cache.has_condition(args.model, corruption, sev):
                ds = cache.load_spatial_condition(args.model, corruption, sev)
                _, mask_s = tune_test_masks(ds["paths"], args.frac_tune, args.split_seed)
                paths_s = np.asarray(ds["paths"])[mask_s]
                probs_s = ds["probs"][mask_s].astype(np.float32)
                labels_s = np.asarray(ds["labels"])[mask_s]
                spatial = probs_s.mean(axis=1).argmax(-1) == labels_s

                if np.array_equal(paths_c, paths_s):
                    fa = fused
                    sb = spatial
                else:
                    pos = {str(p): i for i, p in enumerate(paths_s)}
                    common_i = [i for i, p in enumerate(paths_c) if str(p) in pos]
                    fa = fused[common_i]
                    sb = np.asarray([spatial[pos[str(paths_c[i])]] for i in common_i], dtype=bool)
                if len(fa):
                    vs_spatial.append({
                        "corruption": corruption,
                        "severity": sev,
                        "a": np.asarray(fa, bool),
                        "b": np.asarray(sb, bool),
                        "n": len(fa),
                    })

            if not args.quiet_progress:
                print(f"  {corruption:18s}/{sev}: n={len(fused)}  [{time.time()-tc:.2f}s]")

    if not vs_orig:
        raise SystemExit("no combined conditions found")
    orig_stats = bootstrap(vs_orig, args.n_boot, args.seed)
    spatial_stats = bootstrap(vs_spatial, args.n_boot, args.seed + 1) if vs_spatial else None

    def pack(stats, n_conditions):
        point, lo, hi, p = stats
        return {
            "n_conditions": n_conditions,
            "point": point,
            "ci95": [lo, hi],
            "p_le_zero": p,
        }

    report = {
        "model": args.model,
        "combined_cache": args.combined_cache,
        "spatial_cache": args.spatial_cache,
        "tuning": args.tuning,
        "n_boot": args.n_boot,
        "fused_minus_original": pack(orig_stats, len(vs_orig)),
        "fused_minus_spatial": pack(spatial_stats, len(vs_spatial)) if spatial_stats else None,
        "elapsed_s": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    if args.out_orig:
        orec = report["fused_minus_original"]
        Path(args.out_orig).write_text(json.dumps({
            "model": args.model,
            "a": {"mode": "fused", "cache": args.combined_cache, "tuning": args.tuning},
            "b": {"mode": "original", "cache": args.combined_cache, "tuning": None},
            "n_conditions": orec["n_conditions"], "n_boot": args.n_boot,
            "point": orec["point"], "ci95": orec["ci95"],
            "p_le_zero": orec["p_le_zero"],
        }, indent=2))
    if args.out_spatial and report["fused_minus_spatial"]:
        srec = report["fused_minus_spatial"]
        Path(args.out_spatial).write_text(json.dumps({
            "model": args.model,
            "a": {"mode": "fused", "cache": args.combined_cache, "tuning": args.tuning},
            "b": {"mode": "spatial", "cache": args.spatial_cache, "tuning": None},
            "n_conditions": srec["n_conditions"], "n_boot": args.n_boot,
            "point": srec["point"], "ci95": srec["ci95"],
            "p_le_zero": srec["p_le_zero"],
        }, indent=2))

    o = report["fused_minus_original"]
    print(f"fused - original: {100*o['point']:+.3f} pts  95% CI [{100*o['ci95'][0]:+.3f}, {100*o['ci95'][1]:+.3f}]")
    if report["fused_minus_spatial"]:
        s = report["fused_minus_spatial"]
        print(f"fused - spatial : {100*s['point']:+.3f} pts  95% CI [{100*s['ci95'][0]:+.3f}, {100*s['ci95'][1]:+.3f}]")
    print(f"elapsed {report['elapsed_s']:.1f}s")
    print("saved", args.out)


if __name__ == "__main__":
    main()
