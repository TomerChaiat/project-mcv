"""Held-out check on ImageNet-C's 4 "extra" corruptions, which no tuning or
selection step ever saw (speckle noise, Gaussian blur, spatter, saturate).

Frozen configs, no re-tuning: each extra corruption is scored with the
selected method's leave-one-family-out config of its family
(src.splits.EXTRA_FAMILY), i.e. a config tuned without that family. Test half,
5 severities, paired cluster bootstrap over images (src/bootstrap.py), per
corruption and over all four.

Needs the extra corruptions in cache_fixed/ and cache_spatial/ (the combined
cache is virtual):

    python scripts/fetch_imagenet_c.py --tarballs extra
    python scripts/cache_profiles.py --models resnet18 deit convnext --ladder fixed \\
        --cache-dir cache_fixed --corruptions speckle_noise gaussian_blur spatter saturate
    python scripts/cache_spatial_profiles.py --models resnet18 deit convnext \\
        --cache-dir cache_spatial --corruptions speckle_noise gaussian_blur spatter saturate
    python scripts/evaluate_extra.py --out results/extra_corruptions.json
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import cache
from src.bootstrap import cluster_bootstrap, image_keys
from src.evaluation import device_from_arg, fused_correct_many_torch
from src.metrics import FusionConfig, top1
from src.splits import EXTRA_FAMILY, tune_test_masks

#: combined-profile view indices of the 8-view spatial TTA (original + 7 spatial)
TTA_VIEWS = [0] + list(range(8, 15))
#: (A, B, label) pairs bootstrapped per corruption and overall
DELTAS = [("spectral", "orig", "spectral - orig"), ("combined", "orig", "combined - orig"),
          ("tta", "orig", "spatial TTA - orig"), ("spectral", "tta", "spectral - spatial TTA"),
          ("combined", "tta", "combined - spatial TTA"), ("combined", "equal", "combined - equal avg")]


def final_configs(results, model, tag):
    """Family -> FusionConfig of the newest frozen (selected) tuning for this tag."""
    rx = re.compile(rf"tuning_{model}_{tag}_final_\d{{8}}_\d{{6}}\.json$")
    files = sorted(f for f in glob.glob(f"{results}/tuning_{model}_{tag}_final_*.json") if rx.search(f))
    if not files:
        raise SystemExit(f"no frozen tuning for {model} {tag} in {results}")
    t = json.loads(Path(files[-1]).read_text())
    return {fam: FusionConfig.from_dict(v["config"]) for fam, v in t["held_out"].items()}, files[-1], t.get("method")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit", "convnext"])
    ap.add_argument("--corruptions", nargs="+", default=list(EXTRA_FAMILY))
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--results", default="results")
    ap.add_argument("--spectral-cache", default="cache_fixed")
    ap.add_argument("--combined-cache", default="cache_combined")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/extra_corruptions.json")
    args = ap.parse_args()
    device = device_from_arg(args.device)

    report = {"protocol": "frozen selected configs; each extra corruption scored with the LOFO "
                          "config of its family (tuned without it); test half",
              "family": {c: EXTRA_FAMILY[c] for c in args.corruptions}, "models": {}}
    for m in args.models:
        spec_cfg, spec_src, spec_meth = final_configs(args.results, m, "cache_fixed_lofo")
        comb_cfg, comb_src, comb_meth = final_configs(args.results, m, "cache_combined_lofo")
        conds = []
        for c in args.corruptions:
            fam = EXTRA_FAMILY[c]
            for s in args.severities:
                cache.CACHE_ROOT = Path(args.combined_cache)
                if not cache.has_condition(m, c, s):
                    raise SystemExit(f"missing cache: {m} {c}/{s} - see the docstring")
                d = cache.load_condition(m, c, s)
                _, mask = tune_test_masks(d["paths"])
                probs = np.ascontiguousarray(d["probs"][mask]).astype(np.float32)
                labels = d["labels"][mask]
                # the combined record's first 8 views are the spectral profile, bit-exact
                spec_ok, orig_ok = fused_correct_many_torch(
                    probs[:, :8], labels, {"f": spec_cfg[fam]}, device=device)
                comb_ok, _ = fused_correct_many_torch(
                    probs, labels, {"f": comb_cfg[fam]}, ladder_len=8, device=device)
                view_ok = probs.argmax(2) == labels[:, None]
                conds.append({
                    "corruption": c, "severity": s, "keys": image_keys(np.asarray(d["paths"])[mask]),
                    "orig": np.asarray(orig_ok, bool), "spectral": np.asarray(spec_ok["f"], bool),
                    "combined": np.asarray(comb_ok["f"], bool),
                    "tta": top1(probs[:, TTA_VIEWS].mean(1)) == labels,
                    "equal": top1(probs.mean(1)) == labels,
                    "view_acc": view_ok.mean(0)})

        rec = {"spectral_config": spec_src, "spectral_method": spec_meth,
               "combined_config": comb_src, "combined_method": comb_meth, "per_corruption": {}}
        print(f"\n=== {m}  (spectral: {spec_meth}, combined: {comb_meth})")
        for c in args.corruptions + ["all"]:
            sub = [x for x in conds if c == "all" or x["corruption"] == c]
            acc = {k: float(100 * np.mean([x[k].mean() for x in sub]))
                   for k in ("orig", "spectral", "combined", "tta", "equal")}
            views = 100 * np.mean([x["view_acc"] for x in sub], 0)
            r = {"acc": acc, "lowpass_view_minus_orig": (views[1:8] - views[0]).tolist(), "deltas": {}}
            for i, (a, b, label) in enumerate(DELTAS):
                bs = cluster_bootstrap([{"keys": x["keys"], "a": x[a], "b": x[b]} for x in sub],
                                       n_boot=args.n_boot, seed=args.seed + i)
                r["deltas"][label] = bs
            rec["per_corruption"][c] = r
            dd = r["deltas"]
            fmt = lambda k: f"{100 * dd[k]['point']:+.2f} [{100 * dd[k]['ci95'][0]:+.2f},{100 * dd[k]['ci95'][1]:+.2f}]"
            print(f"  {c:14s} orig {acc['orig']:.2f}  spec {fmt('spectral - orig')}  "
                  f"comb {fmt('combined - orig')}  spec-TTA {fmt('spectral - spatial TTA')}  "
                  f"best LP view {max(r['lowpass_view_minus_orig']):+.2f}")
        report["models"][m] = rec

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
