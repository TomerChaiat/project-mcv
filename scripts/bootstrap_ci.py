"""Paired bootstrap confidence intervals for the headline deltas: does fusion
really beat the original classifier, and how does it really compare to the
spatial-TTA baseline? evaluate.py reports point estimates only; this adds
error bars to the same statistic (mean accuracy over corruptions x severities,
each condition itself a mean over its held-out images).

Resampling is a paired cluster bootstrap over images (src/bootstrap.py): one
draw of validation images per replicate, applied to every (corruption,
severity) condition at once - the same images appear in all 75 conditions, so
resampling each condition independently would understate the variance. The
same draw is used for both predictors being compared, so the bootstrap
targets the paired difference directly.

    python scripts/bootstrap_ci.py                      # both models, cache_fixed
    python scripts/bootstrap_ci.py --models resnet18 --n-boot 5000
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import cache
from src.bootstrap import cluster_bootstrap, image_keys
from src.metrics import BASELINE_TAU, FusionConfig, top1
from src.evaluation import device_from_arg, fused_correct_many_torch
from src.splits import CORRUPTION_FAMILY, tune_test_masks


def latest_tuning(model_name, tag, results_dir="results"):
    """Path of the newest <results_dir>/tuning_<model>[_<tag>]_*.json, or exit."""
    pattern = (f"{results_dir}/tuning_{model_name}_{tag}_*.json" if tag
               else f"{results_dir}/tuning_{model_name}_2*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no {pattern} found -- run tune_fusion.py first")
    return files[-1]


def config_picker(model_name, tag, results_dir="results"):
    """Build the corruption -> FusionConfig mapping from the newest tuning
    json (same held-out-family-or-global logic as evaluate.py's).

    Returns:
        (pick, path) where `pick(corruption)` -> FusionConfig and `path` is
        the tuning json it came from.
    """
    path = latest_tuning(model_name, tag, results_dir)
    tune = json.loads(Path(path).read_text())
    held = {fam: FusionConfig.from_dict(tune["held_out"][fam]["config"])
            for fam in tune["held_out"]}
    glob_cfg = FusionConfig.from_dict(tune["global"]["config"])

    def pick(corruption):
        fam = CORRUPTION_FAMILY.get(corruption)
        return held.get(fam, glob_cfg)

    return pick, path


#: training-free fusion baselines as fusion configs (same definitions as
#: score_many_torch's equal_acc / conf_acc / agree_acc), so they share the
#: fused predictor's signal pass
BASELINE_CONFIGS = {
    "equal": FusionConfig(0.0, 0.0, 0.0, BASELINE_TAU, True, 1, "all"),
    "conf": FusionConfig(1.0, 0.0, 0.0, BASELINE_TAU, True, 1, "all"),
    "agree": FusionConfig(0.0, 1.0, 0.0, BASELINE_TAU, True, 1, "all"),
}
#: (key, label) of every predictor bootstrapped against the original
VS_ORIG = [("view1", "mild LP (0.90) - orig"), ("equal", "equal avg - orig"),
           ("conf", "confidence-only - orig"), ("agree", "agreement-only - orig")]


def per_image_correctness(model_name, cache_dir, tag, ladder_len, severities,
                          frac_tune, split_seed, *, device,
                          signal_chunk=256, signal_pair_batch=32,
                          results_dir="results", spatial_cache="cache_spatial"):
    """Fast orig/fused/(optional spatial) per-image correctness.

    Fusion signals run on torch/CUDA.
    """
    cache.CACHE_ROOT = Path(cache_dir)
    pick, tuning_path = config_picker(model_name, tag, results_dir)

    spatial_root = Path(spatial_cache)
    has_spatial = spatial_root.exists()
    conditions = []
    for corruption in CORRUPTION_FAMILY:
        for sev in severities:
            cache.CACHE_ROOT = Path(cache_dir)
            if not cache.has_condition(model_name, corruption, sev):
                continue
            d = cache.load_condition(model_name, corruption, sev)
            _, mask = tune_test_masks(d["paths"], frac_tune, split_seed)
            probs = np.ascontiguousarray(d["probs"][mask])
            labels = d["labels"][mask]
            cfg = pick(corruption)
            fused_map, orig_correct = fused_correct_many_torch(
                probs, labels, {"fused": cfg, **BASELINE_CONFIGS}, ladder_len=ladder_len,
                device=device, signal_chunk=signal_chunk,
                signal_pair_batch=signal_pair_batch,
            )
            fused_correct = fused_map["fused"]

            tta_correct = None
            if has_spatial:
                cache.CACHE_ROOT = spatial_root
                if cache.has_condition(model_name, corruption, sev):
                    ds = cache.load_spatial_condition(model_name, corruption, sev)
                    paths = list(d["paths"])
                    pos = {p: i for i, p in enumerate(ds["paths"])}
                    idx = [pos[p] for p in paths]
                    sp = ds["probs"][idx][mask].astype(np.float32)
                    tta_correct = top1(sp.mean(axis=1)) == labels

            conditions.append({"keys": image_keys(np.asarray(d["paths"])[mask]),
                               "orig": np.asarray(orig_correct, bool),
                               "fused": np.asarray(fused_correct, bool),
                               "tta": tta_correct, "n": int(mask.sum()),
                               "view1": top1(probs[:, 1]) == labels,
                               **{k: np.asarray(fused_map[k], bool) for k in BASELINE_CONFIGS}})
    return conditions, tuning_path


def bootstrap_delta(conditions, key_a, key_b, n_boot, seed):
    """Paired cluster bootstrap of key_a - key_b (see src/bootstrap.py), or None."""
    usable = [{"keys": c["keys"], "a": c[key_a], "b": c[key_b]}
              for c in conditions if c[key_a] is not None and c[key_b] is not None]
    if not usable:
        return None
    return cluster_bootstrap(usable, n_boot=n_boot, seed=seed)


def main():
    """Compute and print the bootstrap CIs for one cache, both models."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"],
                    choices=["resnet18", "deit", "convnext"])
    ap.add_argument("--cache-dir", default="cache_fixed")
    ap.add_argument("--tag", default="cache_fixed")
    ap.add_argument("--ladder-len", type=int, default=None)
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--signal-chunk", type=int, default=256)
    ap.add_argument("--signal-pair-batch", type=int, default=32)
    ap.add_argument("--out", default="results")
    ap.add_argument("--results-dir", default="results",
                    help="where to look for the tuning_<model>_<tag>_*.json")
    ap.add_argument("--spatial-cache", default="cache_spatial")
    ap.add_argument("--suffix", default="",
                    help="appended to the report name, bootstrap_ci_<tag><suffix>.json, "
                         "e.g. to keep a run against another --spatial-cache separate")
    args = ap.parse_args()

    device = device_from_arg(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for model_name in args.models:
        print(f"\n=== {model_name}  ({args.cache_dir}) ===")
        conditions, tuning_path = per_image_correctness(
            model_name, args.cache_dir, args.tag, args.ladder_len,
            args.severities, args.frac_tune, args.split_seed,
            device=device, signal_chunk=args.signal_chunk,
            signal_pair_batch=args.signal_pair_batch,
            results_dir=args.results_dir, spatial_cache=args.spatial_cache,
        )
        print(f"  {len(conditions)} conditions, config: {tuning_path}")

        model_report = {"tuning_config": tuning_path, "n_boot": args.n_boot}
        pairs = ([("fused", "orig", "fused - orig"), ("fused", "tta", "fused - spatial TTA"),
                  ("tta", "orig", "spatial TTA - orig")]
                 + [(k, "orig", label) for k, label in VS_ORIG])
        for i, (key_a, key_b, label) in enumerate(pairs):
            res = bootstrap_delta(conditions, key_a, key_b, args.n_boot, args.seed + i)
            if res is None:
                print(f"  {label:22s}: n/a (no spatial cache)")
                continue
            lo, hi = res["ci95"]
            print(f"  {label:22s}: {res['point']*100:+.2f} pts   95% CI [{lo*100:+.2f}, {hi*100:+.2f}]   "
                  f"P(delta <= 0) = {res['p_le_zero']:.3f}  ({res['n_conditions']} conditions, "
                  f"{res['n_images']} images)")
            model_report[label] = res
        report[model_name] = model_report

    # Merge into an existing report so models bootstrapped in separate runs
    # (e.g. convnext after resnet18/deit) don't overwrite each other.
    path = out_dir / f"bootstrap_ci_{args.tag}{args.suffix}.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(report)
    path.write_text(json.dumps(merged, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
