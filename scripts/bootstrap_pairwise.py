"""Paired bootstrap for any two cached predictors.

- if A and B use the same cache, each condition is loaded only once;
- if several fused predictors share that cache, pairwise JSD/signals are
  computed once on the GPU for all of them;
- resampling is a paired *cluster* bootstrap over images (src/bootstrap.py):
  the same validation images appear in every condition, so one draw of images
  is applied to all conditions at once.
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
from src.bootstrap import cluster_bootstrap, image_keys
from src.evaluation import device_from_arg, fused_correct_many_torch
from src.metrics import FusionConfig
from src.splits import CORRUPTION_FAMILY, tune_test_masks


def config_picker(tuning_path):
    """Build a corruption -> FusionConfig picker from a tuning json.

    Returns:
        `pick(corruption)` -> the family's held-out FusionConfig, or the
        global one if that family wasn't held out.
    """
    tune = json.loads(Path(tuning_path).read_text())
    held = {
        fam: FusionConfig.from_dict(entry["config"])
        for fam, entry in tune.get("held_out", {}).items()
    }
    global_cfg = FusionConfig.from_dict(tune["global"]["config"])

    def pick(corruption):
        """FusionConfig for `corruption` (held-out family config, else global)."""
        fam = CORRUPTION_FAMILY.get(corruption)
        return held.get(fam, global_cfg)
    return pick


def _load_condition(cache_dir, model, corruption, sev):
    """Load one cached (corruption, severity) condition from `cache_dir`.

    Returns:
        The cache dict (paths, probs, labels, ...), or None if not cached.
    """
    cache.CACHE_ROOT = Path(cache_dir)
    if not cache.has_condition(model, corruption, sev):
        return None
    return cache.load_condition(model, corruption, sev)


def _masked_record(d, frac_tune, split_seed):
    """Restrict a loaded condition to its held-out (test) images.

    Returns:
        (paths, probs, labels) for the test split only.
    """
    _, mask = tune_test_masks(d["paths"], frac_tune, split_seed)
    return (
        np.asarray(d["paths"])[mask],
        np.ascontiguousarray(d["probs"][mask]),
        np.asarray(d["labels"])[mask],
    )


def _simple_correct(mode, probs, labels):
    """Per-image correctness of a non-fused predictor: "original" (view 0) or
    "spatial" (mean over views).

    Returns:
        Bool array, True where the prediction matches the label.
    """
    if mode == "original":
        pred = probs[:, 0].astype(np.float32).argmax(-1)
    elif mode == "spatial":
        pred = probs.astype(np.float32).mean(axis=1).argmax(-1)
    else:
        raise ValueError(mode)
    return pred == labels


def _same_cache_pair(args, corruption, sev, pick_a, pick_b, device):
    """Correctness of A and B on one condition when both read the same cache;
    the condition is loaded once and fused signals are computed once for both.

    Returns:
        (paths, correct_a, correct_b) on the test split, or None if the
        condition isn't cached.
    """
    d =_load_condition(args.a_cache, args.model, corruption, sev)
    if d is None:
        return None
    paths, probs, labels = _masked_record(d, args.frac_tune, args.split_seed)

    configs = {}
    if args.a_mode == "fused":
        configs["a"] = pick_a(corruption)
    if args.b_mode == "fused":
        configs["b"] = pick_b(corruption)
    fused = {}
    if configs:
        # A/B ladder lengths must agree when they share one probability tensor.
        lens = {
            args.a_ladder_len if args.a_mode == "fused" else None,
            args.b_ladder_len if args.b_mode == "fused" else None,
        }
        lens.discard(None)
        if len(lens) > 1:
            raise ValueError("same-cache fused comparison uses conflicting ladder lengths")
        ladder_len = next(iter(lens)) if lens else None
        fused, _orig = fused_correct_many_torch(
            probs, labels, configs,
            ladder_len=ladder_len,
            device=device,
            signal_chunk=args.signal_chunk,
            signal_pair_batch=args.signal_pair_batch,
        )

    ca = fused["a"] if args.a_mode == "fused" else _simple_correct(args.a_mode, probs, labels)
    cb = fused["b"] if args.b_mode == "fused" else _simple_correct(args.b_mode, probs, labels)
    return paths, np.asarray(ca, bool), np.asarray(cb, bool)


def _one_side(mode, cache_dir, tuning_picker, ladder_len, args, corruption, sev, device):
    """Correctness of a single predictor (fused/original/spatial) on one condition.

    Returns:
        (paths, correct) on the test split, or None if the condition isn't cached.
    """
    d =_load_condition(cache_dir, args.model, corruption, sev)
    if d is None:
        return None
    paths, probs, labels = _masked_record(d, args.frac_tune, args.split_seed)
    if mode == "fused":
        cfg = tuning_picker(corruption)
        corr, _ = fused_correct_many_torch(
            probs, labels, {"x": cfg},
            ladder_len=ladder_len,
            device=device,
            signal_chunk=args.signal_chunk,
            signal_pair_batch=args.signal_pair_batch,
        )
        correct = corr["x"]
    else:
        correct = _simple_correct(mode, probs, labels)
    return paths, np.asarray(correct, bool)


def paired_conditions(args, device):
    """Per-image A/B correctness for every (corruption, severity) available to
    both sides, aligned by image path when the caches differ.

    Returns:
        List of dicts with corruption, severity, keys (image ids), a, b
        (bool correctness arrays) and n.
    """
    pick_a = config_picker(args.a_tuning) if args.a_mode == "fused" else None
    pick_b = config_picker(args.b_tuning) if args.b_mode == "fused" else None
    out = []
    same_cache = Path(args.a_cache).resolve() == Path(args.b_cache).resolve()

    for corruption in CORRUPTION_FAMILY:
        for sev in args.severities:
            t0 = time.time()
            if same_cache:
                rec = _same_cache_pair(args, corruption, sev, pick_a, pick_b, device)
                if rec is None:
                    continue
                paths, aa, bb = rec
                keys = image_keys(paths)
                n = len(aa)
            else:
                a = _one_side(
                    args.a_mode, args.a_cache, pick_a, args.a_ladder_len,
                    args, corruption, sev, device,
                )
                b = _one_side(
                    args.b_mode, args.b_cache, pick_b, args.b_ladder_len,
                    args, corruption, sev, device,
                )
                if a is None or b is None:
                    continue
                pa, ca = a
                pb, cb = b
                # Align by path only when caches differ.
                map_b = {str(p): bool(v) for p, v in zip(pb, cb)}
                common = [(str(p), bool(v)) for p, v in zip(pa, ca) if str(p) in map_b]
                if not common:
                    continue
                aa = np.asarray([v for _, v in common], dtype=bool)
                bb = np.asarray([map_b[p] for p, _ in common], dtype=bool)
                keys = image_keys([p for p, _ in common])
                n = len(aa)

            out.append({
                "corruption": corruption,
                "severity": sev,
                "keys": keys,
                "a": aa,
                "b": bb,
                "n": n,
            })
            if not args.quiet_progress:
                print(f"  {corruption:18s}/{sev}: n={n}  [{time.time()-t0:.2f}s]")
    return out


def bootstrap(conditions, n_boot, seed):
    """Paired cluster bootstrap over images; returns (point, lo, hi, p_le_zero).

    Each condition record needs 'keys' (image ids), 'a' and 'b'.
    """
    out = cluster_bootstrap(conditions, n_boot=n_boot, seed=seed)
    lo, hi = out["ci95"]
    return out["point"], lo, hi, out["p_le_zero"]


def main():
    """Bootstrap the A - B accuracy delta, print it and optionally save a json report."""
    ap =argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet18", "deit", "convnext"])
    for side in ("a", "b"):
        ap.add_argument(f"--{side}-mode", choices=["fused", "original", "spatial"], required=True)
        ap.add_argument(f"--{side}-cache", required=True)
        ap.add_argument(f"--{side}-tuning", default=None)
        ap.add_argument(f"--{side}-ladder-len", type=int, default=None)
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--signal-chunk", type=int, default=256)
    ap.add_argument("--signal-pair-batch", type=int, default=32)
    ap.add_argument("--quiet-progress", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.a_mode == "fused" and not args.a_tuning:
        ap.error("--a-mode fused requires --a-tuning")
    if args.b_mode == "fused" and not args.b_tuning:
        ap.error("--b-mode fused requires --b-tuning")

    device = device_from_arg(args.device)
    t0 = time.time()
    conditions = paired_conditions(args, device)
    if not conditions:
        raise SystemExit("no paired conditions found")
    point, lo, hi, p_le_zero = bootstrap(conditions, args.n_boot, args.seed)

    report = {
        "model": args.model,
        "a": {"mode": args.a_mode, "cache": args.a_cache, "tuning": args.a_tuning},
        "b": {"mode": args.b_mode, "cache": args.b_cache, "tuning": args.b_tuning},
        "n_conditions": len(conditions),
        "n_boot": args.n_boot,
        "point": point,
        "ci95": [lo, hi],
        "p_le_zero": p_le_zero,
        "elapsed_s": time.time() - t0,
        "bootstrap_impl": "paired cluster bootstrap over images (src/bootstrap.py)",
    }
    print(
        f"A - B: {point*100:+.3f} pts  95% CI [{lo*100:+.3f}, {hi*100:+.3f}]  "
        f"P(delta<=0)={p_le_zero:.4f}  ({len(conditions)} conditions)"
    )
    print(f"elapsed {report['elapsed_s']:.1f}s")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("saved", args.out)


if __name__ == "__main__":
    main()
