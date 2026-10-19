"""Evaluate several tuning JSONs together, sharing cache loads and GPU signals.

This is the grouped path for ablations where several methods are evaluated on the
same probability cache.  Each condition is decompressed once, pairwise JSD is
computed once on the GPU, and every requested method is scored before the
condition is released.

Example
-------
python scripts/evaluate_many.py \
  --model resnet18 --cache-dir cache_fixed_fresh \
  --method fixed_grid results/...grid.json \
  --method fixed_ce results/...ce.json \
  --method fixed_ce_bias results/...bias.json \
  --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import cache
from src.evaluation import device_from_arg, score_many_torch
from src.metrics import FusionConfig
from src.splits import CORRUPTION_FAMILY, tune_test_masks
from scripts.evaluate import _report


def make_picker(tuning_path):
    rec = json.loads(Path(tuning_path).read_text())
    held = {
        fam: FusionConfig.from_dict(entry["config"])
        for fam, entry in rec.get("held_out", {}).items()
    }
    global_cfg = FusionConfig.from_dict(rec["global"]["config"])

    def pick(corruption):
        fam = CORRUPTION_FAMILY.get(corruption)
        return held.get(fam, global_cfg)

    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet18", "deit"])
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument(
        "--method", nargs=2, action="append", required=True,
        metavar=("TAG", "TUNING_JSON"),
        help="repeat for every method to evaluate on this cache",
    )
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--eval-half", choices=["test", "all"], default="test")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--ladder-len", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--signal-chunk", type=int, default=256)
    ap.add_argument("--signal-pair-batch", type=int, default=32)
    ap.add_argument("--out", default="results")
    ap.add_argument("--manifest-out", default=None)
    ap.add_argument(
        "--correctness-out", default=None,
        help="optional npz of per-image fused/original correctness for later bootstrap",
    )
    ap.add_argument("--quiet-progress", action="store_true")
    args = ap.parse_args()

    methods = []
    seen = set()
    for tag, path in args.method:
        if tag in seen:
            ap.error(f"duplicate method tag: {tag}")
        if not Path(path).exists():
            ap.error(f"tuning json does not exist: {path}")
        seen.add(tag)
        methods.append((tag, path, make_picker(path)))

    cache.CACHE_ROOT = Path(args.cache_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(args.device)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = {tag: [] for tag, _, _ in methods}
    clean_rows = {tag: None for tag, _, _ in methods}
    correctness = {}
    t0 = time.time()
    n_conditions = 0

    print(
        f"Grouped evaluation: model={args.model} methods={len(methods)} "
        f"device={device} cache={args.cache_dir}"
    )

    for corruption in CORRUPTION_FAMILY:
        for sev in args.severities:
            if not cache.has_condition(args.model, corruption, sev):
                continue
            tc = time.time()
            d = cache.load_condition(args.model, corruption, sev)
            if args.eval_half == "test":
                _, mask = tune_test_masks(d["paths"], args.frac_tune, args.split_seed)
            else:
                mask = np.ones(len(d["labels"]), dtype=bool)

            configs = {tag: pick(corruption) for tag, _, pick in methods}
            if args.correctness_out:
                scored, corr = score_many_torch(
                    d["probs"][mask], d["labels"][mask], configs,
                    ladder_len=args.ladder_len,
                    device=device,
                    signal_chunk=args.signal_chunk,
                    signal_pair_batch=args.signal_pair_batch,
                    return_correctness=True,
                )
                key0 = f"{corruption}__{sev}"
                correctness[f"{key0}__orig"] = corr["__orig__"]
                # Store paths too so a later cross-cache bootstrap can align by path.
                correctness[f"{key0}__paths"] = np.asarray(d["paths"])[mask]
                for tag in configs:
                    correctness[f"{key0}__{tag}"] = corr[tag]
            else:
                scored = score_many_torch(
                    d["probs"][mask], d["labels"][mask], configs,
                    ladder_len=args.ladder_len,
                    device=device,
                    signal_chunk=args.signal_chunk,
                    signal_pair_batch=args.signal_pair_batch,
                )

            for tag, _, _ in methods:
                r = scored[tag]
                r.update(
                    model=args.model,
                    corruption=corruption,
                    family=CORRUPTION_FAMILY[corruption],
                    severity=sev,
                )
                rows[tag].append(r)

            n_conditions += 1
            if not args.quiet_progress:
                accs = "  ".join(f"{tag}={100*scored[tag]['fused_acc']:.2f}%" for tag, _, _ in methods)
                print(f"  {corruption:18s}/{sev}: {accs}  [{time.time()-tc:.2f}s]")

    # Clean uses all images and the global config for each tuning JSON.
    if cache.has_condition(args.model, "clean", 0):
        tc = time.time()
        d = cache.load_condition(args.model, "clean", 0)
        configs = {tag: pick("clean") for tag, _, pick in methods}
        scored = score_many_torch(
            d["probs"], d["labels"], configs,
            ladder_len=args.ladder_len,
            device=device,
            signal_chunk=args.signal_chunk,
            signal_pair_batch=args.signal_pair_batch,
        )
        for tag in configs:
            clean_rows[tag] = scored[tag]
        if not args.quiet_progress:
            print(f"  {'clean':18s}/0: scored {len(configs)} methods  [{time.time()-tc:.2f}s]")

    manifest = {
        "model": args.model,
        "cache_dir": args.cache_dir,
        "device": str(device),
        "n_conditions": n_conditions,
        "elapsed_s": time.time() - t0,
        "evaluations": {},
    }
    for tag, tuning_path, _ in methods:
        csv_path, json_path = _report(
            args.model, rows[tag], clean_rows[tag], out_dir, stamp, tag
        )
        manifest["evaluations"][tag] = {
            "tuning": str(tuning_path),
            "csv": str(csv_path),
            "json": str(json_path),
        }

    if args.correctness_out:
        out = Path(args.correctness_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Correctness arrays are tiny; compressed output is cheap here.
        np.savez_compressed(out, **correctness)
        manifest["correctness_npz"] = str(out)
        print("saved", out)

    manifest_path = Path(args.manifest_out) if args.manifest_out else (
        out_dir / f"eval_group_{args.model}_{Path(args.cache_dir).name}_{stamp}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"grouped evaluation finished in {manifest['elapsed_s']:.1f}s "
        f"for {len(methods)} methods x {n_conditions} conditions"
    )
    print("saved", manifest_path)


if __name__ == "__main__":
    main()
