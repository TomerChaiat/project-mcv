"""Run the canonical spectral ablation on existing caches.

Default protocol:
  - fixed + geometric caches
  - ResNet-18 + DeiT-Tiny
  - tuning severities 1..5 for every method
  - CUDA grid baseline
  - CE direction + accuracy scale + output temperature
  - optional mean-zero view-bias extension
  - optional unconstrained-theta diagnostic

No image/model inference is performed; this consumes existing probability caches.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def run(cmd, *, verbose=False, label=None):
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = [str(x) for x in cmd]
    if verbose:
        print("\n$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)
        return
    if label:
        print(label, end="", flush=True)
    import time
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="backslashreplace")
    if proc.returncode != 0:
        if label:
            print(" FAILED")
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    if label:
        print(f" done ({time.time()-t0:.1f}s)")


def matching_json(pattern, severities, *, method=None):
    for path in reversed(sorted(glob.glob(str(pattern)))):
        try:
            rec = json.loads(Path(path).read_text())
        except Exception:
            continue
        if list(rec.get("tune_severities", [])) != list(severities):
            continue
        if method is not None and rec.get("method") != method:
            continue
        return path
    return None


def ensure_grid(py, model, cache_dir, results_dir, severities, args):
    tag = Path(cache_dir).name
    patterns = [Path(results_dir) / f"tuning_grid_{model}_{tag}_*.json",
                Path(results_dir) / f"tuning_fast_{model}_{tag}_*.json"]
    found = next((matching_json(p, severities) for p in patterns if matching_json(p, severities)), None)
    if found and not args.force_tuning:
        print("reuse all-severity grid:", found)
        return found
    cmd = [
        py, "scripts/tune_fusion.py",
        "--models", model,
        "--cache-dir", cache_dir,
        "--tag", tag,
        "--out", results_dir,
        "--tune-severities", *map(str, severities),
        "--device", args.device,
        "--config-batch", str(args.config_batch),
        "--signal-chunk", str(args.signal_chunk),
        "--signal-pair-batch", str(args.signal_pair_batch),
    ]
    if args.tf32:
        cmd.append("--tf32")
    run(cmd, verbose=args.verbose, label=None if args.verbose else f"[{model}] grid tuning ............")
    found = next((matching_json(p, severities) for p in patterns if matching_json(p, severities)), None)
    if not found:
        raise RuntimeError(f"no all-severity grid result found for {model}/{tag}")
    return found


_VARIANT_TO_METHOD = {
    "base": "ce",
    "bias": "ce_bias",
    "unconstrained": "ce_unconstrained",
}


def ensure_ce_group(
    py,
    model,
    cache_dir,
    results_dir,
    severities,
    grid_path,
    args,
    variants,
    *,
    ladder_len=None,
):
    """Run all missing CE variants in one process so they share precompute/RAM.

    All requested CE variants share one signal/probability precompute.
    """
    tag = Path(cache_dir).name
    found = {}
    missing = []
    for variant in variants:
        method = _VARIANT_TO_METHOD[variant]
        pattern = Path(results_dir) / f"tuning_{method}_{model}_{tag}_*.json"
        path = None if args.force_tuning else matching_json(pattern, severities, method=method)
        if path:
            print(f"reuse {method}: {path}")
            found[variant] = path
        else:
            missing.append(variant)

    if missing:
        cmd = [
            py, "scripts/tune_fusion_ce.py",
            "--models", model,
            "--cache-dir", cache_dir,
            "--tag", tag,
            "--out", results_dir,
            "--tune-severities", *map(str, severities),
            "--grid-tuning", grid_path,
            "--l2", str(args.l2),
            "--bias-l2", str(args.bias_l2),
            "--device", args.device,
            "--signal-chunk", str(args.signal_chunk),
            "--signal-pair-batch", str(args.signal_pair_batch),
            "--optimizer-backend", args.optimizer_backend,
            "--ce-dtype", args.ce_dtype,
            "--scale-batch", str(args.scale_batch),
            "--temperature-batch", str(args.temperature_batch),
            "--resident-probs", args.resident_probs,
            "--variants", *missing,
        ]
        if ladder_len is not None:
            cmd += ["--ladder-len", str(ladder_len)]
        run(cmd, verbose=args.verbose, label=None if args.verbose else f"[{model}] CE variants .............")

        for variant in missing:
            method = _VARIANT_TO_METHOD[variant]
            pattern = Path(results_dir) / f"tuning_{method}_{model}_{tag}_*.json"
            path = matching_json(pattern, severities, method=method)
            if not path:
                raise RuntimeError(f"no {method} result matched {pattern}")
            found[variant] = path

    return found

def evaluate_grouped(py, model, cache_dir, methods, ladder, results_dir, args, ladder_len=None):
    """Evaluate every method for one model/cache in a single GPU/JSD pass per condition."""
    manifest = Path(results_dir) / f"eval_group_manifest_{model}_{ladder}.json"
    cmd = [
        py, "scripts/evaluate_many.py",
        "--model", model,
        "--cache-dir", cache_dir,
        "--out", results_dir,
        "--manifest-out", str(manifest),
        "--device", args.device,
        "--signal-chunk", str(args.signal_chunk),
        "--signal-pair-batch", str(args.signal_pair_batch),
    ]
    if ladder_len is not None:
        cmd += ["--ladder-len", str(ladder_len)]
    for method, tuning in methods:
        cmd += ["--method", f"{ladder}_{method}", tuning]
    if not args.verbose:
        cmd.append("--quiet-progress")
    run(cmd, verbose=args.verbose, label=None if args.verbose else f"[{model}] grouped evaluation ......")
    rec = json.loads(manifest.read_text())
    out = {}
    for method, _ in methods:
        tag = f"{ladder}_{method}"
        out[method] = rec["evaluations"][tag]["json"]
    return out


def summary_row(model, ladder, method, cache_dir, tuning_path, eval_path):
    t = json.loads(Path(tuning_path).read_text())
    e = json.loads(Path(eval_path).read_text())
    cfg = t.get("global", {}).get("config", {})
    tau = float(cfg.get("tau", 1.0) or 1.0)
    return {
        "model": model,
        "ladder": ladder,
        "method": method,
        "cache_dir": cache_dir,
        "tuning": tuning_path,
        "evaluation": eval_path,
        "mCA_orig": e.get("mCA_orig"),
        "mCA_fused": e.get("mCA_fused"),
        "gain_pts": 100.0 * (e["mCA_fused"] - e["mCA_orig"]),
        "mean_fused_nll": e.get("mean_fused_nll"),
        "mean_fused_ece": e.get("mean_fused_ece"),
        "negative_transfer": e.get("mean_negative_transfer"),
        "clean_fused_acc": e.get("clean_fused_acc"),
        "alpha_global": cfg.get("alpha"),
        "beta_global": cfg.get("beta"),
        "gamma_global": cfg.get("gamma"),
        "tau_global": tau,
        "alpha_over_tau_global": cfg.get("alpha") / tau if cfg.get("alpha") is not None else None,
        "beta_over_tau_global": cfg.get("beta") / tau if cfg.get("beta") is not None else None,
        "gamma_over_tau_global": cfg.get("gamma") / tau if cfg.get("gamma") is not None else None,
        "m_global": cfg.get("m"),
        "agreement_global": cfg.get("agreement"),
        "output_temperature_global": cfg.get("output_temperature", 1.0),
        "selection_scale_global": t.get("global", {}).get("selection_scale"),
        "held_out_configs": {k: v.get("config", {}) for k, v in t.get("held_out", {}).items()},
        "config_note": "corrupted evaluation uses family-specific held-out configs; global is used for clean",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"], choices=["resnet18", "deit"])
    ap.add_argument("--fixed-cache", default="cache_fixed_fresh")
    ap.add_argument("--geometric-cache", default="cache_geometric_fresh")
    ap.add_argument("--results", default="results_ablation")
    ap.add_argument("--tune-severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config-batch", type=int, default=64)
    ap.add_argument("--signal-chunk", type=int, default=256)
    ap.add_argument("--signal-pair-batch", type=int, default=32)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--bias-l2", type=float, default=1e-3)
    ap.add_argument("--optimizer-backend", choices=["auto", "torch", "numpy"], default="auto")
    ap.add_argument("--ce-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--scale-batch", type=int, default=32)
    ap.add_argument("--temperature-batch", type=int, default=49)
    ap.add_argument("--resident-probs", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--force-tuning", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-bias", action="store_true")
    ap.add_argument("--skip-unconstrained", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    Path(args.results).mkdir(parents=True, exist_ok=True)
    rows = []

    for ladder, cache_dir in (("fixed", args.fixed_cache), ("geometric", args.geometric_cache)):
        for model in args.models:
            print(f"\n===== {model}: {ladder} =====")
            grid_path = ensure_grid(py, model, cache_dir, args.results, args.tune_severities, args)
            methods = [("grid", grid_path)]

            variants = ["base"]
            if not args.skip_bias:
                variants.append("bias")
            if not args.skip_unconstrained:
                variants.append("unconstrained")

            refined_paths = ensure_ce_group(
                py, model, cache_dir, args.results, args.tune_severities,
                grid_path, args, variants
            )
            for variant in variants:
                methods.append((_VARIANT_TO_METHOD[variant], refined_paths[variant]))

            eval_paths = evaluate_grouped(
                py, model, cache_dir, methods, ladder, args.results, args
            )
            for method, tuning in methods:
                rows.append(
                    summary_row(model, ladder, method, cache_dir, tuning, eval_paths[method])
                )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.results) / f"ablation_summary_{stamp}.json"
    out_csv = Path(args.results) / f"ablation_summary_{stamp}.csv"
    out_json.write_text(json.dumps(rows, indent=2))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print("\n=== held-out spectral ablation ===")
    print(f"{'model':10s} {'ladder':10s} {'method':26s} {'mCA':>8s} {'gain':>8s} {'NLL':>8s} {'ECE':>8s} {'neg':>7s}")
    for r in rows:
        print(
            f"{r['model']:10s} {r['ladder']:10s} {r['method']:26s} "
            f"{100*r['mCA_fused']:7.2f}% {r['gain_pts']:+7.2f} "
            f"{r['mean_fused_nll']:8.4f} {r['mean_fused_ece']:8.4f} "
            f"{100*r['negative_transfer']:6.2f}%"
        )
    print("saved", out_csv)
    print("saved", out_json)


if __name__ == "__main__":
    main()
