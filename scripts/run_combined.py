"""Run the selected spectral+spatial experiment and paired bootstrap tests.

The default methods reflect the spectral-selection bootstrap in this branch:
ResNet-18 uses the fixed-ladder grid rule and DeiT-Tiny uses CE + view bias.
Use --resnet-method/--deit-method to override them.

The script reuses existing tuning JSONs when possible. Console output is concise
by default; pass --verbose for child-command output.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

METHOD_TO_VARIANT = {
    "ce": "base",
    "ce_bias": "bias",
    "ce_unconstrained": "unconstrained",
}


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
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
    )
    if proc.returncode != 0:
        if label:
            print(" FAILED")
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    if label:
        print(f" done ({time.time() - t0:.1f}s)")


def newest(patterns, *, severities=None, methods=None):
    if isinstance(patterns, (str, Path)):
        patterns = [patterns]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(str(pattern)))
    for path in reversed(sorted(set(files))):
        try:
            rec = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if severities is not None and list(rec.get("tune_severities", [])) != list(severities):
            continue
        if methods is not None and rec.get("method") not in set(methods):
            continue
        return path
    return None


def method_for_model(args, model):
    return args.resnet_method if model == "resnet18" else args.deit_method


def ensure_grid(py, model, cache_dir, results, args, severities):
    tag = Path(cache_dir).name
    patterns = [
        results / f"tuning_grid_{model}_{tag}_*.json",
        # Compatibility with development runs produced before repository cleanup.
        results / f"tuning_fast_{model}_{tag}_*.json",
    ]
    path = None if args.force_tuning else newest(patterns, severities=severities)
    if path:
        if not args.verbose:
            print(f"[{model}] grid tuning ................ reuse")
        return path

    run([
        py, "scripts/tune_fusion.py",
        "--models", model,
        "--cache-dir", cache_dir,
        "--tag", tag,
        "--out", str(results),
        "--tune-severities", *map(str, severities),
        "--ladder-len", "8",
        "--device", args.device,
        "--config-batch", str(args.config_batch),
        "--signal-chunk", str(args.signal_chunk),
        "--signal-pair-batch", str(args.signal_pair_batch),
    ], verbose=args.verbose, label=None if args.verbose else f"[{model}] grid tuning ................")
    path = newest(patterns, severities=severities)
    if not path:
        raise RuntimeError(f"grid tuning output not found for {model}")
    return path


def ensure_method(py, model, method, cache_dir, grid_path, results, args, severities):
    if method == "grid":
        return grid_path

    tag = Path(cache_dir).name
    legacy = {
        "ce": "ce_refined",
        "ce_bias": "ce_refined_bias",
        "ce_unconstrained": "ce_refined_unconstrained",
    }[method]
    patterns = [
        results / f"tuning_{method}_{model}_{tag}_*.json",
        results / f"tuning_{legacy}_{model}_{tag}_*.json",
    ]
    path = None if args.force_tuning else newest(
        patterns, severities=severities, methods=[method, legacy]
    )
    if path:
        if not args.verbose:
            print(f"[{model}] {method:27s} reuse")
        return path

    run([
        py, "scripts/tune_fusion_ce.py",
        "--models", model,
        "--cache-dir", cache_dir,
        "--tag", tag,
        "--out", str(results),
        "--tune-severities", *map(str, severities),
        "--ladder-len", "8",
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
        "--variants", METHOD_TO_VARIANT[method],
    ], verbose=args.verbose, label=None if args.verbose else f"[{model}] {method:27s}")

    path = newest(patterns, severities=severities, methods=[method, legacy])
    if not path:
        raise RuntimeError(f"{method} tuning output not found for {model}")
    return path


def zero_comparison(n_conditions=75):
    return {"point": 0.0, "ci95": [0.0, 0.0], "p_le_zero": 1.0, "n_conditions": n_conditions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectral-cache", default="cache_fixed_fresh")
    ap.add_argument("--spatial-cache", default="cache_spatial")
    ap.add_argument("--combined-cache", default="cache_combined")
    ap.add_argument("--results", default="results_final")
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"], choices=["resnet18", "deit"])
    ap.add_argument("--resnet-method", default="grid", choices=["grid", "ce", "ce_bias", "ce_unconstrained"])
    ap.add_argument("--deit-method", default="ce_bias", choices=["grid", "ce", "ce_bias", "ce_unconstrained"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config-batch", type=int, default=64)
    ap.add_argument("--signal-chunk", type=int, default=256)
    ap.add_argument("--signal-pair-batch", type=int, default=32)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--bias-l2", type=float, default=1e-3)
    ap.add_argument("--optimizer-backend", choices=["auto", "torch", "numpy"], default="torch")
    ap.add_argument("--ce-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--scale-batch", type=int, default=32)
    ap.add_argument("--temperature-batch", type=int, default=49)
    ap.add_argument("--resident-probs", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--force-tuning", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    severities = [1, 2, 3, 4, 5]

    print("=== combined spectral + spatial experiment ===")
    run([
        py, "scripts/build_combined_cache.py",
        "--models", *args.models,
        "--spectral-dir", args.spectral_cache,
        "--spatial-dir", args.spatial_cache,
        "--out-dir", args.combined_cache,
    ], verbose=args.verbose, label=None if args.verbose else "[combined] build/verify cache ..........")

    summary = []
    final_tuning = {}
    grid_tuning = {}

    for model in args.models:
        grid = ensure_grid(py, model, args.combined_cache, results, args, severities)
        method = method_for_model(args, model)
        tuning = ensure_method(py, model, method, args.combined_cache, grid, results, args, severities)
        grid_tuning[model] = grid
        final_tuning[model] = tuning

        run([
            py, "scripts/evaluate.py",
            "--models", model,
            "--cache-dir", args.combined_cache,
            "--tuning", tuning,
            "--tag", f"combined_{method}",
            "--out", str(results),
            "--ladder-len", "8",
            "--device", args.device,
            "--signal-chunk", str(args.signal_chunk),
            "--signal-pair-batch", str(args.signal_pair_batch),
            "--quiet-progress",
        ], verbose=args.verbose, label=None if args.verbose else f"[{model}] evaluation ....................")

        final_boot = results / f"bootstrap_{model}_{method}_final.json"
        run([
            py, "scripts/bootstrap_final.py",
            "--model", model,
            "--combined-cache", args.combined_cache,
            "--spatial-cache", args.spatial_cache,
            "--tuning", tuning,
            "--ladder-len", "8",
            "--n-boot", str(args.n_boot),
            "--device", args.device,
            "--signal-chunk", str(args.signal_chunk),
            "--signal-pair-batch", str(args.signal_pair_batch),
            "--quiet-progress",
            "--out", str(final_boot),
        ], verbose=args.verbose, label=None if args.verbose else f"[{model}] bootstrap vs orig/spatial .....")

        final_boot_rec = json.loads(final_boot.read_text(encoding="utf-8"))
        if method == "grid":
            vs_grid = zero_comparison()
        else:
            boot_vs_grid = results / f"bootstrap_{model}_{method}_vs_grid.json"
            run([
                py, "scripts/bootstrap_pairwise.py",
                "--model", model,
                "--a-mode", "fused",
                "--a-cache", args.combined_cache,
                "--a-tuning", tuning,
                "--a-ladder-len", "8",
                "--b-mode", "fused",
                "--b-cache", args.combined_cache,
                "--b-tuning", grid,
                "--b-ladder-len", "8",
                "--n-boot", str(args.n_boot),
                "--device", args.device,
                "--signal-chunk", str(args.signal_chunk),
                "--signal-pair-batch", str(args.signal_pair_batch),
                "--quiet-progress",
                "--out", str(boot_vs_grid),
            ], verbose=args.verbose, label=None if args.verbose else f"[{model}] bootstrap vs grid .............")
            rec = json.loads(boot_vs_grid.read_text(encoding="utf-8"))
            vs_grid = {
                "point": rec["point"], "ci95": rec["ci95"],
                "p_le_zero": rec["p_le_zero"], "n_conditions": rec["n_conditions"],
            }

        summary.append({
            "model": model,
            "method": method,
            "spectral_cache": args.spectral_cache,
            "combined_cache": args.combined_cache,
            "tuning": tuning,
            "grid_tuning": grid,
            "fused_minus_original": final_boot_rec.get("fused_minus_original"),
            "fused_minus_spatial": final_boot_rec.get("fused_minus_spatial"),
            "fused_minus_grid": vs_grid,
        })

    manifest = results / "combined_manifest.json"
    manifest.write_text(json.dumps({
        "spectral_cache": args.spectral_cache,
        "spatial_cache": args.spatial_cache,
        "combined_cache": args.combined_cache,
        "methods": {m: method_for_model(args, m) for m in args.models},
        "tuning": final_tuning,
        "grid_tuning": grid_tuning,
    }, indent=2), encoding="utf-8")

    out = results / "combined_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== paired comparisons ===")
    for rec in summary:
        print(f"{rec['model']} ({rec['method']})")
        for label, key in [
            ("combined - original", "fused_minus_original"),
            ("combined - spatial", "fused_minus_spatial"),
            ("combined - grid", "fused_minus_grid"),
        ]:
            x = rec.get(key)
            if not x:
                continue
            print(
                f"  {label:21s}: {100*x['point']:+.3f} pts  "
                f"95% CI [{100*x['ci95'][0]:+.3f}, {100*x['ci95'][1]:+.3f}]  "
                f"P<=0={x['p_le_zero']:.4f}"
            )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
