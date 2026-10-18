"""Paired-bootstrap spectral candidates before freezing the final method.

Console output is concise by default: one line per completed comparison plus the
final output paths. Use --verbose to restore child-command and per-condition logs.
"""
from __future__ import annotations

import argparse
import csv
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


def key(rec):
    return rec["model"], rec["ladder"], rec["method"]


def get_record(records, model, ladder, method):
    matches = [r for r in records if key(r) == (model, ladder, method)]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one record for {(model, ladder, method)}, got {len(matches)}"
        )
    return matches[0]


def comparison_set(records, model):
    # Accept both the cleaned method names and old development-result names.
    def one_of(ladder, methods):
        for method in methods:
            matches = [r for r in records if key(r) == (model, ladder, method)]
            if len(matches) == 1:
                return matches[0]
        raise RuntimeError(f"missing record for {model}/{ladder}/{methods}")

    final = one_of("fixed", ["ce_bias", "ce_refined_bias"])
    fixed_grid = one_of("fixed", ["grid"])
    geometric_bias = one_of("geometric", ["ce_bias", "ce_refined_bias"])

    excluded = {id(final), id(fixed_grid), id(geometric_bias)}
    remaining = [r for r in records if r["model"] == model and id(r) not in excluded]
    strongest = max(remaining, key=lambda r: float(r["mCA_fused"]))

    return final, [
        ("vs_fixed_grid", fixed_grid),
        ("vs_geometric_bias", geometric_bias),
        ("vs_strongest_other", strongest),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True,
                    help="ablation_summary_*.json produced by run_ablation.py")
    ap.add_argument("--out", default="results_selection")
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"],
                    choices=["resnet18", "deit"])
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--signal-chunk", type=int, default=256)
    ap.add_argument("--signal-pair-batch", type=int, default=32)
    ap.add_argument("--verbose", action="store_true",
                    help="show child commands and their full per-condition output")
    args = ap.parse_args()

    records = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("summary JSON must contain the ablation record list")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    rows = []
    total = len(args.models) * 3
    done = 0

    for model in args.models:
        final, comparisons = comparison_set(records, model)
        for i, (label, other) in enumerate(comparisons):
            done += 1
            outfile = out_dir / f"bootstrap_{model}_fixed_bias_{label}.json"
            cmd = [
                py, "scripts/bootstrap_pairwise.py",
                "--model", model,
                "--a-mode", "fused",
                "--a-cache", final["cache_dir"],
                "--a-tuning", final["tuning"],
                "--b-mode", "fused",
                "--b-cache", other["cache_dir"],
                "--b-tuning", other["tuning"],
                "--n-boot", str(args.n_boot),
                "--seed", str(args.seed + i),
                "--device", args.device,
                "--signal-chunk", str(args.signal_chunk),
                "--signal-pair-batch", str(args.signal_pair_batch),
                "--out", str(outfile),
            ]
            if not args.verbose:
                cmd.append("--quiet-progress")
            run(
                cmd,
                verbose=args.verbose,
                label=None if args.verbose else f"[{done}/{total}] {model:8s} {label:22s}",
            )
            boot = json.loads(outfile.read_text(encoding="utf-8"))
            row = {
                "model": model,
                "candidate_ladder": final["ladder"],
                "candidate_method": final["method"],
                "candidate_mCA": final["mCA_fused"],
                "comparison": label,
                "other_ladder": other["ladder"],
                "other_method": other["method"],
                "other_mCA": other["mCA_fused"],
                "delta_pts": 100.0 * boot["point"],
                "ci95_lo_pts": 100.0 * boot["ci95"][0],
                "ci95_hi_pts": 100.0 * boot["ci95"][1],
                "p_le_zero": boot["p_le_zero"],
                "bootstrap_json": str(outfile),
            }
            rows.append(row)
            if not args.verbose:
                print(
                    f"      delta {row['delta_pts']:+.3f} pts  "
                    f"95% CI [{row['ci95_lo_pts']:+.3f}, {row['ci95_hi_pts']:+.3f}]"
                )

    json_out = out_dir / "spectral_selection_bootstrap.json"
    csv_out = out_dir / "spectral_selection_bootstrap.csv"
    json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if args.verbose:
        print("\n=== spectral candidate bootstrap summary ===")
        print(f"{'model':10s} {'comparison':24s} {'delta':>9s} {'95% CI':>24s} {'P<=0':>8s}")
        for r in rows:
            ci = f"[{r['ci95_lo_pts']:+.3f}, {r['ci95_hi_pts']:+.3f}]"
            print(
                f"{r['model']:10s} {r['comparison']:24s} "
                f"{r['delta_pts']:+8.3f} {ci:>24s} {r['p_le_zero']:8.4f}"
            )
    print(f"saved {json_out}")
    print(f"saved {csv_out}")


if __name__ == "__main__":
    main()
