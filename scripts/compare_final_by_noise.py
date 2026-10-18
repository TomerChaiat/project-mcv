from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

MODELS = {
    "resnet18": {
        "final_pattern": "eval_resnet18_combined_grid_*.csv",
        "label": "ResNet-18 (final: grid)",
    },
    "deit": {
        "final_pattern": "eval_deit_combined_ce_bias_*.csv",
        "label": "DeiT-Tiny (final: CE + view bias)",
    },
}


def newest(pattern: str) -> Path:
    files = [Path(p) for p in glob.glob(pattern)]
    if not files:
        raise FileNotFoundError(f"No file matched: {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


def newest_from(patterns: list[str]) -> Path:
    files = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No file matched any of:\n  " + "\n  ".join(patterns))
    return max(files, key=lambda p: p.stat().st_mtime)


def load_rows(path: Path):
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["model"], r["corruption"], int(r["severity"]))
            out[key] = r
    return out


def pct(x: float) -> float:
    return 100.0 * float(x)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(
        description="Compare final branch tuning against previous combined fusion and spatial TTA per corruption/severity."
    )
    ap.add_argument("--final-dir", default="results/final_branch_combined")
    ap.add_argument("--previous-dir", default="results/previous_team_recheck_full")
    ap.add_argument("--family", choices=["noise", "blur", "weather", "digital", "all"], default="noise")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    final_dir = Path(args.final_dir)
    previous_dir = Path(args.previous_dir)
    out_path = Path(args.out) if args.out else final_dir / "final_vs_previous_by_condition.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged = []

    for model, meta in MODELS.items():
        final_csv = newest(str(final_dir / meta["final_pattern"]))
        previous_csv = newest(str(previous_dir / f"eval_{model}_previous_team_combined_*.csv"))
        tta_csv = newest_from([
            str(final_dir / f"tta_{model}_*.csv"),
            str(previous_dir / f"tta_{model}_*.csv"),
        ])

        final_rows = load_rows(final_csv)
        previous_rows = load_rows(previous_csv)
        tta_rows = load_rows(tta_csv)

        print(f"\n{meta['label']}")
        print(f"  final    : {final_csv}")
        print(f"  previous : {previous_csv}")
        print(f"  spatial  : {tta_csv}")

        keys = sorted(final_rows, key=lambda k: (k[1], k[2]))
        for key in keys:
            if key not in previous_rows or key not in tta_rows:
                continue
            f = final_rows[key]
            p = previous_rows[key]
            t = tta_rows[key]

            # The comparison is meaningful only when all methods use the same held-out examples.
            n_final = int(f["n"])
            n_prev = int(p["n"])
            n_tta = int(t["n"])
            if not (n_final == n_prev == n_tta):
                raise RuntimeError(f"Sample-count mismatch for {key}: final={n_final}, previous={n_prev}, tta={n_tta}")

            orig = float(f["orig_acc"])
            orig_prev = float(p["orig_acc"])
            orig_tta = float(t["orig_acc"])
            if max(abs(orig-orig_prev), abs(orig-orig_tta)) > 1e-6:
                raise RuntimeError(f"Original-accuracy mismatch for {key}; check split/cache alignment")

            spatial = float(t["tta_acc"])
            previous = float(p["fused_acc"])
            final = float(f["fused_acc"])

            merged.append({
                "model": model,
                "family": f["family"],
                "corruption": f["corruption"],
                "severity": int(f["severity"]),
                "n": n_final,
                "orig_acc": orig,
                "spatial_tta_acc": spatial,
                "previous_combined_acc": previous,
                "final_combined_acc": final,
                "final_minus_orig_pts": pct(final - orig),
                "final_minus_spatial_pts": pct(final - spatial),
                "final_minus_previous_pts": pct(final - previous),
                "previous_minus_spatial_pts": pct(previous - spatial),
            })

    fields = list(merged[0])
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged)

    selected = [r for r in merged if args.family == "all" or r["family"] == args.family]

    for model, meta in MODELS.items():
        rows = [r for r in selected if r["model"] == model]
        if not rows:
            continue
        print(f"\n=== {meta['label']} | family={args.family} ===")
        print(f"{'corruption':18s} {'sev':>3s} {'orig':>7s} {'spatial':>7s} {'previous':>8s} {'final':>7s} {'F-S':>7s} {'F-P':>7s}")
        print("-" * 76)
        for r in sorted(rows, key=lambda x: (x["corruption"], x["severity"])):
            print(
                f"{r['corruption']:18s} {r['severity']:3d} "
                f"{pct(r['orig_acc']):6.2f}% {pct(r['spatial_tta_acc']):6.2f}% "
                f"{pct(r['previous_combined_acc']):7.2f}% {pct(r['final_combined_acc']):6.2f}% "
                f"{r['final_minus_spatial_pts']:+6.2f} {r['final_minus_previous_pts']:+6.2f}"
            )

        print("\nMean by corruption (across severities 1-5):")
        by_corr = defaultdict(list)
        for r in rows:
            by_corr[r["corruption"]].append(r)
        print(f"{'corruption':18s} {'orig':>7s} {'spatial':>7s} {'previous':>8s} {'final':>7s} {'F-S':>7s} {'F-P':>7s}")
        for corr, cr in sorted(by_corr.items()):
            o = mean(x["orig_acc"] for x in cr)
            s = mean(x["spatial_tta_acc"] for x in cr)
            p = mean(x["previous_combined_acc"] for x in cr)
            f = mean(x["final_combined_acc"] for x in cr)
            print(f"{corr:18s} {pct(o):6.2f}% {pct(s):6.2f}% {pct(p):7.2f}% {pct(f):6.2f}% {pct(f-s):+6.2f} {pct(f-p):+6.2f}")

        print("\nMean by severity (across corruption types in this family):")
        by_sev = defaultdict(list)
        for r in rows:
            by_sev[r["severity"]].append(r)
        print(f"{'severity':>8s} {'orig':>7s} {'spatial':>7s} {'previous':>8s} {'final':>7s} {'F-S':>7s} {'F-P':>7s}")
        for sev, sr in sorted(by_sev.items()):
            o = mean(x["orig_acc"] for x in sr)
            s = mean(x["spatial_tta_acc"] for x in sr)
            p = mean(x["previous_combined_acc"] for x in sr)
            f = mean(x["final_combined_acc"] for x in sr)
            print(f"{sev:8d} {pct(o):6.2f}% {pct(s):6.2f}% {pct(p):7.2f}% {pct(f):6.2f}% {pct(f-s):+6.2f} {pct(f-p):+6.2f}")

    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
