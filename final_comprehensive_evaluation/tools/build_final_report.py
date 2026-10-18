#!/usr/bin/env python3
"""Build a complete vanilla-vs-final report from the held-out evaluation CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(row, key):
    return float(row[key])


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def pct(x):
    return 100.0 * x


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fobj:
        w = csv.DictWriter(fobj, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def condition_rows(rows):
    out = []
    for r in rows:
        out.append({
            "model": r["model"],
            "family": r["family"],
            "corruption": r["corruption"],
            "severity": int(r["severity"]),
            "n_test": int(r["n"]),
            "vanilla_acc_pct": pct(f(r, "orig_acc")),
            "final_acc_pct": pct(f(r, "fused_acc")),
            "gain_pts": pct(f(r, "fused_acc") - f(r, "orig_acc")),
            "vanilla_nll": f(r, "orig_nll"),
            "final_nll": f(r, "fused_nll"),
            "vanilla_ece": f(r, "orig_ece"),
            "final_ece": f(r, "fused_ece"),
            "negative_transfer_pct": pct(f(r, "negative_transfer")),
            "positive_transfer_pct": pct(f(r, "positive_transfer")),
        })
    return out


def aggregate(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, recs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        base = {k: v for k, v in zip(keys, key)}
        base.update({
            "conditions": len(recs),
            "vanilla_acc_pct": mean([r["vanilla_acc_pct"] for r in recs]),
            "final_acc_pct": mean([r["final_acc_pct"] for r in recs]),
            "gain_pts": mean([r["gain_pts"] for r in recs]),
            "vanilla_nll": mean([r["vanilla_nll"] for r in recs]),
            "final_nll": mean([r["final_nll"] for r in recs]),
            "vanilla_ece": mean([r["vanilla_ece"] for r in recs]),
            "final_ece": mean([r["final_ece"] for r in recs]),
            "negative_transfer_pct": mean([r["negative_transfer_pct"] for r in recs]),
        })
        out.append(base)
    return out


def load_optional_json(path):
    return json.loads(Path(path).read_text()) if path and Path(path).is_file() else None


def load_timing(path):
    if not path or not Path(path).is_file():
        return []
    return read_csv(path)


def fmt(x, digits=2):
    return f"{float(x):.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet-csv", required=True)
    ap.add_argument("--deit-csv", required=True)
    ap.add_argument("--resnet-json", required=True)
    ap.add_argument("--deit-json", required=True)
    ap.add_argument("--resnet-tuning", required=True)
    ap.add_argument("--deit-tuning", required=True)
    ap.add_argument("--timing-csv", default=None)
    ap.add_argument("--bootstrap-resnet", default=None)
    ap.add_argument("--bootstrap-deit", default=None)
    ap.add_argument("--stage-times", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    raw = read_csv(args.resnet_csv) + read_csv(args.deit_csv)
    cond = condition_rows(raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    condition_fields = list(cond[0].keys())
    write_csv(out_dir / "all_conditions.csv", cond, condition_fields)

    by_corr = aggregate(cond, ["model", "family", "corruption"])
    by_fam_sev = aggregate(cond, ["model", "family", "severity"])
    by_sev = aggregate(cond, ["model", "severity"])
    overall = aggregate(cond, ["model"])

    agg_fields = list(by_corr[0].keys())
    write_csv(out_dir / "by_corruption.csv", by_corr, agg_fields)
    write_csv(out_dir / "by_family_severity.csv", by_fam_sev, list(by_fam_sev[0].keys()))
    write_csv(out_dir / "by_severity.csv", by_sev, list(by_sev[0].keys()))
    write_csv(out_dir / "overall_summary.csv", overall, list(overall[0].keys()))

    eval_json = {
        "resnet18": json.loads(Path(args.resnet_json).read_text()),
        "deit": json.loads(Path(args.deit_json).read_text()),
    }
    tuning = {
        "resnet18": json.loads(Path(args.resnet_tuning).read_text()),
        "deit": json.loads(Path(args.deit_tuning).read_text()),
    }
    timing = load_timing(args.timing_csv)
    boot = {
        "resnet18": load_optional_json(args.bootstrap_resnet),
        "deit": load_optional_json(args.bootstrap_deit),
    }
    stage_times = load_optional_json(args.stage_times)

    # Human-readable exhaustive condition tables.
    md_all = ["# All held-out corruption/severity results", "", "Accuracy values are percentages. Gain = final fused - vanilla.", ""]
    for model in ("resnet18", "deit"):
        md_all += [f"## {model}", ""]
        fams = sorted({r["family"] for r in cond if r["model"] == model})
        for fam in fams:
            md_all += [f"### {fam}", "", "| Corruption | Sev | Vanilla | Final | Gain |", "|---|---:|---:|---:|---:|"]
            rows = [r for r in cond if r["model"] == model and r["family"] == fam]
            rows.sort(key=lambda r: (r["corruption"], r["severity"]))
            for r in rows:
                md_all.append(
                    f"| {r['corruption']} | {r['severity']} | {r['vanilla_acc_pct']:.2f} | "
                    f"{r['final_acc_pct']:.2f} | {r['gain_pts']:+.2f} |"
                )
            md_all.append("")
    (out_dir / "ALL_CONDITIONS.md").write_text("\n".join(md_all))

    overall_map = {r["model"]: r for r in overall}
    timing_map = {(r["model"], r["method"]): r for r in timing}

    report = [
        "# Final comprehensive evaluation",
        "",
        "## Protocol",
        "",
        "- Data: 100-class ImageNet-C subset, 15 corruption types x 5 severities.",
        "- Tuning uses the established deterministic tune/test split (50/50, seed 0).",
        "- **One global configuration per architecture is tuned on the entire tune half across all 75 corruption/severity conditions.**",
        "- The held-out test half is never used to select fusion parameters.",
        "- ResNet-18 final method: fixed spectral ladder + spatial views + global grid fusion.",
        "- DeiT-Tiny final method: fixed spectral ladder + spatial views + global CE + view-bias fusion.",
        "- Vanilla baseline: the same pretrained classifier on only the original center-crop view.",
        "",
        "## Overall held-out accuracy",
        "",
        "| Model | Vanilla mCA | Final mCA | Gain | Clean vanilla | Clean final |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("resnet18", "deit"):
        o = overall_map[model]
        ej = eval_json[model]
        report.append(
            f"| {model} | {float(o['vanilla_acc_pct']):.2f}% | {float(o['final_acc_pct']):.2f}% | "
            f"{float(o['gain_pts']):+.2f} pts | {100*ej['clean_orig_acc']:.2f}% | {100*ej['clean_fused_acc']:.2f}% |"
        )

    report += ["", "## Gain by corruption family and severity", ""]
    for model in ("resnet18", "deit"):
        report += [f"### {model}", "", "| Family | Sev 1 | Sev 2 | Sev 3 | Sev 4 | Sev 5 |", "|---|---:|---:|---:|---:|---:|"]
        fams = sorted({r["family"] for r in by_fam_sev if r["model"] == model})
        for fam in fams:
            vals = []
            for sev in range(1, 6):
                row = next(r for r in by_fam_sev if r["model"] == model and r["family"] == fam and int(r["severity"]) == sev)
                vals.append(float(row["gain_pts"]))
            report.append(f"| {fam} | " + " | ".join(f"{x:+.2f}" for x in vals) + " |")
        report.append("")

    if timing:
        report += [
            "## Inference computation time",
            "",
            "Disk/JPEG loading is excluded for both methods. Both start from the same already-loaded 256x256 image batch.",
            "",
            "| Model | Vanilla ms/img | Final ms/img | Final overhead | Vanilla img/s | Final img/s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for model in ("resnet18", "deit"):
            v = timing_map[(model, "vanilla")]
            o = timing_map[(model, "final_combined")]
            report.append(
                f"| {model} | {float(v['ms_per_image']):.3f} | {float(o['ms_per_image']):.3f} | "
                f"{float(o['overhead_vs_vanilla']):.2f}x | {float(v['images_per_second']):.1f} | {float(o['images_per_second']):.1f} |"
            )
        report.append("")

    if any(boot.values()):
        report += ["## Paired bootstrap: final minus vanilla", "", "| Model | Delta | 95% CI | p(delta <= 0) |", "|---|---:|---:|---:|"]
        for model in ("resnet18", "deit"):
            b = boot[model]
            if not b:
                continue
            s = b["fused_minus_original"]
            report.append(
                f"| {model} | {100*s['point']:+.3f} pts | [{100*s['ci95'][0]:+.3f}, {100*s['ci95'][1]:+.3f}] | {s['p_le_zero']:.4f} |"
            )
        report.append("")

    report += ["## Final global tuning parameters", ""]
    for model in ("resnet18", "deit"):
        cfg = tuning[model]["global"]["config"]
        report += [f"### {model}", "", "```json", json.dumps(cfg, indent=2), "```", ""]

    if stage_times:
        report += ["## Pipeline wall-clock stages", "", "```json", json.dumps(stage_times, indent=2), "```", ""]

    report += [
        "## Files",
        "",
        "- `all_conditions.csv`: every model x corruption x severity test result (150 rows).",
        "- `ALL_CONDITIONS.md`: same exhaustive comparison in readable tables.",
        "- `by_corruption.csv`: averages over severities for each corruption.",
        "- `by_family_severity.csv`: family-level behavior at every severity.",
        "- `by_severity.csv`: average over all 15 corruptions at each severity.",
        "- `overall_summary.csv`: overall held-out mCA comparison.",
        "- Timing and bootstrap files remain in their sibling result directories.",
    ]
    (out_dir / "FINAL_REPORT.md").write_text("\n".join(report))

    manifest = {
        "resnet_eval_csv": args.resnet_csv,
        "deit_eval_csv": args.deit_csv,
        "resnet_tuning": args.resnet_tuning,
        "deit_tuning": args.deit_tuning,
        "timing_csv": args.timing_csv,
        "bootstrap_resnet": args.bootstrap_resnet,
        "bootstrap_deit": args.bootstrap_deit,
        "stage_times": args.stage_times,
        "rows": len(cond),
    }
    (out_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(out_dir / "FINAL_REPORT.md")


if __name__ == "__main__":
    main()
