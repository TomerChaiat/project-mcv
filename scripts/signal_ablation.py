"""What each per-image reliability signal contributes, on the 8-view spectral cache.

Part A - how informative is each signal on its own (no fitting). Test half.
For every non-original view k and every signal s in
    C   confidence (top-class probability),
    A   agreement with all other views (agreement="all"),
    Ao  agreement with the original view only (agreement="origin"),
    S   stability over neighbouring ladder rungs (m=1),
computed raw (not z-scored) with src.analysis.profile_signals:
    1. AUROC of s_k for "view k is correct";
    2. AUROC of s_k - s_0 for "view k is correct and the original is wrong".
The AUROC is computed within each corruption x severity condition, averaged
over the conditions of a family (or all 75), then averaged over the 7
non-original views. 95% CIs from a cluster bootstrap over images: one draw of
images per replicate, applied to every condition, view and signal.

Part B - fusion with signal subsets. The CE + bias fits refit by
tune_fusion_ce.py --variants bias --signals ... (leave-one-family-out, tuning
half; see the 'signals' phase of run_full_pipeline.sh) are scored on the test
half, each corruption with the fold that held its family out. Mean accuracy
over the 75 conditions, and paired cluster-bootstrap differences against
{C,A,S} and against bias only.

    python scripts/signal_ablation.py                    # both parts, all models
    python scripts/signal_ablation.py --parts A
    python scripts/signal_ablation.py --plot-only        # tables + figure from the JSONs

Writes <out>/signal_auroc.json, <out>/signal_subsets.json,
<figures>/table_signal_auroc.tex, <figures>/table_signal_subsets.tex and
<figures>/signal_auroc.pdf/.png.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import cache
from src.analysis import profile_signals
from src.bootstrap import cluster_bootstrap, image_keys
from src.evaluation import device_from_arg, fused_correct_many_torch
from src.metrics import FusionConfig
from src.splits import CORRUPTION_FAMILY, FAMILIES, tune_test_masks

MODELS = ["resnet18", "deit", "convnext"]
MODEL_LABEL = {"resnet18": "ResNet-18", "deit": "DeiT-Tiny", "convnext": "ConvNeXt-Tiny"}
SEVERITIES = [1, 2, 3, 4, 5]
SIGNALS = ["C", "A", "Ao", "S"]
SIGNAL_LABEL = {"C": "confidence $C$", "A": "agreement $A$ (all)",
                "Ao": "agreement $A$ (origin)", "S": "stability $S$"}
TARGETS = {"correct": "view k correct",
           "beats_original": "view k correct and original wrong (score s_k - s_0)"}
GROUPS = ["all"] + list(FAMILIES)
#: signal subsets of Part B, in table order; labels as written by tune_fusion_ce.py
SUBSETS = ["none", "C", "A", "S", "CA", "CS", "AS", "CAS", "CAoS"]
SUBSET_LABEL = {"none": "bias only", "C": "C", "A": "A", "S": "S", "CA": "C, A", "CS": "C, S",
                "AS": "A, S", "CAS": "C, A, S", "CAoS": "C, A (origin), S"}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_test_half(model, split, with_probs):
    """Per condition on the test half: raw signals, per-view correctness, image
    keys, and (for Part B) the probabilities."""
    out = {}
    t0 = time.time()
    for corruption in CORRUPTION_FAMILY:
        for sev in SEVERITIES:
            d = cache.load_condition(model, corruption, sev)
            _, test = tune_test_masks(d["paths"], split["frac_tune"], split["seed"])
            probs = np.ascontiguousarray(d["probs"][test])
            labels = np.asarray(d["labels"][test], dtype=np.int64)
            p32 = probs.astype(np.float32)
            C, A, S = profile_signals(p32, m=1, standardize=False, agreement="all")
            _, Ao, _ = profile_signals(p32, m=1, standardize=False, agreement="origin")
            rec = {"signals": {"C": C, "A": A, "Ao": Ao, "S": S},
                   "correct": p32.argmax(-1) == labels[:, None],
                   "labels": labels, "keys": image_keys(d["paths"][test])}
            if with_probs:
                rec["probs"] = probs
            out[(corruption, sev)] = rec
            del p32
    print(f"  loaded {len(out)} test-half conditions in {time.time() - t0:.0f}s")
    return out


# --------------------------------------------------------------------------- #
# Part A: within-condition AUROC with image weights
# --------------------------------------------------------------------------- #
class ConditionAUROC:
    """Weighted AUROC per condition for one (score, label) set, reusable across
    bootstrap replicates: the sort by (condition, score) and the tie groups are
    built once, each replicate only changes the per-image weights.

    With integer weights w (how often each image was drawn) this is exactly the
    Mann-Whitney AUROC of the resampled data, ties counted as 1/2.
    """

    def __init__(self, score, label, cond, img, n_cond, device):
        """Sort by (condition, score) once and precompute tie groups so AUROC can be
        recomputed for many bootstrap weightings on `device`.
        """
        order = np.lexsort((score, cond))
        s, c = score[order], cond[order]
        new = np.r_[True, (s[1:] != s[:-1]) | (c[1:] != c[:-1])]
        group = np.cumsum(new) - 1
        n_group = int(group[-1]) + 1
        group_cond = c[new]
        first_group = np.searchsorted(group_cond, np.arange(n_cond))
        t = lambda a, dt=torch.long: torch.as_tensor(a, device=device, dtype=dt)
        self.y = t(label[order], torch.float32)
        self.img = t(img[order])
        self.cond = t(c)
        self.group = t(group)
        self.group_cond = t(group_cond)
        self.first_group = t(np.minimum(first_group, n_group - 1))
        self.n_group, self.n_cond = n_group, n_cond

    def __call__(self, W):
        """W: (B, n_images) weights -> (B, n_cond) AUROC (NaN without both classes)."""
        B = W.shape[0]
        w = W[:, self.img]
        pos, neg = w * self.y, w * (1.0 - self.y)
        neg_g = torch.zeros(B, self.n_group, device=w.device).index_add_(1, self.group, neg)
        before = torch.cumsum(neg_g, 1) - neg_g                         # exclusive, global
        before = before - before[:, self.first_group][:, self.group_cond]  # within condition
        hits = pos * (before + 0.5 * neg_g)[:, self.group]
        z = lambda: torch.zeros(B, self.n_cond, device=w.device, dtype=torch.float64)
        num = z().index_add_(1, self.cond, hits.double())
        P = z().index_add_(1, self.cond, pos.double())
        N = z().index_add_(1, self.cond, neg.double())
        return torch.where((P > 0) & (N > 0), num / (P * N).clamp_min(1e-300),
                           torch.full_like(num, float("nan")))


def group_means(auc, cond_family):
    """(B, n_cond) -> {group: (B,)} nan-mean over the group's conditions."""
    out = {}
    for g in GROUPS:
        sel = [i for i, f in enumerate(cond_family) if g == "all" or f == g]
        out[g] = torch.nanmean(auc[:, sel], dim=1)
    return out


def part_a(model, data, args, device):
    """Part A: per-signal AUROC for predicting view correctness (and beating
    the original), by family, with image-cluster bootstrap CIs.

    Returns:
        Dict with n_images, n_conditions, views and per-target signal AUROCs.
    """
    conds = list(data)
    cond_family = [CORRUPTION_FAMILY[c] for c, _ in conds]
    keys = sorted({str(k) for d in data.values() for k in d["keys"]})
    index = {k: i for i, k in enumerate(keys)}
    n_img = len(keys)
    img = np.concatenate([[index[str(k)] for k in d["keys"]] for d in data.values()])
    cond = np.concatenate([np.full(len(d["keys"]), i) for i, d in enumerate(data.values())])
    n_views = next(iter(data.values()))["correct"].shape[1]
    views = list(range(1, n_views))

    def cat(fn):
        """Concatenate `fn(d)` over all conditions in `data`. Returns one array."""
        return np.concatenate([fn(d) for d in data.values()])

    calcs, base_rate = {}, {}
    for target in TARGETS:
        for k in views:
            if target == "correct":
                y = cat(lambda d: d["correct"][:, k])
            else:
                y = cat(lambda d: d["correct"][:, k] & ~d["correct"][:, 0])
            base_rate[(target, k)] = float(y.mean())
            for s in SIGNALS:
                if target == "correct":
                    score = cat(lambda d: d["signals"][s][:, k])
                else:
                    score = cat(lambda d: d["signals"][s][:, k] - d["signals"][s][:, 0])
                calcs[(target, s, k)] = ConditionAUROC(score.astype(np.float64), y, cond, img,
                                                       len(conds), device)

    def stats(W):
        """{(target, signal): {group: (B,) view-averaged AUROC}} and per-view 'all'."""
        res, per_view = {}, {}
        for target in TARGETS:
            for s in SIGNALS:
                acc = {g: 0.0 for g in GROUPS}
                for k in views:
                    gm = group_means(calcs[(target, s, k)](W), cond_family)
                    per_view[(target, s, k)] = gm["all"]
                    for g in GROUPS:
                        acc[g] = acc[g] + gm[g] / len(views)
                res[(target, s)] = acc
        return res, per_view

    point, point_view = stats(torch.ones(1, n_img, device=device))
    rng = np.random.default_rng(args.seed)
    reps = {key: {g: [] for g in GROUPS} for key in point}
    t0 = time.time()
    for lo in range(0, args.n_boot, args.boot_chunk):
        nb = min(args.boot_chunk, args.n_boot - lo)
        W = rng.multinomial(n_img, np.full(n_img, 1.0 / n_img), size=nb)
        r, _ = stats(torch.as_tensor(W, device=device, dtype=torch.float32))
        for key, gs in r.items():
            for g, v in gs.items():
                reps[key][g].append(v.cpu().numpy())
    print(f"  AUROC bootstrap ({args.n_boot} replicates) in {time.time() - t0:.0f}s")

    out = {"n_images": n_img, "n_conditions": len(conds), "views": views, "targets": {}}
    for target in TARGETS:
        t = out["targets"][target] = {"definition": TARGETS[target],
                                      "base_rate_per_view": {str(k): base_rate[(target, k)]
                                                             for k in views},
                                      "signals": {}}
        for s in SIGNALS:
            row = {}
            for g in GROUPS:
                r = np.concatenate(reps[(target, s)][g])
                lo_q, hi_q = np.nanpercentile(r, [2.5, 97.5])
                row[g] = {"auroc": float(point[(target, s)][g][0]),
                          "ci95": [float(lo_q), float(hi_q)]}
            row["per_view_all"] = {str(k): float(point_view[(target, s, k)][0]) for k in views}
            t["signals"][s] = row
        print(f"  {target:15s} " + "  ".join(
            f"{s}={t['signals'][s]['all']['auroc']:.3f}" for s in SIGNALS))
    return out


# --------------------------------------------------------------------------- #
# Part B: test-half evaluation of the subset fits
# --------------------------------------------------------------------------- #
def latest_fit(fits, method, model, tag):
    """Newest tuning json for (method, model, tag) under `fits`, or None."""
    rx = re.compile(rf"tuning_{re.escape(method)}_{re.escape(model)}_{re.escape(tag)}_\d{{8}}_\d{{6}}\.json$")
    files = [f for f in glob.glob(str(Path(fits) / f"tuning_{method}_{model}_{tag}_*.json"))
             if rx.search(f)]
    return sorted(files)[-1] if files else None


def part_b(model, data, args, device):
    """Part B: test-half accuracy of CE + bias fits on each signal subset,
    checked against the pipeline's own fits, with bootstrapped comparisons.

    Returns:
        Dict with orig_mCA, reproduction checks and per-subset results.
    """
    recs = {}
    for sub in SUBSETS:
        path = latest_fit(args.fits, f"ce_bias_sig{sub}", model, args.tag)
        if path is None:
            raise SystemExit(f"{model}: no fit for subset {sub} in {args.fits}")
        recs[sub] = (path, json.loads(Path(path).read_text()))
    # reference: the pipeline's own CE + bias and bias-only fits, to check this
    # evaluation reproduces the pipeline numbers
    refs = {}
    for meth in ("ce_bias", "ce_bias_only"):
        p = latest_fit(args.results, meth, model, args.tag)
        e = sorted(glob.glob(str(Path(args.results) / f"eval_{model}_{args.tag}_{meth}_2*.json")))
        if p and e:
            refs[meth] = (p, json.loads(Path(p).read_text()), json.loads(Path(e[-1]).read_text()))
    cfgs = {name: {fam: FusionConfig.from_dict(v["config"]) for fam, v in rec["held_out"].items()}
            for name, (_, rec) in recs.items()}
    cfgs.update({f"ref_{m}": {fam: FusionConfig.from_dict(v["config"])
                              for fam, v in r[1]["held_out"].items()} for m, r in refs.items()})

    correct = {name: {} for name in cfgs}
    orig = {}
    for cs, d in data.items():
        fam = CORRUPTION_FAMILY[cs[0]]
        ok, o = fused_correct_many_torch(d["probs"], d["labels"],
                                         {n: c[fam] for n, c in cfgs.items()}, device=device)
        for n in cfgs:
            correct[n][cs] = ok[n]
        orig[cs] = o

    def mca(c):
        """Mean accuracy over conditions of a {condition: correctness} dict."""
        return float(np.mean([v.mean() for v in c.values()]))

    def boot(a, b):
        """Paired cluster bootstrap of a - b over the test images.

        Returns:
            {"point", "ci95", "p_le_zero"} with point/CI in percentage points.
        """
        r = cluster_bootstrap([{"keys": data[cs]["keys"], "a": a[cs], "b": b[cs]} for cs in data],
                              n_boot=args.n_boot, seed=args.seed)
        return {"point": 100 * r["point"], "ci95": [100 * x for x in r["ci95"]],
                "p_le_zero": r["p_le_zero"]}

    checks = {}
    for m, (p, _, ev) in refs.items():
        mine, theirs = mca(correct[f"ref_{m}"]), ev["mCA_fused"]
        checks[m] = {"tuning": p, "pipeline_mCA": 100 * theirs, "this_mCA": 100 * mine}
        if abs(mine - theirs) > 1e-6:
            raise SystemExit(f"{model}: {m} re-evaluated at {mine} != pipeline {theirs}")
    out = {"orig_mCA": 100 * mca(orig), "checks": checks, "subsets": {}}
    for sub in SUBSETS:
        path, rec = recs[sub]
        c = correct[sub]
        g = rec["global"]["config"]
        out["subsets"][sub] = {
            "fit": path,
            "signals": rec["signals"],
            "agreement_modes": rec["agreement_modes"],
            "mCA": 100 * mca(c),
            "vs_CAS": None if sub == "CAS" else boot(c, correct["CAS"]),
            "vs_bias_only": None if sub == "none" else boot(c, correct["none"]),
            "vs_original": boot(c, orig),
            "global_coefficients": {"alpha": g["alpha"], "beta": g["beta"], "gamma": g["gamma"],
                                    "m": g["m"], "agreement": g["agreement"]},
            "fold_coefficients": {fam: {k: v["config"][k] for k in ("alpha", "beta", "gamma")}
                                  for fam, v in rec["held_out"].items()},
            "tune_half_acc_global": 100 * rec["global"]["mean_fused_acc"],
        }
        r = out["subsets"][sub]
        print(f"  {SUBSET_LABEL[sub]:18s} {r['mCA']:.2f}" + (
            "" if r["vs_CAS"] is None else
            f"  vs CAS {r['vs_CAS']['point']:+.2f} [{r['vs_CAS']['ci95'][0]:+.2f}, {r['vs_CAS']['ci95'][1]:+.2f}]"))
    print("  checks:", {m: round(v["this_mCA"] - v["pipeline_mCA"], 6) for m, v in checks.items()})
    return out


# --------------------------------------------------------------------------- #
# tables and figure
# --------------------------------------------------------------------------- #
def _ci(x, scale=1.0, sign=False):
    """Format a point estimate with its 95% CI as 'p [lo, hi]'. Returns a string."""
    f = "{:+.2f}" if sign else "{:.2f}"
    p, (lo, hi) = x["point"] if "point" in x else x["auroc"], x["ci95"]
    return f"{f.format(scale * p)} [{f.format(scale * lo)}, {f.format(scale * hi)}]"


def table_auroc(res, path):
    """Write the LaTeX AUROC table (per model x signal, by family). Returns nothing."""
    fam = list(FAMILIES)
    lines = [r"\begin{tabular}{ll" + "r" * 10 + "}", r"\toprule",
             r" & & \multicolumn{5}{c}{view $k$ correct} & \multicolumn{5}{c}{view $k$ correct, original wrong ($s_k - s_0$)} \\",
             r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}",
             "Model & Signal & all & " + " & ".join(fam) + " & all & " + " & ".join(fam) + r" \\",
             r"\midrule"]
    models = [m for m in MODELS if m in res["models"]]
    for i, m in enumerate(models):
        t = res["models"][m]["targets"]
        for j, s in enumerate(SIGNALS):
            cells = []
            for target in TARGETS:
                row = t[target]["signals"][s]
                a = row["all"]
                cells.append(f"{a['auroc']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]")
                cells += [f"{row[f]['auroc']:.3f}" for f in fam]
            lines.append(f"{MODEL_LABEL[m] if j == 0 else ''} & {SIGNAL_LABEL[s]} & "
                         + " & ".join(cells) + r" \\")
        if i < len(models) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")
    print("saved", path)


def table_subsets(res, path):
    """Write the LaTeX signal-subset accuracy table. Returns nothing."""
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Model & Signals fitted & Accuracy & $-$ \{C, A, S\} & $-$ bias only & $-$ original \\",
             r"\midrule"]
    models = [m for m in MODELS if m in res["models"]]
    for i, m in enumerate(models):
        r = res["models"][m]
        best = max(SUBSETS, key=lambda s: r["subsets"][s]["mCA"])
        for j, sub in enumerate(SUBSETS):
            x = r["subsets"][sub]
            acc = f"{x['mCA']:.2f}"
            acc = rf"\textbf{{{acc}}}" if sub == best else acc
            vc = "--" if x["vs_CAS"] is None else _ci(x["vs_CAS"], sign=True)
            vb = "--" if x["vs_bias_only"] is None else _ci(x["vs_bias_only"], sign=True)
            lines.append(f"{MODEL_LABEL[m] if j == 0 else ''} & {SUBSET_LABEL[sub]} & {acc} & "
                         f"{vc} & {vb} & {_ci(x['vs_original'], sign=True)} \\\\")
        if i < len(models) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")
    print("saved", path)


def plot_auroc(res, stem):
    """Plot per-signal AUROC by family for every model; saves <stem>.pdf/.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {"resnet18": "#2a78d6", "deit": "#eb6834", "convnext": "#1baf7a"}
    marker = {"resnet18": "o", "deit": "s", "convnext": "D"}
    ink, ink2, grid, muted = "#0b0b0b", "#52514e", "#dedcd4", "#8a8880"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": ink2,
        "axes.linewidth": 0.6, "axes.labelcolor": ink, "text.color": ink,
        "xtick.color": ink2, "ytick.color": ink2, "axes.grid": True, "grid.color": grid,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
    })
    models = [m for m in MODELS if m in res["models"]]
    targets = list(TARGETS)
    fig, axes = plt.subplots(len(targets), len(SIGNALS), figsize=(8.6, 5.2), sharey="row")
    x = np.arange(len(GROUPS))
    off = (np.arange(len(models)) - (len(models) - 1) / 2) * 0.22
    row_title = {"correct": "(a) target: view $k$ is correct (score $s_k$)",
                 "beats_original": "(b) target: view $k$ is correct and the original is wrong "
                                   "(score $s_k - s_0$)"}
    for r, target in enumerate(targets):
        for c, s in enumerate(SIGNALS):
            ax = axes[r, c]
            ax.axhline(0.5, color=muted, lw=0.8, ls=":", zorder=1)
            for i, m in enumerate(models):
                row = res["models"][m]["targets"][target]["signals"][s]
                y = np.array([row[g]["auroc"] for g in GROUPS])
                lo = y - np.array([row[g]["ci95"][0] for g in GROUPS])
                hi = np.array([row[g]["ci95"][1] for g in GROUPS]) - y
                ax.errorbar(x + off[i], y, yerr=[lo, hi], ls="none", marker=marker[m], ms=4,
                            color=color[m], mec="white", mew=0.6, elinewidth=0.9, capsize=0,
                            label=MODEL_LABEL[m], zorder=3)
            ax.axvline(0.5, color=grid, lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(["all"] + list(FAMILIES), rotation=35, ha="right")
            ax.grid(axis="x", visible=False)
            ax.set_title(SIGNAL_LABEL[s])
            if c == 0:
                ax.set_ylabel("AUROC")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(models), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3.0)
    for r, target in enumerate(targets):
        top = axes[r, 0].get_position().y1
        fig.text(axes[r, 0].get_position().x0, top + 0.045, row_title[target],
                 ha="left", va="bottom", fontsize=9, fontweight="bold")
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    print("saved", f"{stem}.pdf/.png")


def write_outputs(out, figures, parts):
    """Write the tables/figures for the requested parts from the saved jsons."""
    if "A" in parts and (out / "signal_auroc.json").exists():
        a = json.loads((out / "signal_auroc.json").read_text())
        table_auroc(a, Path(figures) / "table_signal_auroc.tex")
        plot_auroc(a, Path(figures) / "signal_auroc")
    if "B" in parts and (out / "signal_subsets.json").exists():
        table_subsets(json.loads((out / "signal_subsets.json").read_text()),
                      Path(figures) / "table_signal_subsets.tex")


def main():
    """Run parts A and/or B for every model, save the jsons and write the outputs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--parts", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--cache-dir", default="cache_fixed")
    ap.add_argument("--tag", default="cache_fixed_lofo")
    ap.add_argument("--results", default="results",
                    help="pipeline results (for the reproduction check of Part B)")
    ap.add_argument("--fits", default="results/signal_ablation",
                    help="where tune_fusion_ce.py --signals wrote the subset fits")
    ap.add_argument("--out", default="results/signal_ablation")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-chunk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if args.plot_only:
        write_outputs(out, args.figures, args.parts)
        return
    out.mkdir(parents=True, exist_ok=True)
    cache.CACHE_ROOT = Path(args.cache_dir)
    device = device_from_arg(args.device)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_a = {"timestamp": stamp, "cache_dir": args.cache_dir, "protocol": {
        "signals": {"C": "max class probability", "A": "agreement, mean -JSD to all other views",
                    "Ao": "agreement, -JSD to the original view", "S": "stability, m=1"},
        "raw_signals": "not z-scored (src.analysis.profile_signals, standardize=False)",
        "targets": TARGETS,
        "auroc": "within each corruption x severity condition (ties = 1/2), mean over the "
                 "conditions of the group, then mean over the 7 non-original views",
        "bootstrap": f"cluster bootstrap over test-half images, the same draw for every "
                     f"condition, view and signal; n_boot={args.n_boot}, seed={args.seed}",
        "split": "test half"}, "models": {}}
    res_b = {"timestamp": stamp, "cache_dir": args.cache_dir, "protocol": {
        "fit": "tune_fusion_ce.py --variants bias --signals <subset> (CE + per-view bias, "
               "non-negative signal coefficients, coefficients of excluded signals held at 0; "
               "same m search, grid warm start, l2, scale and output-temperature search as the "
               "pipeline's CE + bias); agreement = all, except the 'CAoS' subset (origin)",
        "evaluation": "test half, each corruption scored with the fold that held its family "
                      "out; mean accuracy over the 75 conditions (%)",
        "bootstrap": f"paired cluster bootstrap over images, n_boot={args.n_boot}, "
                     f"seed={args.seed}"}, "models": {}}
    for m in args.models:
        print(f"\n=== {m} ===")
        ref = latest_fit(args.results, "ce_bias", m, args.tag)
        split = (json.loads(Path(ref).read_text()).get("split") if ref else None) \
            or {"frac_tune": 0.5, "seed": 0}
        data = load_test_half(m, split, with_probs="B" in args.parts)
        if "A" in args.parts:
            res_a["models"][m] = part_a(m, data, args, device)
            (out / "signal_auroc.json").write_text(json.dumps(res_a, indent=2))
        if "B" in args.parts:
            res_b["models"][m] = part_b(m, data, args, device)
            (out / "signal_subsets.json").write_text(json.dumps(res_b, indent=2))
        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_outputs(out, args.figures, args.parts)


if __name__ == "__main__":
    main()
