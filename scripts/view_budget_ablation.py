"""Accuracy vs. number of views: the combined method at budgets of 2, 4, 8 and 15
views (forward passes), every budget including the original image.

Works only on the existing combined cache (no model inference) and the frozen
combined configurations from the main pipeline:

1. View subsets. For each leave-one-family-out fold, views are added greedily
   to {original}: at each step the view whose addition gives the highest
   tune-half accuracy of the plain average (mean over the fold's 11-12 tuning
   corruptions x 5 severities) is added; ties go to the lower view index. The
   subsets are nested (2 in 4 in 8 in 15) and depend only on tune-half images
   of the families the fold was tuned on. 15 = all views.
2. Fusion. Each model keeps the fusion method selected for the full combined
   set (results/tuning_<model>_cache_combined_lofo_final_*.json). For budgets
   below 15 that method is re-fit per fold on the subset with the same CE fit,
   accuracy-scale and output-temperature search as scripts/tune_fusion_ce.py
   (same signals, grids, l2 and optimizer settings), warm-started from the
   fold's 15-view CE direction. Within a subset the low-pass views keep their
   ladder order and form the ladder for the stability signal. The 15-view
   budget uses the frozen configuration unchanged.
3. Evaluation. Test half only, each corruption scored with the subset and
   configuration of the fold that held its family out; mean accuracy over the
   75 corruption x severity conditions; 95% CIs from the paired cluster
   bootstrap over images (src/bootstrap.py).

    python scripts/view_budget_ablation.py                 # all three models
    python scripts/view_budget_ablation.py --plot-only     # figure from the saved JSON

Writes <out>/view_budget.json, <out>/view_budget.csv and <figure>.pdf/.png.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import cache
from src.bootstrap import cluster_bootstrap, image_keys
from src.ce_fusion import fit_coefficients_torch
from src.evaluation import device_from_arg, fused_correct_many_torch, signal_variants_torch
from src.metrics import FusionConfig
from src.splits import CORRUPTION_FAMILY, leave_one_family_out, tune_test_masks
from scripts.tune_fusion_ce import (concatenate_conditions, tune_output_temperature,
                                    tune_scale_accuracy)

MODELS = ["resnet18", "deit", "convnext"]
SEVERITIES = [1, 2, 3, 4, 5]
#: views of the combined profile (view 0 = original, 1-7 low-pass, 8-14 crop/flip)
VIEW_NAMES = ["original", "lp_0.90", "lp_0.75", "lp_0.60", "lp_0.45", "lp_0.30",
              "lp_0.20", "lp_0.12", "hflip", "top_left", "top_right", "bottom_left",
              "bottom_right", "top_left_hflip", "bottom_right_hflip"]
LADDER_LEN = 8


def frozen_tuning(results, model):
    files = sorted(glob.glob(str(Path(results) / f"tuning_{model}_cache_combined_lofo_final_*.json")))
    if not files:
        raise SystemExit(f"no frozen combined tuning for {model} in {results}")
    return files[-1]


def pipeline_eval(results, model):
    files = sorted(glob.glob(str(Path(results) / f"eval_{model}_cache_combined_lofo_final_*.json")))
    return json.loads(Path(files[-1]).read_text()) if files else None


def load_model(model, split, device):
    """Every condition once: tune half resident on the GPU, test half in RAM."""
    tune, test = {}, {}
    t0 = time.time()
    for corruption in CORRUPTION_FAMILY:
        for sev in SEVERITIES:
            d = cache.load_condition(model, corruption, sev)
            tm, sm = tune_test_masks(d["paths"], split["frac_tune"], split["seed"])
            lab = np.asarray(d["labels"], dtype=np.int64)
            tune[(corruption, sev)] = {
                "probs": torch.from_numpy(np.ascontiguousarray(d["probs"][tm])).to(device),
                "labels": lab[tm],
                "labels_t": torch.from_numpy(lab[tm]).to(device),
            }
            test[(corruption, sev)] = {
                "probs": np.ascontiguousarray(d["probs"][sm]),
                "labels": lab[sm],
                "keys": image_keys(d["paths"][sm]),
            }
    print(f"  loaded {len(tune)} conditions in {time.time() - t0:.0f}s")
    return tune, test


def greedy_order(tune, conds, n_views):
    """Greedy forward order of all views, starting from the original.

    Returns (order, trace): the views in the order they were added and the
    tune-half plain-average accuracy after each addition.
    """
    order, trace = [0], [float(np.mean([
        (tune[cs]["probs"][:, 0].argmax(-1) == tune[cs]["labels_t"]).float().mean().item()
        for cs in conds]))]
    while len(order) < n_views:
        rem = [v for v in range(n_views) if v not in order]
        score = torch.zeros(len(rem), dtype=torch.float64, device=tune[conds[0]]["labels_t"].device)
        with torch.inference_mode():
            for cs in conds:
                p, y = tune[cs]["probs"], tune[cs]["labels_t"]
                base = p[:, order].float().sum(1)
                for j, v in enumerate(rem):
                    score[j] += (((base + p[:, v].float()).argmax(-1)) == y).double().mean()
        score = (score / len(conds)).cpu().numpy()
        j = int(np.argmax(score))  # first maximum = lowest view index
        order.append(rem[j])
        trace.append(float(score[j]))
    return order, trace


def subset_of(order, budget):
    """The first 'budget' views, in profile order (low-pass ladder first)."""
    idx = sorted(order[:budget])
    return idx, sum(1 for v in idx if v < LADDER_LEN)


def refit(tune, conds, idx, ladder_len, rec, fold_entry, device):
    """Re-fit the frozen method on a view subset, as tune_fusion_ce.py does."""
    m_values, agreements = rec["m_values"], rec["agreement_modes"]
    keys = [(m, a) for m in m_values for a in agreements]
    by_key = {k: {} for k in keys}
    conditions = {}
    with torch.inference_mode():
        for cs in conds:
            p = tune[cs]["probs"][:, idx].contiguous()
            y = tune[cs]["labels_t"]
            pf = p.float()
            _, std = signal_variants_torch(pf, m_values, agreements, ladder_len=ladder_len)
            p_true = pf.gather(2, y[:, None, None].expand(-1, pf.shape[1], 1)).squeeze(2)
            for k in keys:
                by_key[k][cs] = torch.stack(std[k], dim=-1).cpu().numpy().astype(np.float32)
            conditions[cs] = {"probs": p, "labels": tune[cs]["labels"], "labels_t": y,
                              "p_true": p_true.cpu().numpy().astype(np.float32)}
            del pf
    corrs = sorted({c for c, _ in conds}, key=list(CORRUPTION_FAMILY).index)
    fixed_zero = bool(rec["theta_fixed_zero"])
    starts = None if fixed_zero else [np.asarray(fold_entry["ce_direction"], dtype=float)]
    best = None
    for k in keys:
        X, p_true = concatenate_conditions(by_key, conditions, corrs, SEVERITIES, k)
        theta, bias, obj, res = fit_coefficients_torch(
            X, p_true, l2=rec["l2"], bias_l2=rec["bias_l2"], eps=rec["eps"], starts=starts,
            random_restarts=4, seed=0, maxiter=500, nonnegative=bool(rec["nonnegative"]),
            view_bias=bool(rec["view_bias"]), theta_fixed_zero=fixed_zero,
            device=device, dtype=rec.get("ce_dtype", "float32"))
        if best is None or obj < best["obj"]:
            best = {"key": k, "theta": theta, "bias": bias, "obj": float(obj),
                    "success": bool(res.success)}
    k = best["key"]
    scales = np.asarray(rec["scale_grid"], dtype=np.float32)
    temps = np.asarray(rec["temperature_grid"], dtype=np.float32)
    scale, tune_acc, _ = tune_scale_accuracy(conditions, by_key, k, best["theta"], best["bias"],
                                             corrs, SEVERITIES, scales, device=device)
    temp, _, _ = tune_output_temperature(conditions, by_key, k, best["theta"], best["bias"],
                                         scale, corrs, SEVERITIES, temps, device=device,
                                         eps=rec["eps"])
    th, b = scale * best["theta"], scale * best["bias"]
    cfg = FusionConfig(alpha=float(th[0]), beta=float(th[1]), gamma=float(th[2]), tau=1.0,
                       standardize=True, m=int(k[0]), agreement=k[1],
                       view_bias=tuple(float(x) for x in b), output_temperature=float(temp))
    return cfg, {"tune_half_acc": float(tune_acc), "objective": best["obj"],
                 "optimizer_success": best["success"], "selection_scale": float(scale)}


def evaluate(test, idx_of, ladder_of, cfg_of, device):
    """Per-image test-half correctness of the uniform and fused predictors."""
    uni, fus, orig = {}, {}, {}
    for cs, d in test.items():
        fam = CORRUPTION_FAMILY[cs[0]]
        p = d["probs"][:, idx_of[fam]]
        with torch.inference_mode():
            pt = torch.from_numpy(p).to(device)
            uni[cs] = (pt.float().mean(1).argmax(-1).cpu().numpy() == d["labels"])
            orig[cs] = (pt[:, 0].float().argmax(-1).cpu().numpy() == d["labels"])
        if cfg_of is not None:
            ok, _ = fused_correct_many_torch(p, d["labels"], {"f": cfg_of[fam]},
                                             ladder_len=ladder_of[fam], device=device)
            fus[cs] = ok["f"]
    return uni, fus, orig


def mca(correct):
    return float(np.mean([v.mean() for v in correct.values()]))


def boot(test, a, b, n_boot, seed):
    r = cluster_bootstrap([{"keys": test[cs]["keys"], "a": a[cs], "b": b[cs]} for cs in test],
                          n_boot=n_boot, seed=seed)
    return {"point": 100 * r["point"], "ci95": [100 * x for x in r["ci95"]],
            "p_le_zero": r["p_le_zero"]}


def run_model(model, args, device):
    print(f"\n=== {model} ===")
    frozen_path = frozen_tuning(args.results, model)
    rec = json.loads(Path(frozen_path).read_text())
    if rec.get("ladder_len") != LADDER_LEN:
        raise SystemExit(f"{frozen_path}: ladder_len {rec.get('ladder_len')} != {LADDER_LEN}")
    split = rec.get("split") or {"frac_tune": 0.5, "seed": 0}
    tune, test = load_model(model, split, device)
    n_views = next(iter(tune.values()))["probs"].shape[1]
    budgets = sorted(args.budgets)
    if budgets[-1] != n_views:
        raise SystemExit(f"largest budget must be all {n_views} views")

    folds = {}
    for held, tune_corr, _ in leave_one_family_out():
        conds = [(c, s) for c in tune_corr for s in SEVERITIES]
        order, trace = greedy_order(tune, conds, n_views)
        folds[held] = {"conds": conds, "order": order, "trace": trace}
        print(f"  fold {held:8s}: " + " > ".join(VIEW_NAMES[v] for v in order[:8]) + " > ...")

    frozen_cfg = {fam: FusionConfig.from_dict(rec["held_out"][fam]["config"]) for fam in folds}
    corr = {}
    out = {"selected_method": rec["method"], "frozen_tuning": frozen_path,
           "subsets": {fam: {"greedy_order": [VIEW_NAMES[v] for v in f["order"]],
                             "tune_half_plain_avg_acc_after_each_step": f["trace"]}
                       for fam, f in folds.items()},
           "budgets": {}}
    for b in budgets:
        idx_of, ladder_of, cfg_of, fit_info = {}, {}, {}, {}
        for fam, f in folds.items():
            idx_of[fam], ladder_of[fam] = subset_of(f["order"], b)
            if b == n_views:
                cfg_of[fam] = frozen_cfg[fam]
            else:
                t0 = time.time()
                cfg_of[fam], fit_info[fam] = refit(tune, f["conds"], idx_of[fam], ladder_of[fam],
                                                   rec, rec["held_out"][fam], device)
                print(f"  budget {b:2d} fold {fam:8s}: refit {time.time() - t0:.0f}s, "
                      f"tune acc {100 * fit_info[fam]['tune_half_acc']:.2f}%")
        uni, fus, orig = evaluate(test, idx_of, ladder_of, cfg_of, device)
        corr[b] = {"uniform": uni, "fused": fus}
        corr["orig"] = orig
        out["budgets"][str(b)] = {
            "views": {fam: [VIEW_NAMES[v] for v in idx_of[fam]] for fam in folds},
            "ladder_len": ladder_of,
            "configs": {fam: cfg_of[fam].as_dict() for fam in folds},
            "fit": fit_info or "frozen 15-view configuration",
        }
        print(f"  budget {b:2d}: uniform {100 * mca(uni):.2f}  fused {100 * mca(fus):.2f}")

    # the refit procedure applied to all 15 views, as a check that it reproduces
    # the frozen configuration's accuracy
    refit_check = {}
    if not args.skip_refit_check:
        idx_full = list(range(n_views))
        cfg_full = {fam: refit(tune, f["conds"], idx_full, LADDER_LEN, rec, rec["held_out"][fam],
                               device)[0] for fam, f in folds.items()}
        _, fus_refit, _ = evaluate(test, {f: idx_full for f in folds},
                                   {f: LADDER_LEN for f in folds}, cfg_full, device)
        refit_check = {"refit_15_fused_mCA": 100 * mca(fus_refit)}
    del tune

    orig = corr["orig"]
    full = corr[n_views]["fused"]
    out["orig_mCA"] = 100 * mca(orig)
    ref = pipeline_eval(args.results, model)
    out["checks"] = {
        "pipeline_mCA_fused": None if ref is None else 100 * ref["mCA_fused"],
        "pipeline_mCA_equal": None if ref is None else 100 * ref["mCA_equal"],
        "pipeline_mCA_orig": None if ref is None else 100 * ref["mCA_orig"],
        "this_15_fused_mCA": 100 * mca(full),
        "this_15_uniform_mCA": 100 * mca(corr[n_views]["uniform"]),
        **refit_check,
    }
    if ref is not None and abs(ref["mCA_fused"] - mca(full)) > 1e-6:
        raise SystemExit(f"{model}: 15-view fused {mca(full)} != pipeline {ref['mCA_fused']}")
    print("  checks:", {k: (round(v, 4) if v is not None else v) for k, v in out["checks"].items()})

    for b in budgets:
        row = out["budgets"][str(b)]
        for meth in ("uniform", "fused"):
            c = corr[b][meth]
            row[meth] = {
                "mCA": 100 * mca(c),
                "gain_vs_original": boot(test, c, orig, args.n_boot, args.seed),
                "diff_vs_full_15_fused": (None if (b == n_views and meth == "fused")
                                          else boot(test, c, full, args.n_boot, args.seed)),
            }
        row["fused_minus_uniform"] = boot(test, corr[b]["fused"], corr[b]["uniform"],
                                          args.n_boot, args.seed)
    return out


def write_csv(res, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "forward_passes", "method", "mCA", "gain_vs_original",
                    "gain_ci_lo", "gain_ci_hi", "diff_vs_full_15", "diff_ci_lo", "diff_ci_hi"])
        for m, r in res["models"].items():
            w.writerow([m, 1, "original", f"{r['orig_mCA']:.3f}", "", "", "", "", "", ""])
            for b, row in r["budgets"].items():
                for meth in ("uniform", "fused"):
                    x = row[meth]
                    g, d = x["gain_vs_original"], x["diff_vs_full_15_fused"]
                    w.writerow([m, b, meth, f"{x['mCA']:.3f}", f"{g['point']:.3f}",
                                f"{g['ci95'][0]:.3f}", f"{g['ci95'][1]:.3f}",
                                *(["", "", ""] if d is None else
                                  [f"{d['point']:.3f}", f"{d['ci95'][0]:.3f}", f"{d['ci95'][1]:.3f}"])])


def plot(res, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {"resnet18": "#2a78d6", "deit": "#eb6834", "convnext": "#1baf7a"}
    label = {"resnet18": "ResNet-18", "deit": "DeiT-Tiny", "convnext": "ConvNeXt-Tiny"}
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
    fig, axes = plt.subplots(1, len(models), figsize=(2.5 * len(models), 2.6))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        r = res["models"][m]
        o = r["orig_mCA"]
        bs = sorted(int(b) for b in r["budgets"])
        xs = [1] + bs
        ax.axhline(o, color=muted, lw=0.8, ls=":", zorder=1)
        for meth, col, ls, mk, fc in (("uniform", ink2, "--", "s", "white"),
                                      ("fused", color[m], "-", "o", color[m])):
            y = [o] + [r["budgets"][str(b)][meth]["mCA"] for b in bs]
            lo = [0] + [y[i + 1] - (o + r["budgets"][str(b)][meth]["gain_vs_original"]["ci95"][0])
                        for i, b in enumerate(bs)]
            hi = [0] + [(o + r["budgets"][str(b)][meth]["gain_vs_original"]["ci95"][1]) - y[i + 1]
                        for i, b in enumerate(bs)]
            ax.errorbar(xs, y, yerr=[lo, hi], color=col, ls=ls, lw=1.6 if meth == "fused" else 1.2,
                        marker=mk, ms=4.5, mfc=fc, mec=col, mew=1.1, capsize=2, elinewidth=0.8,
                        label="fused (selected method)" if meth == "fused" else "uniform average",
                        zorder=3 if meth == "fused" else 2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.minorticks_off()
        ax.set_xlim(0.8, 19)
        ax.set_title(label[m])
        ax.set_xlabel("forward passes (views)")
        ax.text(19, o, "original", color=ink2, fontsize=7, ha="right", va="bottom")
    axes[0].set_ylabel("mean corruption accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    print("saved", f"{stem}.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--budgets", nargs="+", type=int, default=[2, 4, 8, 15])
    ap.add_argument("--cache-dir", default="cache_combined")
    ap.add_argument("--results", default="results", help="where the frozen tuning JSONs are")
    ap.add_argument("--out", default="results/view_budget")
    ap.add_argument("--figure", default="figures/view_budget")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-refit-check", action="store_true",
                    help="skip re-fitting all 15 views (a check of the refit procedure)")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if args.plot_only:
        plot(json.loads((out / "view_budget.json").read_text()), args.figure)
        return
    out.mkdir(parents=True, exist_ok=True)
    cache.CACHE_ROOT = Path(args.cache_dir)
    device = device_from_arg(args.device)
    res = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "cache_dir": args.cache_dir,
        "budgets": sorted(args.budgets),
        "protocol": {
            "views": VIEW_NAMES,
            "subset_selection": "per leave-one-family-out fold, greedy forward from {original}: "
                                "add the view maximising tune-half accuracy of the plain average, "
                                "averaged over the fold's tuning corruptions x severities 1-5; "
                                "ties -> lower view index; nested subsets",
            "fusion": "method selected for the full combined set, re-fit per fold on the subset "
                      "(tune_fusion_ce.py CE fit + scale + temperature, same grids/l2/optimizer, "
                      "warm start = the fold's 15-view CE direction); 15 views = frozen config",
            "evaluation": "test half, LOFO (each corruption uses its held-out fold's subset and "
                          "config), mean accuracy over 75 conditions, clean images not scored",
            "bootstrap": f"paired cluster bootstrap over images, n_boot={args.n_boot}, "
                         f"seed={args.seed}",
        },
        "models": {},
    }
    for m in args.models:
        res["models"][m] = run_model(m, args, device)
        torch.cuda.empty_cache() if device.type == "cuda" else None
        (out / "view_budget.json").write_text(json.dumps(res, indent=2))
    write_csv(res, out / "view_budget.csv")
    print("saved", out / "view_budget.json", out / "view_budget.csv")
    plot(res, args.figure)


if __name__ == "__main__":
    main()
