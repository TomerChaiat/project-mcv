"""Paper-grade figures and tables (vector PDF + 300 DPI PNG + LaTeX) from the
pipeline outputs in results/ (scripts/run_full_pipeline.sh).

Every number comes from the true leave-one-family-out, test-half evaluation
of the per-model *selected* method (chosen on tune-half accuracy only by
scripts/select_method.py), and every CI from the paired cluster bootstrap
over images (src/bootstrap.py). Three models, two profiles:
  spectral  = cache_fixed     (8 views: original + 7 low-pass)
  combined  = cache_combined  (15 views: spectral + 7 spatial-TTA views)

    python scripts/make_paper_figures.py
    RESULTS_DIR=... FIGURES_DIR=... python scripts/make_paper_figures.py
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(os.environ.get("RESULTS_DIR", "results"))
OUT = Path(os.environ.get("FIGURES_DIR", "figures"))
OUT.mkdir(exist_ok=True)
SPECTRAL_CACHE = Path(os.environ.get("SPECTRAL_CACHE", "cache_fixed"))

# ---- categorical palette: reference slots 1-3 (blue / orange / aqua), ----
# ---- validated all-pairs: worst CVD Delta E 9.2, normal-vision 24.0 (light) ----
RESNET_COLOR = "#2a78d6"
DEIT_COLOR = "#eb6834"
CONVNEXT_COLOR = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dedcd4"
ZERO_LINE = "#8a8880"
NEUTRAL_BAR = "#c9c7bd"

MODEL_COLOR = {"resnet18": RESNET_COLOR, "deit": DEIT_COLOR, "convnext": CONVNEXT_COLOR}
MODEL_LABEL = {"resnet18": "ResNet-18", "deit": "DeiT-Tiny", "convnext": "ConvNeXt-Tiny"}
MODEL_ORDER = ["resnet18", "deit", "convnext"]

PROFILE_TAG = {"spectral": "cache_fixed_lofo", "combined": "cache_combined_lofo"}
PROFILE_LABEL = {"spectral": "Spectral (8 views)", "combined": "Combined (15 views)"}
METHODS = ["grid", "ce", "ce_bias", "ce_unconstrained", "ce_bias_only"]
METHOD_LABEL = {"grid": "Grid", "ce": "CE", "ce_bias": "CE + bias",
                "ce_unconstrained": "CE (unconstr.)", "ce_bias_only": "Bias only"}

FAMILIES = {
    "noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
CORRUPTION_ORDER = [c for cs in FAMILIES.values() for c in cs]
CORRUPTION_FAMILY = {c: f for f, cs in FAMILIES.items() for c in cs}
CORR_LABEL = {
    "gaussian_noise": "Gaussian\nnoise", "shot_noise": "Shot\nnoise",
    "impulse_noise": "Impulse\nnoise", "defocus_blur": "Defocus\nblur",
    "glass_blur": "Glass\nblur", "motion_blur": "Motion\nblur",
    "zoom_blur": "Zoom\nblur", "snow": "Snow", "frost": "Frost", "fog": "Fog",
    "brightness": "Bright-\nness", "contrast": "Con-\ntrast",
    "elastic_transform": "Elastic\ntransform", "pixelate": "Pixelate",
    "jpeg_compression": "JPEG\ncompression",
}
LADDER_LABELS = ["Orig.", "0.90", "0.75", "0.60", "0.45", "0.30", "0.20", "0.12"]
VIEW_NAMES = ["Original", "LP 0.90", "LP 0.75", "LP 0.60", "LP 0.45", "LP 0.30",
              "LP 0.20", "LP 0.12", "H-flip", "Top-left", "Top-right",
              "Bottom-left", "Bottom-right", "TL+flip", "BR+flip"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _latest(pattern, regex):
    """Newest file matching a glob, filtered by an anchored regex (so that e.g.
    '..._ce_bias_<stamp>' never matches '..._ce_bias_only_<stamp>')."""
    rx = re.compile(regex)
    files = sorted(f for f in glob.glob(str(R / pattern)) if rx.search(os.path.basename(f)))
    if not files:
        raise SystemExit(f"missing result: {R / pattern}")
    return files[-1]


def _eval_path(model, profile, method="final", ext="json"):
    """Path of the newest eval_<model>_<profile tag>_<method> file with extension `ext`."""
    tag = f"{PROFILE_TAG[profile]}_{method}"
    return _latest(f"eval_{model}_{tag}_*.{ext}",
                   rf"^eval_{model}_{re.escape(tag)}_\d{{8}}_\d{{6}}\.{ext}$")


def load_eval_json(model, profile, method="final"):
    """Newest eval json (summary metrics) for (model, profile, method), as a dict."""
    return json.loads(Path(_eval_path(model, profile, method, "json")).read_text())


def load_eval_csv(model, profile, method="final"):
    """Newest eval csv (per-condition rows) for (model, profile, method), as a DataFrame."""
    return pd.read_csv(_eval_path(model, profile, method, "csv"))


def load_bootstrap(profile):
    """The final bootstrap_ci json for `profile`, as a dict keyed by model."""
    return json.loads((R / f"bootstrap_ci_{PROFILE_TAG[profile]}_final.json").read_text())


def load_final_tuning(model, profile):
    """Newest frozen (final) tuning json for (model, profile), as a dict."""
    tag = PROFILE_TAG[profile]
    return json.loads(Path(_latest(f"tuning_{model}_{tag}_final_*.json",
                                   rf"^tuning_{model}_{tag}_final_\d{{8}}_\d{{6}}\.json$")).read_text())


def load_selection(profile):
    """Per-model selection records, merged over every selection file of this
    profile (models may have been selected in separate runs); newest wins."""
    tag = PROFILE_TAG[profile]
    rx = re.compile(rf"^selection_{tag}_\d{{8}}_\d{{6}}\.json$")
    merged = {}
    for f in sorted(glob.glob(str(R / f"selection_{tag}_*.json"))):
        if rx.search(os.path.basename(f)):
            merged.update(json.loads(Path(f).read_text()))
    if not merged:
        raise SystemExit(f"missing selection file for {tag} in {R}")
    return merged


def load_vs_biasonly(model, profile):
    """Selected predictor vs the bias-only fit bootstrap record, or None if not run."""
    p = R / f"bootstrap_selected_vs_biasonly_{model}_{PROFILE_TAG[profile]}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_vs_equal(model, profile):
    """Selected fused predictor vs the plain equal-weight average of the same views."""
    p = R / f"bootstrap_selected_vs_equal_{model}_{PROFILE_TAG[profile]}.json"
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
def save(fig, name):
    """Save `fig` as <name>.pdf and a 300 DPI <name>.png in OUT, then close it."""
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


def zero_line(ax):
    """Draw a horizontal reference line at y = 0 on `ax`."""
    ax.axhline(0.0, color=ZERO_LINE, linewidth=0.8, zorder=1)


def clean_axes(ax, xgrid=False):
    """Hide the top/right spines and, unless `xgrid`, the vertical grid lines."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if not xgrid:
        ax.grid(axis="x", visible=False)


def grouped_bars(ax, x, values_by_model, width=0.8):
    """One bar group per x position, one bar per model, fixed model order."""
    n = len(MODEL_ORDER)
    w = width / n
    for i, m in enumerate(MODEL_ORDER):
        ax.bar(x + (i - (n - 1) / 2) * w, values_by_model[m], width=w * 0.92,
               color=MODEL_COLOR[m], label=MODEL_LABEL[m], zorder=3)


def family_separators(ax, ymax=None, labels=True):
    """Draw vertical lines between corruption families and, if `labels` and
    `ymax` are given, the family names at the top. Returns nothing.
    """
    pos = 0
    for fam, corrs in FAMILIES.items():
        if pos > 0:
            ax.axvline(pos - 0.5, color=GRID, linewidth=0.8, zorder=0)
        if labels and ymax is not None:
            ax.text(pos + (len(corrs) - 1) / 2, ymax * 0.97, fam, ha="center", va="top",
                    fontsize=7.5, color=INK2, style="italic")
        pos += len(corrs)


def gains_by(df, key, values):
    """Mean fused - original accuracy (pts) of `df` for each value of column `key`.

    Returns:
        List of gains, one per entry of `values`.
    """
    return [100 * (df[df[key] == v]["fused_acc"] - df[df[key] == v]["orig_acc"]).mean()
            for v in values]


def fmt_ci(point, ci):
    """Format a fractional point estimate and 95% CI as '+p [+lo, +hi]' in pts."""
    return f"{100 * point:+.2f} [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"


# --------------------------------------------------------------------------- #
# Figure: per-family gain, combined profile
# --------------------------------------------------------------------------- #
def fig_family_gain():
    """Figure: per-family fused - original gain, combined profile, all models."""
    fam_order = list(FAMILIES)
    gains = {}
    for m in MODEL_ORDER:
        df = load_eval_csv(m, "combined")
        gains[m] = [100 * (df[df["family"] == f]["fused_acc"] - df[df["family"] == f]["orig_acc"]).mean()
                    for f in fam_order]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    grouped_bars(ax, np.arange(len(fam_order)), gains)
    zero_line(ax)
    ax.set_xticks(np.arange(len(fam_order)))
    ax.set_xticklabels([f.capitalize() for f in fam_order])
    ax.set_ylabel("Top-1 accuracy gain (pts)")
    ax.legend(frameon=False, loc="upper right")
    clean_axes(ax)
    save(fig, "family_gain_combined")
    for i, f in enumerate(fam_order):
        print(f"    {f:8s} " + "  ".join(f"{m} {gains[m][i]:+.2f}" for m in MODEL_ORDER))


# --------------------------------------------------------------------------- #
# Figure: severity trend, combined profile
# --------------------------------------------------------------------------- #
def fig_severity_trend():
    """Figure: mean fused - original gain vs severity, combined profile."""
    sev = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in MODEL_ORDER:
        g = gains_by(load_eval_csv(m, "combined"), "severity", sev)
        ax.plot(sev, g, marker="o", markersize=4, linewidth=1.4, color=MODEL_COLOR[m],
                label=MODEL_LABEL[m])
        print(f"    {m}: " + "  ".join(f"s{s}={v:+.2f}" for s, v in zip(sev, g)))
    zero_line(ax)
    ax.set_xticks(sev)
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("Mean accuracy gain (pts)")
    ax.legend(frameon=False, loc="best")
    clean_axes(ax, xgrid=True)
    save(fig, "severity_trend_combined")


# --------------------------------------------------------------------------- #
# Table: main results (combined profile) and profile comparison
# --------------------------------------------------------------------------- #
def _tuned_spatial(m):
    """(mCA of the tuned spatial TTA, combined - tuned spatial, spectral - tuned
    spatial), from phase spatialfit; None if it has not been run."""
    pc, ps = R / f"bootstrap_combined_vs_spatial_tuned_{m}.json", R / f"bootstrap_spectral_vs_spatial_tuned_{m}.json"
    if not (pc.exists() and ps.exists()):
        return None
    f = _latest(f"eval_{m}_cache_spatial_lofo_final_*.json",
                rf"^eval_{m}_cache_spatial_lofo_final_\d{{8}}_\d{{6}}\.json$")
    return json.loads(Path(f).read_text())["mCA_fused"], _pair(pc), _pair(ps)


def table_main_results():
    """LaTeX table of the headline results (combined profile): vanilla,
    spatial TTA, final mCA and bootstrapped deltas, plus clean accuracy.
    """
    boot = load_bootstrap("combined")
    tuned = {m: _tuned_spatial(m) for m in MODEL_ORDER}
    has_tuned = all(tuned.values())
    head = ("Model & Vanilla & Spatial TTA (tuned) & Final (15v) & $\\Delta$ vs.\\ vanilla & "
            "$\\Delta$ vs.\\ spatial (tuned) & $\\Delta$ vs.\\ spatial (plain) & "
            "Clean (van.\\ $\\to$ final) \\\\") if has_tuned else (
            "Model & Vanilla & Spatial TTA (plain) & Final (15v) & $\\Delta$ vs.\\ vanilla & "
            "$\\Delta$ vs.\\ spatial (plain) & Clean (van.\\ $\\to$ final) \\\\")
    lines = [r"\begin{tabular}{l" + "r" * (7 if has_tuned else 6) + "}", r"\toprule", head, r"\midrule"]
    for m in MODEL_ORDER:
        j = load_eval_json(m, "combined")
        bo, bs = boot[m]["fused - orig"], boot[m]["fused - spatial TTA"]
        if has_tuned:
            sp_acc, bt, _ = tuned[m]
            cells = [f"{100 * sp_acc:.2f}", f"\\textbf{{{100 * j['mCA_fused']:.2f}}}",
                     fmt_ci(bo["point"], bo["ci95"]), fmt_ci(bt["point"], bt["ci95"]), fmt_ci(bs["point"], bs["ci95"])]
        else:
            cells = [f"{100 * (j['mCA_fused'] - bs['point']):.2f}", f"\\textbf{{{100 * j['mCA_fused']:.2f}}}",
                     fmt_ci(bo["point"], bo["ci95"]), fmt_ci(bs["point"], bs["ci95"])]
        lines.append(f"{MODEL_LABEL[m]} & {100 * j['mCA_orig']:.2f} & " + " & ".join(cells) +
                     f" & {100 * j['clean_orig_acc']:.1f} $\\to$ {100 * j['clean_fused_acc']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_table("table_main_results", lines)
    print("  Caption note: spatial TTA (tuned) = the 8 crop/flip views fused with the same methods, "
          "leave-one-family-out tuning and selection as the spectral views; (plain) = their plain average.")


def _pair(path):
    """The delta record ({'point', 'ci95', ...}) from a pairwise bootstrap json."""
    r = json.loads(path.read_text())
    d = r.get("delta", r)
    return d if "point" in d else next(v for v in d.values() if isinstance(v, dict) and "point" in v)


def table_spatial_tuned():
    """Like-for-like 8 vs 8: spatial TTA tuned exactly like the spectral views
    (phase spatialfit), and both fused view sets against it."""
    if not all((R / f"bootstrap_spectral_vs_spatial_tuned_{m}.json").exists() for m in MODEL_ORDER):
        print("  no spatialfit results (optional phase 'spatialfit') -- skipped")
        return
    rows = [("Spatial TTA, plain average", lambda m: f"{100 * load_eval_json(m, 'spectral')['mCA_orig'] + 100 * load_bootstrap('spectral')[m]['spatial TTA - orig']['point']:.2f}"),
            ("Spatial TTA, tuned", lambda m: f"{100 * json.loads(Path(_latest(f'eval_{m}_cache_spatial_lofo_final_*.json', rf'^eval_{m}_cache_spatial_lofo_final_\d{{8}}_\d{{6}}\.json$')).read_text())['mCA_fused']:.2f}"),
            ("Spectral, tuned", lambda m: f"{100 * load_eval_json(m, 'spectral')['mCA_fused']:.2f}"),
            ("Spectral $-$ spatial, both tuned", lambda m: fmt_ci(*(lambda d: (d['point'], d['ci95']))(_pair(R / f'bootstrap_spectral_vs_spatial_tuned_{m}.json')))),
            ("Combined $-$ spatial tuned", lambda m: fmt_ci(*(lambda d: (d['point'], d['ci95']))(_pair(R / f'bootstrap_combined_vs_spatial_tuned_{m}.json'))))]
    lines = [r"\begin{tabular}{l" + "r" * len(MODEL_ORDER) + "}", r"\toprule",
             " & " + " & ".join(MODEL_LABEL[m] for m in MODEL_ORDER) + r" \\", r"\midrule"]
    for label, f in rows:
        lines.append(f"{label} & " + " & ".join(f(m) for m in MODEL_ORDER) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_table("table_spatial_tuned", lines)
    print("  Caption note: mean corrupted accuracy (test half); spatial TTA tuned with the same fusion "
          "methods, leave-one-family-out tuning and tune-half selection as the spectral views; 95% paired "
          "cluster-bootstrap CIs.")


def worst_corruption(model, profile):
    """Worst-corruption accuracy (min over the 15 corruptions of the
    mean-over-severities accuracy), vanilla -> fused; the worst corruption
    itself can differ between the two."""
    g = load_eval_csv(model, profile).groupby("corruption")[["orig_acc", "fused_acc"]].mean() * 100
    return f"{g['orig_acc'].min():.1f} $\\to$ {g['fused_acc'].min():.1f}"


def fixed_broken(df):
    """Mean over conditions of images fixed / broken by fusion, in % of ALL
    images. The raw CSV rates divide by different denominators (positive =
    fixed / originally wrong, negative = broken / originally right), so they
    are not comparable to each other; fixed - broken here is the net gain."""
    fixed = 100 * (df["positive_transfer"] * (1 - df["orig_acc"])).mean()
    broken = 100 * (df["negative_transfer"] * df["orig_acc"]).mean()
    return fixed, broken


def table_profiles():
    """Spectral-only vs combined, per model, with cluster-bootstrap CIs."""
    tuned = {m: _tuned_spatial(m) for m in MODEL_ORDER}
    has_tuned = all(tuned.values())
    tta_cols = ("$\\Delta$ vs.\\ spatial (tuned) & $\\Delta$ vs.\\ spatial (plain)" if has_tuned
                else "$\\Delta$ vs.\\ spatial TTA (plain)")
    lines = [r"\begin{tabular}{ll" + "r" * (7 if has_tuned else 6) + "}", r"\toprule",
             "Model & Profile & mCA (\\%) & $\\Delta$ vs.\\ vanilla & " + tta_cols + " & "
             "Worst corr.\\ (van.\\ $\\to$ fused) & Fixed / broken (\\% of all) & Clean (\\%) \\\\", r"\midrule"]
    for m in MODEL_ORDER:
        for k, p in enumerate(("spectral", "combined")):
            j = load_eval_json(m, p)
            b = load_bootstrap(p)[m]
            bo, bs = b["fused - orig"], b["fused - spatial TTA"]
            name = MODEL_LABEL[m] if k == 0 else ""
            tta = fmt_ci(bs["point"], bs["ci95"])
            if has_tuned:
                bt = tuned[m][2] if p == "spectral" else tuned[m][1]
                tta = fmt_ci(bt["point"], bt["ci95"]) + " & " + tta
            lines.append(
                f"{name} & {PROFILE_LABEL[p]} & {100 * j['mCA_fused']:.2f} & "
                f"{fmt_ci(bo['point'], bo['ci95'])} & {tta} & "
                f"{worst_corruption(m, p)} & "
                "{:.1f} / {:.1f} & ".format(*fixed_broken(load_eval_csv(m, p)))
                + f"{100 * j['clean_fused_acc']:.1f} \\\\")
        if m != MODEL_ORDER[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_table("table_profiles", lines)
    print("  Caption note: spatial TTA (tuned) = the 8 crop/flip views fused with the same methods and "
          "tuning as the spectral views; the spectral row vs tuned spatial is the like-for-like 8-vs-8 "
          "comparison. (plain) = plain average of the 8 crop/flip views. Fixed / broken = images fusion "
          "turns right / wrong, as % of all images (their difference is the gain).")


def signal_ceiling(model, profile):
    """mCA of the LOFO logistic per-image scorer of C/A/S (scripts/signal_ceiling.py),
    family accuracies weighted by corruptions per family; None if not run."""
    cache_name = {"spectral": "cache_fixed", "combined": "cache_combined"}[profile]
    p = R / f"signal_ceiling_{cache_name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text()).get(model, {})
    if set(d) != set(FAMILIES):
        return None
    return 100 * sum(len(FAMILIES[f]) * d[f]["logistic_acc"] for f in FAMILIES) / len(CORRUPTION_ORDER)


def table_signal_ablation():
    """Test-half mCA of the equal-weight average and of every fusion method;
    which one tune-half selection picked; and whether it beats the equal
    average and the bias-only (fixed per-view prior) fit."""
    lines = [r"\begin{tabular}{ll" + "r" * (len(METHODS) + 2) + "rr}", r"\toprule",
             "Model & Profile & Equal avg. & " + " & ".join(METHOD_LABEL[x] for x in METHODS)
             + " & Per-image ceiling & Selected $-$ equal & Selected $-$ bias only \\\\", r"\midrule"]
    for m in MODEL_ORDER:
        for k, p in enumerate(("spectral", "combined")):
            sel = load_selection(p)[m]["selected"]
            cells = []
            for meth in METHODS:
                v = 100 * load_eval_json(m, p, meth)["mCA_fused"]
                cells.append(f"\\textbf{{{v:.2f}}}" if meth == sel else f"{v:.2f}")
            equal = 100 * load_eval_json(m, p)["mCA_equal"]
            b = load_vs_biasonly(m, p)
            diff = "(selected)" if sel == "ce_bias_only" or b is None else fmt_ci(b["point"], b["ci95"])
            e = load_vs_equal(m, p)
            ediff = "--" if e is None else fmt_ci(e["point"], e["ci95"])
            name = MODEL_LABEL[m] if k == 0 else ""
            ceil = signal_ceiling(m, p)
            ceil = "--" if ceil is None else f"{ceil:.2f}"
            lines.append(f"{name} & {PROFILE_LABEL[p]} & {equal:.2f} & " + " & ".join(cells)
                         + f" & {ceil} & {ediff} & {diff} \\\\")
        if m != MODEL_ORDER[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_table("table_signal_ablation", lines)
    print("  Caption note: bold = method chosen by tune-half accuracy (never test half); "
          "'Equal avg.' = plain unweighted mean over the same views; 'Per-image ceiling' = a logistic "
          "per-view correctness scorer on C/A/S, trained leave-one-family-out, used as the fusion score; "
          "'Bias only' fits a fixed per-view prior with alpha=beta=gamma=0, i.e. no per-image signal.")


def _write_table(name, lines):
    """Write `lines` to OUT/<name>.tex and print the table."""
    text = "\n".join(lines)
    (OUT / f"{name}.tex").write_text(text)
    print(f"  saved {name}.tex")
    print(text)


# --------------------------------------------------------------------------- #
# Figure: per-corruption gain, combined profile
# --------------------------------------------------------------------------- #
def fig_per_corruption_gain():
    """Figure: per-corruption fused - original gain, combined profile."""
    gains = {m: gains_by(load_eval_csv(m, "combined"), "corruption", CORRUPTION_ORDER)
             for m in MODEL_ORDER}
    x = np.arange(len(CORRUPTION_ORDER))
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    grouped_bars(ax, x, gains)
    zero_line(ax)
    ymax = max(max(v) for v in gains.values()) * 1.15
    ymin = min(0.0, min(min(v) for v in gains.values()) * 1.15)
    ax.set_ylim(ymin, ymax)
    family_separators(ax, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels([CORR_LABEL[c] for c in CORRUPTION_ORDER], fontsize=6.8)
    ax.set_ylabel("Top-1 accuracy gain (pts)")
    clean_axes(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, borderaxespad=0)
    save(fig, "per_corruption_gain_combined")


# --------------------------------------------------------------------------- #
# Table + figure: baselines and oracle, combined profile
# --------------------------------------------------------------------------- #
BASELINE_KEYS = [
    ("mCA_orig", "Original"),
    ("mCA_view1", "Mild LP (0.90)"),
    ("spatial", "Spatial TTA, plain (8 views)"),
    ("spatial_tuned", "Spatial TTA, tuned (8 views)"),
    ("mCA_equal", "Equal avg. (15 views)"),
    ("mCA_conf", "Confidence-only"),
    ("mCA_agree", "Agreement-only"),
    ("mCA_fused", "Final (fused)"),
    ("mCA_oracle", "Oracle (upper bound)"),
]
#: predictor -> its "X - orig" record in bootstrap_ci_<tag>_final.json
GAIN_KEYS = [
    ("mCA_view1", "mild LP (0.90) - orig"),
    ("spatial", "spatial TTA - orig"),
    ("spatial_tuned", "spatial TTA tuned - orig"),
    ("mCA_equal", "equal avg - orig"),
    ("mCA_conf", "confidence-only - orig"),
    ("mCA_agree", "agreement-only - orig"),
    ("mCA_fused", "fused - orig"),
]


def table_and_fig_baselines_oracle():
    """LaTeX table of every baseline's mCA (and the oracle) plus a figure of
    each baseline's bootstrapped gain over the original, combined profile.
    """
    js = {m: load_eval_json(m, "combined") for m in MODEL_ORDER}
    boot = load_bootstrap("combined")
    for m in MODEL_ORDER:
        if "spatial TTA - orig" not in boot.get(m, {}):
            raise SystemExit(f"bootstrap_ci for {m} has no baseline deltas - run scripts/bootstrap_ci.py")
        js[m]["spatial"] = js[m]["mCA_orig"] + boot[m]["spatial TTA - orig"]["point"]
    # tuned spatial TTA (phase spatialfit): its "fused - orig" record, if present
    p_sp = R / "bootstrap_ci_cache_spatial_lofo_final.json"
    sp = json.loads(p_sp.read_text()) if p_sp.exists() else {}
    keys = [k for k in BASELINE_KEYS if k[0] != "spatial_tuned" or sp]
    gain_keys = [k for k in GAIN_KEYS if k[0] != "spatial_tuned" or sp]
    for m in MODEL_ORDER:
        if sp:
            boot[m]["spatial TTA tuned - orig"] = sp[m]["fused - orig"]
            js[m]["spatial_tuned"] = js[m]["mCA_orig"] + sp[m]["fused - orig"]["point"]
    lines = [r"\begin{tabular}{l" + "r" * len(MODEL_ORDER) + "}", r"\toprule",
             "Variant & " + " & ".join(MODEL_LABEL[m] for m in MODEL_ORDER) + r" \\", r"\midrule"]
    for key, label in keys:
        lines.append(f"{label} & " + " & ".join(f"{100 * js[m][key]:.2f}" for m in MODEL_ORDER) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_table("table_baselines_oracle", lines)

    # The 1-2 pt differences between predictors are invisible on an absolute
    # 0-70% axis next to the oracle, so plot the paired gain over the original
    # with its cluster-bootstrap CI; the oracle stays in the table.
    labels = dict(BASELINE_KEYS)
    x = np.arange(len(gain_keys))
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    n, w = len(MODEL_ORDER), 0.8 / len(MODEL_ORDER)
    for i, m in enumerate(MODEL_ORDER):
        recs = [boot[m][k] for _, k in gain_keys]
        pts = np.array([100 * r["point"] for r in recs])
        lo = np.array([100 * r["ci95"][0] for r in recs])
        hi = np.array([100 * r["ci95"][1] for r in recs])
        ax.bar(x + (i - (n - 1) / 2) * w, pts, width=w * 0.92, color=MODEL_COLOR[m],
               label=MODEL_LABEL[m], zorder=3,
               yerr=[pts - lo, hi - pts], error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=1.5))
    zero_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k, _ in gain_keys], rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("Gain over original (pts)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, borderaxespad=0)
    clean_axes(ax)
    save(fig, "baselines_oracle_combined")
    print("  Caption note: mean corrupted accuracy minus the original's, test half, 95% paired "
          "cluster-bootstrap CI over images; the oracle (any of the 15 views right) is in the table only.")


# --------------------------------------------------------------------------- #
# Figure: corruption x severity heatmaps, combined profile
# --------------------------------------------------------------------------- #
def fig_heatmaps():
    """Figure per model: corruption x severity heatmap of the fused - original
    gain, combined profile, on one shared color scale.
    """
    mats = {}
    for m in MODEL_ORDER:
        df = load_eval_csv(m, "combined")
        mat = np.zeros((len(CORRUPTION_ORDER), 5))
        for i, c in enumerate(CORRUPTION_ORDER):
            for s in range(1, 6):
                row = df[(df["corruption"] == c) & (df["severity"] == s)]
                mat[i, s - 1] = 100 * (row["fused_acc"].values[0] - row["orig_acc"].values[0])
        mats[m] = mat
    vmax = max(np.abs(v).max() for v in mats.values())   # one shared scale across models
    for m, mat in mats.items():
        fig, ax = plt.subplots(figsize=(3.6, 5.2))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels([str(s) for s in range(1, 6)])
        ax.set_xlabel("Severity")
        ax.set_yticks(range(len(CORRUPTION_ORDER)))
        ax.set_yticklabels([c.replace("_", " ") for c in CORRUPTION_ORDER], fontsize=7)
        ax.set_title(MODEL_LABEL[m])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > vmax * 0.55 else INK)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Percentage points", fontsize=8)
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(CORRUPTION_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        save(fig, f"heatmap_{m}_combined")


# --------------------------------------------------------------------------- #
# Figure + table: learned per-view bias, global config, all three models
# --------------------------------------------------------------------------- #
def fig_view_bias():
    """Figure + LaTeX table of the learned per-view bias of the selected global
    config (models whose method has no bias are skipped).
    """
    biases, methods, omitted = {}, {}, {}
    for m in MODEL_ORDER:
        t = load_final_tuning(m, "combined")
        b = t["global"]["config"].get("view_bias") or []
        # methods without a learned bias (grid, CE, CE unconstrained) store
        # zeros or nothing; plotting them would only add empty bars
        if len(b) == len(VIEW_NAMES) and any(abs(x) > 0 for x in b):
            biases[m], methods[m] = b, t.get("method", "grid")
        else:
            omitted[m] = t.get("method", "grid")
    if not biases:
        print("  no selected method has a view bias -- skipped")
        return
    x = np.arange(len(VIEW_NAMES))
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    n, w = len(biases), 0.8 / len(biases)
    for i, m in enumerate(biases):
        ax.bar(x + (i - (n - 1) / 2) * w, biases[m], width=w * 0.92, color=MODEL_COLOR[m],
               label=f"{MODEL_LABEL[m]} ({METHOD_LABEL[methods[m]]})", zorder=3)
    zero_line(ax)
    ax.axvline(7.5, color=GRID, linewidth=1.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(VIEW_NAMES, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Learned mean-zero\nview bias")
    clean_axes(ax)
    top = ax.get_ylim()[1]
    ax.text(3.5, top * 0.95, "spectral ladder", ha="center", va="top", fontsize=7.5, color=INK2, style="italic")
    ax.text(11, top * 0.95, "spatial views", ha="center", va="top", fontsize=7.5, color=INK2, style="italic")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(biases),
              borderaxespad=0, fontsize=7)
    save(fig, "view_bias_combined")

    lines = [r"\begin{tabular}{cl" + "r" * len(biases) + "}", r"\toprule",
             "Idx & View & " + " & ".join(MODEL_LABEL[m] for m in biases) + r" \\", r"\midrule"]
    for i, name in enumerate(VIEW_NAMES):
        lines.append(f"{i} & {name} & " + " & ".join(f"{biases[m][i]:+.3f}" for m in biases) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_table("table_view_bias", lines)
    not_shown = "; ".join(f"{MODEL_LABEL[m]} is not shown: its selected method, {METHOD_LABEL[meth]}, "
                          f"has no per-view bias" for m, meth in omitted.items())
    print("  Caption note: learned per-view bias of the selected global config (combined views). A "
          "bias-only fit is a fixed per-view weighting (alpha = beta = gamma = 0), so these bars are the "
          "whole weighting." + (f" {not_shown}." if not_shown else ""))


# --------------------------------------------------------------------------- #
# Table: calibration, transfer, oracle, overhead (combined profile)
# --------------------------------------------------------------------------- #
def _overhead():
    """Per-model inference cost of the combined profile vs one forward pass,
    from inference_overhead.json; {} if not measured.
    """
    p = R / "inference_overhead.json"
    if not p.exists():
        return {}
    return {m: v["combined_x"] for m, v in json.loads(p.read_text())["models"].items()}


def table_diagnostics():
    """LaTeX table of NLL/ECE (vanilla, calibrated vanilla, final), images
    fixed/broken and inference overhead, combined profile.
    """
    ov = _overhead()
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             "Model & \\multicolumn{3}{c}{NLL} & \\multicolumn{3}{c}{ECE} & Fixed (\\% all) & "
             "Broken (\\% all) & Overhead \\\\",
             " & Van. & Van.\\ (cal.) & Final & Van. & Van.\\ (cal.) & Final & & & \\\\", r"\midrule"]
    for m in MODEL_ORDER:
        j = load_eval_json(m, "combined")
        df = load_eval_csv(m, "combined")
        o = f"{ov[m]:.1f}$\\times$" if m in ov else "--"
        lines.append(
            f"{MODEL_LABEL[m]} & {j['mean_orig_nll']:.3f} & {j['mean_orig_nll_calibrated']:.3f} & "
            f"{j['mean_fused_nll']:.3f} & {j['mean_orig_ece']:.3f} & {j['mean_orig_ece_calibrated']:.3f} & "
            f"{j['mean_fused_ece']:.3f} & " + "{:.2f} & {:.2f} & ".format(*fixed_broken(df))
            + f"{o} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_table("table_diagnostics", lines)
    print("  Caption note: 'Van. (cal.)' = vanilla with one temperature fit on the tune half, the "
          "fair comparison for the final predictor's own tuned output temperature. Fixed / broken = "
          "images fusion turns from wrong to right / right to wrong, as % of ALL images, so "
          "fixed - broken = the accuracy gain. (The per-subset rates in the CSVs - broken / "
          "originally right, fixed / originally wrong - have different denominators and are not "
          "comparable.)")


# --------------------------------------------------------------------------- #
# Table: final (global) configuration per model, combined profile
# --------------------------------------------------------------------------- #
def _cfg_cell(glob_v, fold_vs, fmt="{:.2f}"):
    """'global (min-max over the LOFO folds)', or one value if they all agree."""
    lo, hi = min(fold_vs), max(fold_vs)
    if np.isclose(lo, hi) and np.isclose(lo, glob_v):
        return fmt.format(glob_v)
    # "to" when a bound is negative: "-0.49--0.29" would typeset as an em-dash
    sep = " to " if lo < 0 or hi < 0 else "--"
    return f"{fmt.format(glob_v)} ({fmt.format(lo)}{sep}{fmt.format(hi)})"


def _set_cell(glob_v, fold_vs):
    """All distinct values among the global and fold values, joined by '/'."""
    vals = sorted({str(v) for v in fold_vs} | {str(glob_v)})
    return "/".join(vals)


def table_final_config():
    """Selected fusion hyper-parameters, every model x profile: the global
    (all-family) value, and in parentheses the range over the 4 leave-one-
    family-out configs that produced the headline test numbers."""
    lines = [r"\begin{tabular}{lllllllll}", r"\toprule",
             r"Model & Profile & Method & $\alpha$ & $\beta$ & $\gamma$ & $m$ & Agreement & $T_\mathrm{out}$ \\",
             r"\midrule"]
    for mi, m in enumerate(MODEL_ORDER):
        for k, p in enumerate(("spectral", "combined")):
            t = load_final_tuning(m, p)
            g = t["global"]["config"]
            folds = [e["config"] for e in t["held_out"].values()]
            col = lambda key: [f[key] for f in folds]
            beta_used = any(abs(v) > 1e-9 for v in col("beta") + [g["beta"]])
            gamma_used = any(abs(v) > 1e-9 for v in col("gamma") + [g["gamma"]])
            name = MODEL_LABEL[m] if k == 0 else ""
            lines.append(
                f"{name} & {PROFILE_LABEL[p]} & {METHOD_LABEL[t.get('method', 'grid')]} & "
                f"{_cfg_cell(g['alpha'], col('alpha'))} & {_cfg_cell(g['beta'], col('beta'))} & "
                f"{_cfg_cell(g['gamma'], col('gamma'))} & "
                f"{_set_cell(g['m'], col('m')) if gamma_used else '--'} & "
                f"{_set_cell(g['agreement'], col('agreement')) if beta_used else '--'} & "
                f"{_cfg_cell(g.get('output_temperature', 1.0), [f.get('output_temperature', 1.0) for f in folds])} \\\\")
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_table("table_final_config", lines)
    print("  Caption note: global (all-family) value, with the range over the 4 leave-one-family-out "
          "configs (which produce the headline numbers) in parentheses. tau = 1 throughout: the CE fit "
          "absorbs the softmax sharpness into alpha/beta/gamma. '--' = inactive (m needs gamma > 0, "
          "the agreement mode needs beta > 0). Bias-only fits a fixed per-view prior, alpha=beta=gamma=0.")


# --------------------------------------------------------------------------- #
# Table: per-corruption numbers, combined profile, all 15 corruptions
# --------------------------------------------------------------------------- #
def table_per_corruption():
    """LaTeX table of vanilla/final accuracy and their delta per corruption
    and model, combined profile.
    """
    dfs = {m: load_eval_csv(m, "combined") for m in MODEL_ORDER}
    lines = [r"\begin{tabular}{l" + "rrr" * len(MODEL_ORDER) + "}", r"\toprule",
             "Corruption & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{MODEL_LABEL[m]}}}"
                                          for m in MODEL_ORDER) + r" \\",
             " & " + " & ".join("Van.\\ & Final & $\\Delta$" for _ in MODEL_ORDER) + r" \\",
             r"\midrule"]
    for c in CORRUPTION_ORDER:
        cells = []
        for m in MODEL_ORDER:
            rows = dfs[m][dfs[m]["corruption"] == c]
            van, fin = 100 * rows["orig_acc"].mean(), 100 * rows["fused_acc"].mean()
            cells.append(f"{van:.2f} & {fin:.2f} & {fin - van:+.2f}")
        lines.append(f"{c.replace('_', ' ')} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    text = "\n".join(lines)
    (OUT / "table_per_corruption.tex").write_text(text)
    print("  saved table_per_corruption.tex")


# --------------------------------------------------------------------------- #
# Figure: the Gaussian low-pass ladder itself (pure math)
# --------------------------------------------------------------------------- #
def fig_lowpass_ladder():
    """Figure: transmission curves of the 7 Gaussian low-pass masks."""
    scales = [0.90, 0.75, 0.60, 0.45, 0.30, 0.20, 0.12]
    r = np.linspace(0, 1.4, 400)
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    cmap = plt.get_cmap("viridis")
    for i, q in enumerate(scales):
        ax.plot(r, np.exp(-(r ** 2) / (2 * q ** 2)), color=cmap(i / (len(scales) - 1)),
                linewidth=1.3, label=f"q={q:.2f}")
    ax.set_xlabel("Normalized radial frequency")
    ax.set_ylabel("Mask transmission")
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.4, labelspacing=0.3)
    clean_axes(ax, xgrid=True)
    save(fig, "lowpass_ladder")


# --------------------------------------------------------------------------- #
# Figure: per-view accuracy down the low-pass ladder, by family and model.
# The mechanism figure: low-pass views help only where the model is fragile
# to the corruption's high-frequency content.
# --------------------------------------------------------------------------- #
def per_view_accuracy():
    """Mean per-view top-1 accuracy per (model, family), all images, sev 1-5,
    from the spectral cache; cached to results/per_view_accuracy.json."""
    out = R / "per_view_accuracy.json"
    if out.exists():
        return json.loads(out.read_text())
    from src import cache as cache_mod
    cache_mod.CACHE_ROOT = SPECTRAL_CACHE
    res = {}
    for m in MODEL_ORDER:
        acc = {f: [] for f in FAMILIES}
        for c in CORRUPTION_ORDER:
            for s in range(1, 6):
                if not cache_mod.has_condition(m, c, s):
                    continue
                d = cache_mod.load_condition(m, c, s)
                hit = d["probs"].argmax(-1) == d["labels"][:, None]
                acc[CORRUPTION_FAMILY[c]].append(hit.mean(0).tolist())
        res[m] = {f: np.mean(v, axis=0).tolist() for f, v in acc.items() if v}
    out.write_text(json.dumps(res, indent=2))
    return res


def fig_per_view_accuracy():
    """Figure: per-view accuracy change vs the original down the low-pass
    ladder, one panel per family.
    """
    res = per_view_accuracy()
    fams = list(FAMILIES)
    fig, axes = plt.subplots(1, 4, figsize=(7.8, 2.5), sharey=False)
    x = np.arange(len(LADDER_LABELS))
    for ax, f in zip(axes, fams):
        for m in MODEL_ORDER:
            # change vs. the unfiltered view, so the three models share one scale
            v = 100 * (np.asarray(res[m][f]) - res[m][f][0])
            ax.plot(x, v, marker="o", markersize=3, linewidth=1.4, color=MODEL_COLOR[m],
                    label=MODEL_LABEL[m])
        zero_line(ax)
        ax.set_title(f.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(LADDER_LABELS, rotation=60, fontsize=6.5)
        clean_axes(ax, xgrid=True)
    axes[0].set_ylabel("Accuracy vs. original (pts)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=6.5)
    fig.supxlabel("Low-pass cutoff (view)", fontsize=8)
    fig.tight_layout()
    save(fig, "per_view_accuracy")
    for m in MODEL_ORDER:
        best = {f: 100 * (max(res[m][f][1:]) - res[m][f][0]) for f in fams}
        print(f"    {m:9s} best filtered view - original: "
              + "  ".join(f"{f} {v:+.1f}" for f, v in best.items()))


# --------------------------------------------------------------------------- #
# Figure: remaining oracle headroom by corruption, combined profile
# --------------------------------------------------------------------------- #
def fig_oracle_headroom():
    """Figure: oracle - fused accuracy per corruption, combined profile."""
    head = {}
    for m in MODEL_ORDER:
        df = load_eval_csv(m, "combined")
        head[m] = [100 * (df[df["corruption"] == c]["oracle_acc"] - df[df["corruption"] == c]["fused_acc"]).mean()
                   for c in CORRUPTION_ORDER]
    x = np.arange(len(CORRUPTION_ORDER))
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    grouped_bars(ax, x, head)
    family_separators(ax, labels=False)
    ax.set_xticks(x)
    ax.set_xticklabels([CORR_LABEL[c] for c in CORRUPTION_ORDER], fontsize=6.8)
    ax.set_ylabel("Oracle headroom (pts)")
    clean_axes(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, borderaxespad=0)
    ax.text(0.5, -0.32, "Oracle = per-image upper bound over the 15 cached views; not a realizable method.",
            transform=ax.transAxes, ha="center", fontsize=6.5, color=INK2, style="italic")
    save(fig, "oracle_headroom_combined")


# --------------------------------------------------------------------------- #
# Figure: positive vs negative transfer per corruption, combined profile
# --------------------------------------------------------------------------- #
def fig_transfer_scatter():
    """Figure: images fixed vs broken by fusion per corruption (% of all
    images), combined profile.
    """
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    for m in MODEL_ORDER:
        df = load_eval_csv(m, "combined")
        # Both rates as a share of ALL images (the raw rates divide by
        # different denominators: broken / originally right, fixed /
        # originally wrong), so the diagonal is break-even and the height
        # above it is the net gain.
        neg = [100 * (d["negative_transfer"] * d["orig_acc"]).mean()
               for c in CORRUPTION_ORDER for d in [df[df["corruption"] == c]]]
        pos = [100 * (d["positive_transfer"] * (1 - d["orig_acc"])).mean()
               for c in CORRUPTION_ORDER for d in [df[df["corruption"] == c]]]
        ax.scatter(neg, pos, color=MODEL_COLOR[m], label=MODEL_LABEL[m], s=22, zorder=3,
                   edgecolor="white", linewidth=0.6)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 1)
    ax.plot([0, lim], [0, lim], linestyle="--", color=ZERO_LINE, linewidth=0.8, zorder=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Images broken (% of all)")
    ax.set_ylabel("Images fixed (% of all)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, borderaxespad=0, fontsize=7, handletextpad=0.2, columnspacing=0.8)
    clean_axes(ax, xgrid=True)
    save(fig, "transfer_scatter_combined")
    print("  Caption note: one point per corruption (mean over severities); both axes are % of all "
          "test images, so the diagonal is break-even and the height above it is the net gain.")


# --------------------------------------------------------------------------- #
# Spectral energy (scripts/spectral_energy.py): fraction of the
# corruption's energy that a low-pass view removes vs. that view's gain
# --------------------------------------------------------------------------- #
FAMILY_MARKER = {"noise": "o", "blur": "s", "weather": "^", "digital": "D"}
ENERGY_CUTOFF = 0.75


def fig_spectral_energy():
    """Gain of the LP view vs. two spectral measures of the corruption: rows =
    "energy removed" / "energy added", columns = models. Filled markers: the
    15 corruptions (rho in the panel title); hollow: the 4 unseen ones."""
    p = R / "spectral_energy.json"
    if not p.exists():
        print("  results/spectral_energy.json missing -- run scripts/spectral_energy.py")
        return
    rec = json.loads(p.read_text())
    px = R / "spectral_energy_extra.json"
    extra = json.loads(px.read_text()) if px.exists() else None
    qi = rec["cutoffs"].index(ENERGY_CUTOFF)
    measures = [("band_frac", "Corruption energy removed"), ("band_added", "Energy added (exploratory)")]
    measures = [mk for mk in measures if mk[0] in rec["rows"][0]]
    fig, axes = plt.subplots(len(measures), len(MODEL_ORDER), figsize=(7.8, 2.5 * len(measures)),
                             sharex="row", squeeze=False)
    for ri, (key, label) in enumerate(measures):
        for ci, m in enumerate(MODEL_ORDER):
            ax = axes[ri, ci]
            for fam, mk in FAMILY_MARKER.items():
                rows = [r for r in rec["rows"] if r["family"] == fam]
                ax.scatter([r[key][qi] for r in rows], [100 * r["view_gain"][m][qi] for r in rows],
                           marker=mk, s=18, color=MODEL_COLOR[m], edgecolor="white", linewidth=0.5, zorder=3)
                if extra:
                    rows = [r for r in extra["rows"] if r["family"] == fam]
                    ax.scatter([r[key][qi] for r in rows], [100 * r["view_gain"][m][qi] for r in rows],
                               marker=mk, s=20, facecolor="none", edgecolor=MODEL_COLOR[m], linewidth=0.9,
                               zorder=4)
            rho = next(d for d in rec["spearman"][m][key] if d["cutoff"] == ENERGY_CUTOFF)["spearman"]
            zero_line(ax)
            ax.set_title(f"{MODEL_LABEL[m]}  ($\\rho$ = {rho:+.2f})", fontsize=8)
            ax.set_xlabel(f"{label}, LP {ENERGY_CUTOFF}", fontsize=7.5)
            clean_axes(ax, xgrid=True)
        axes[ri, 0].set_ylabel(f"Gain of LP {ENERGY_CUTOFF} view (pts)")
    handles = [plt.Line2D([], [], marker=mk, linestyle="", color=INK2, label=f) for f, mk in FAMILY_MARKER.items()]
    if extra:
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                                  markeredgecolor=INK2, label="unseen corruption"))
    fig.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles))
    fig.tight_layout()
    save(fig, "spectral_energy_gain")
    print("  Caption note: one point per corruption x severity; filled = the 15 standard corruptions "
          "(rho = Spearman over their 75 conditions), hollow = the 4 unseen corruptions (shape = the "
          "family they are scored with). 'Energy added' was defined after the other measures were "
          "seen; exploratory.")


SPECTRAL_STATS = (("band_frac", "corruption energy removed"), ("band_shift", "image energy change"),
                  ("band_added", "energy added"))


def table_spectral_energy():
    """LaTeX table of Spearman correlations between each spectral measure and
    the LP view's gain (skipped if spectral_energy.json is missing).
    """
    p = R / "spectral_energy.json"
    if not p.exists():
        return
    rec = json.loads(p.read_text())
    px = R / "spectral_energy_extra.json"
    extra = json.loads(px.read_text()) if px.exists() else None

    def at(r, m, key, field="spearman"):
        """The `field` entry at ENERGY_CUTOFF of r['spearman'][m][key] (NaN if absent)."""
        return next(d for d in r["spearman"][m][key] if d["cutoff"] == ENERGY_CUTOFF).get(field, float("nan"))

    cols = ["15 corruptions", "within family"] + (["4 unseen"] if extra else [])
    lines = [r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
             "Model & Measure & " + " & ".join(cols) + r" \\", r"\midrule"]
    for m in MODEL_ORDER:
        for k, (key, label) in enumerate(SPECTRAL_STATS):
            if key not in rec["spearman"][m]:
                continue
            cells = [at(rec, m, key), at(rec, m, key, "within_family")]
            if extra:
                cells.append(at(extra, m, key))
            name = MODEL_LABEL[m] if k == 0 else ""
            lines.append(f"{name} & {label} & " + " & ".join(f"{c:+.2f}" for c in cells) + r" \\")
        if m != MODEL_ORDER[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_table("table_spectral_energy", lines)
    print(f"  Caption note: Spearman rank correlation between each measure at LP {ENERGY_CUTOFF} and that "
          "view's accuracy gain over the original: across the 75 conditions of the 15 corruptions, within "
          "each corruption family (mean over families), and across the 20 conditions of the 4 unseen "
          "corruptions. Other cutoffs: results/spectral_energy*.json.")


# --------------------------------------------------------------------------- #
# Profile shape (scripts/profile_shape.py): how far each model's prediction
# drifts from the original as more high frequencies are removed
# --------------------------------------------------------------------------- #
def fig_profile_drift():
    """Figure: mean JSD of each low-pass view to the original, per family."""
    p = R / "profile_shape.json"
    if not p.exists():
        print("  results/profile_shape.json missing -- run scripts/profile_shape.py")
        return
    res = json.loads(p.read_text())
    panels = list(FAMILIES) + ["clean"]
    fig, axes = plt.subplots(1, len(panels), figsize=(8.6, 2.5), sharey=True)
    x = np.arange(len(LADDER_LABELS))
    for ax, f in zip(axes, panels):
        for m in MODEL_ORDER:
            ax.plot(x, res[m][f]["jsd"], marker="o", markersize=3, linewidth=1.4,
                    color=MODEL_COLOR[m], label=MODEL_LABEL[m])
        ax.set_title(f.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(LADDER_LABELS, rotation=60, fontsize=6.5)
        clean_axes(ax, xgrid=True)
    axes[0].set_ylabel("JSD to original view")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.supxlabel("Low-pass cutoff (view)", fontsize=8)
    fig.tight_layout()
    save(fig, "profile_drift")


# --------------------------------------------------------------------------- #
# Table: held-out extra corruptions (scripts/evaluate_extra.py), optional
# --------------------------------------------------------------------------- #
EXTRA_ORDER = ["speckle_noise", "gaussian_blur", "spatter", "saturate", "all"]
EXTRA_DELTAS = [("spectral - orig", "Spectral $-$ orig."), ("combined - orig", "Combined $-$ orig."),
                ("spectral - spatial TTA", "Spectral $-$ spatial TTA")]


def table_extra():
    """LaTeX table of the bootstrapped deltas on the 4 extra (unseen)
    corruptions; skipped if that phase was not run.
    """
    p = R / "extra_corruptions.json"
    if not p.exists():
        print("  no extra_corruptions.json (optional phase 'extra') -- skipped")
        return
    rec = json.loads(p.read_text())["models"]
    lines = [r"\begin{tabular}{ll" + "r" * len(MODEL_ORDER) + "}", r"\toprule",
             "Corruption & $\\Delta$ & " + " & ".join(MODEL_LABEL[m] for m in MODEL_ORDER) + r" \\",
             r"\midrule"]
    for c in EXTRA_ORDER:
        name = "Mean of 4" if c == "all" else c.replace("_", " ").capitalize()
        for k, (key, label) in enumerate(EXTRA_DELTAS):
            cells = []
            for m in MODEL_ORDER:
                d = rec[m]["per_corruption"][c]["deltas"][key]
                cells.append(fmt_ci(d["point"], d["ci95"]))
            lines.append(f"{name if k == 0 else ''} & {label} & " + " & ".join(cells) + r" \\")
        if c != EXTRA_ORDER[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_table("table_extra_corruptions", lines)
    print("  Caption note: ImageNet-C's 4 held-out extra corruptions, never seen by tuning or selection; "
          "frozen selected configs, each corruption scored with the LOFO config of its family; test half, "
          "5 severities; 95% paired cluster-bootstrap CIs.")


# --------------------------------------------------------------------------- #
# Tables: other view families and signals (phase 'views'), optional
# --------------------------------------------------------------------------- #
GROUP_ROWS = [("noise", "Noise"), ("blur", "Blur"), ("weather", "Weather"), ("digital", "Digital"),
              ("all corruptions", "All corruptions"), ("clean", "Clean")]


def _family_table(path, families, labels, name, caption):
    """Write a LaTeX table of test-half gains per image group x view family
    from `path`; skipped if the file is missing. Returns nothing.
    """
    if not path.exists():
        print(f"  no {path.name} (optional phase 'views') -- skipped")
        return
    rec = json.loads(path.read_text())["models"]
    lines = [r"\begin{tabular}{ll" + "r" * len(MODEL_ORDER) + "}", r"\toprule",
             "Images & View family & " + " & ".join(MODEL_LABEL[m] for m in MODEL_ORDER) + r" \\",
             r"\midrule"]
    for g, glabel in GROUP_ROWS:
        for k, fam in enumerate(families):
            cells = [fmt_ci(rec[m][g][fam]["test_gain"]["point"], rec[m][g][fam]["test_gain"]["ci95"])
                     for m in MODEL_ORDER]
            lines.append(f"{glabel if k == 0 else ''} & {labels[fam]} & " + " & ".join(cells) + r" \\")
        if g != GROUP_ROWS[-1][0]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_table(name, lines)
    print("  Caption note: " + caption)


def table_view_families():
    """LaTeX table comparing low-pass, band-pass and notch single views."""
    _family_table(R / "view_family_comparison.json", ["lowpass", "bandpass", "notch"],
                  {"lowpass": "Low-pass", "bandpass": "Band-pass", "notch": "Notch"},
                  "table_view_families",
                  "single views, severity 3; per view family and image group the best setting is chosen "
                  "on the tune half; test-half gain over the original view, 95% paired cluster-bootstrap CI.")


def table_ladders():
    """LaTeX table comparing the fixed and energy-adaptive low-pass ladders."""
    _family_table(R / "ladder_comparison.json", ["lowpass", "energy"],
                  {"lowpass": "Fixed cutoffs", "energy": "Energy-adaptive"},
                  "table_ladders",
                  "single views, severity 3; best ladder step chosen on the tune half; test-half gain over "
                  "the original view, 95% paired cluster-bootstrap CI.")


def table_signals():
    """LaTeX tables of the 4th-signal (ambiguity) experiment and of the
    correlation of learned per-view weights with view accuracy/contribution.
    """
    amb = R / "ambiguity_experiment.json"
    if amb.exists():
        rec = json.loads(amb.read_text())
        lines = [r"\begin{tabular}{lrrr}", r"\toprule",
                 r"Model & 3 signals + bias & 4 signals + bias & Difference \\", r"\midrule"]
        for m in MODEL_ORDER:
            r = rec[m]
            d = r["four_minus_three"]
            lines.append(f"{MODEL_LABEL[m]} & {100 * r['test_mca_3signal']:.2f} & {100 * r['test_mca_4signal']:.2f} & "
                         f"{fmt_ci(d['point'], d['ci95'])} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        _write_table("table_fourth_signal", lines)
    rows = []
    for meth, label in (("ce_bias_only", "Bias only"), ("ce_bias", "CE + bias")):
        p = R / f"view_bias_diagnostic_{meth}.json"
        if p.exists():
            rec = json.loads(p.read_text())["models"]
            for key, klabel in (("view_acc", "view's own accuracy"), ("loo_contribution", "contribution to the ensemble")):
                rows.append(f"{label} & {klabel} & " + " & ".join(
                    f"{rec[m]['corr'][key]['pearson']:+.2f}" for m in MODEL_ORDER) + r" \\")
    if rows:
        lines = [r"\begin{tabular}{ll" + "r" * len(MODEL_ORDER) + "}", r"\toprule",
                 "Fit & Correlation of the per-view weight with & "
                 + " & ".join(MODEL_LABEL[m] for m in MODEL_ORDER) + r" \\", r"\midrule"] + rows
        lines += [r"\bottomrule", r"\end{tabular}"]
        _write_table("table_view_weight_correlation", lines)


if __name__ == "__main__":
    steps = [
        ("family_gain", fig_family_gain),
        ("severity_trend", fig_severity_trend),
        ("main_results_table", table_main_results),
        ("spatial_tuned_table", table_spatial_tuned),
        ("profiles_table", table_profiles),
        ("signal_ablation_table", table_signal_ablation),
        ("per_corruption_gain", fig_per_corruption_gain),
        ("baselines_oracle", table_and_fig_baselines_oracle),
        ("heatmaps", fig_heatmaps),
        ("view_bias", fig_view_bias),
        ("diagnostics", table_diagnostics),
        ("final_config", table_final_config),
        ("per_corruption_table", table_per_corruption),
        ("lowpass_ladder", fig_lowpass_ladder),
        ("per_view_accuracy", fig_per_view_accuracy),
        ("oracle_headroom", fig_oracle_headroom),
        ("transfer_scatter", fig_transfer_scatter),
        ("spectral_energy", fig_spectral_energy),
        ("spectral_energy_table", table_spectral_energy),
        ("profile_drift", fig_profile_drift),
        ("extra_table", table_extra),
        ("view_families_table", table_view_families),
        ("ladders_table", table_ladders),
        ("signals_tables", table_signals),
    ]
    for name, fn in steps:
        print(f"== {name} ==")
        fn()
