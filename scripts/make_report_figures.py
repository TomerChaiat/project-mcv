"""Report-grade figures (300 DPI PNGs) from the numbers already in results/ -
no GPU, no re-computation, just plotting what evaluate.py / bootstrap_ci.py /
the per-view diagnostic already produced. Numbers are hard-coded from the
specific results/*.json files named in each function's docstring rather than
re-derived, so the figures always match what's cited in README.md; re-run the
underlying scripts and update the numbers here if the caches change.

    python scripts/make_report_figures.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# dataviz skill categorical palette (light mode), fixed order, high-contrast subset
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GREEN = "#008300"
YELLOW = "#eda100"
RED = "#e34948"
VIOLET = "#4a3aa7"
GREY = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def fig_headline():
    """Fig 1: does combining view families beat either alone?

    Source: results/eval_{resnet18,deit}_cache_{fixed,combined}_20260913/14*.json
    (mCA_orig / mCA_fused), results/tta_{resnet18,deit}_20260913_221531.json
    (mCA_tta), results/bootstrap_ci_cache_{fixed,combined}.json (95% CIs on
    fused - orig and fused - spatial TTA).
    """
    models = ["ResNet-18", "DeiT-Tiny"]
    methods = ["Original", "Spectral-only\nfusion", "Spatial TTA\n(crop+flip)", "Combined\nfusion"]
    colors = [GREY, ORANGE, AQUA, GREEN]

    # point estimates, mean-corrupted accuracy, %
    vals = {
        "ResNet-18": [32.56, 33.28, 34.44, 34.91],
        "DeiT-Tiny": [43.97, 44.17, 45.55, 45.95],
    }
    oracle = {"ResNet-18": 41.02, "DeiT-Tiny": 52.40}
    # 95% CI half-widths for the "Combined fusion" bar, from the paired
    # bootstrap of (combined fused - orig): resnet [2.24,2.47] -> center 2.35,
    # deit [1.87,2.09] -> center 1.98; half-width applied around the point est.
    combined_err = {"ResNet-18": (2.47 - 2.24) / 2, "DeiT-Tiny": (2.09 - 1.87) / 2}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=False)
    for ax, model in zip(axes, models):
        v = vals[model]
        x = np.arange(len(methods))
        bars = ax.bar(x, v, color=colors, width=0.62, zorder=3)
        # error bar only on the combined bar (the one we bootstrap-verified)
        ax.errorbar(x[3], v[3], yerr=combined_err[model], fmt="none",
                    ecolor=INK, elinewidth=1.2, capsize=4, zorder=4)
        ax.axhline(oracle[model], color=VIOLET, linestyle=(0, (4, 2)), linewidth=1.3, zorder=2)
        ax.text(len(methods) - 0.4, oracle[model] + 0.6, f"oracle {oracle[model]:.1f}%",
                color=VIOLET, fontsize=8.5, ha="right")
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.5, f"{vi:.1f}", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8.5)
        ax.set_ylim(0, max(oracle[model], max(v)) + 5)
        ax.set_title(model, fontsize=10.5, pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(length=0)
    axes[0].set_ylabel("mean corrupted accuracy (%)")
    fig.suptitle("Combining spectral and spatial view families beats either alone",
                 fontsize=11.5, y=1.02)
    fig.text(0.5, -0.04,
             "15 corruptions x 5 severities, held-out half. Error bar: 95% paired-bootstrap CI\n"
             "on combined-fusion - original. Oracle: per-image upper bound (any correct view).",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig_headline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_per_view():
    """Fig 2: why the spectral ladder alone can't beat spatial TTA.

    Source: per-family, per-view accuracy on the held-out half, fixed ladder
    (cache_fixed), computed directly from cache -- see the "How we got here"
    section of README.md for the ResNet-18 numbers; DeiT numbers computed the
    same way (not separately saved to results/, reproducible via the snippet
    in that section with model='deit').
    """
    cutoffs = [0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.12]
    data = {
        "ResNet-18": {
            "noise":   [22.9, 24.4, 24.9, 25.6, 26.3, 27.0, 26.1, 22.0],
            "blur":    [28.3, 28.3, 28.2, 28.1, 27.8, 26.8, 24.7, 18.2],
            "weather": [36.1, 35.5, 35.2, 34.6, 33.3, 29.9, 24.8, 15.1],
            "digital": [40.5, 41.0, 41.0, 40.9, 40.7, 39.0, 34.8, 23.9],
        },
        "DeiT-Tiny": {
            "noise":   [43.61, 44.43, 44.43, 44.49, 44.29, 43.17, 38.50, 28.82],
            "blur":    [35.11, 34.45, 34.24, 33.83, 33.26, 32.10, 30.36, 25.27],
            "weather": [48.51, 47.42, 47.03, 46.32, 44.73, 40.81, 34.24, 22.41],
            "digital": [48.54, 48.55, 48.65, 48.61, 48.43, 46.95, 42.22, 31.42],
        },
    }
    family_color = {"noise": BLUE, "blur": ORANGE, "weather": RED, "digital": YELLOW}
    x = np.arange(8)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
    for ax, model in zip(axes, ["ResNet-18", "DeiT-Tiny"]):
        for fam, ys in data[model].items():
            ax.plot(x, ys, color=family_color[fam], linewidth=2, marker="o",
                    markersize=3.2, zorder=3, label=fam)
            ax.text(x[-1] + 0.15, ys[-1], fam, color=family_color[fam],
                    fontsize=8, va="center")
        ax.set_xticks(x)
        ax.set_xticklabels(["orig"] + [f"{c:.2f}" for c in cutoffs[1:]],
                           rotation=45, ha="right", fontsize=7.5)
        ax.set_xlim(-0.3, 9.3)
        ax.set_title(model, fontsize=10.5, pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(length=0)
    axes[0].set_ylabel("accuracy (%)")
    fig.text(0.5, 0.02, "view (filter cutoff; smaller = stronger low-pass)",
             ha="center", fontsize=9, color=INK2)
    fig.suptitle("Only noise ever benefits from deeper low-pass filtering",
                 fontsize=11.5, y=1.03)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "fig_per_view.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_baselines():
    """Fig 3: the full baseline picture, one model per panel.

    Source: results/eval_{resnet18,deit}_cache_fixed_20260913_234358.json
    (all spectral-only baselines + oracle), results/tta_*_20260913_221531.json
    (spatial TTA), results/eval_*_cache_combined_20260914_023918.json
    (combined fusion).
    """
    rows = ["Original", "Single mild\nview (no fusion)", "Equal average",
            "Confidence-\nonly", "Agreement-\nonly", "Spectral\nfusion (tuned)",
            "Spatial TTA", "Combined\nfusion (tuned)"]
    data = {
        "ResNet-18": [32.56, 32.82, 33.28, 33.34, 32.66, 33.28, 34.44, 34.91],
        "DeiT-Tiny": [43.97, 43.66, 43.65, 43.47, 43.90, 44.17, 45.55, 45.95],
    }
    oracle = {"ResNet-18": 41.02, "DeiT-Tiny": 52.40}
    # grey for plain baselines, colored for the three headline methods
    colors = [GREY, GREY, GREY, GREY, GREY, ORANGE, AQUA, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    y = np.arange(len(rows))
    for ax, model in zip(axes, ["ResNet-18", "DeiT-Tiny"]):
        v = data[model]
        ax.barh(y, v, color=colors, height=0.62, zorder=3)
        ax.axvline(oracle[model], color=VIOLET, linestyle=(0, (4, 2)), linewidth=1.3, zorder=2)
        ax.text(oracle[model] + 0.5, len(rows) - 0.3, f"oracle {oracle[model]:.1f}%",
                color=VIOLET, fontsize=8, rotation=90, va="top")
        for yi, vi in zip(y, v):
            ax.text(vi + 0.4, yi, f"{vi:.1f}", va="center", fontsize=8, color=INK)
        ax.set_title(model, fontsize=10.5, pad=8)
        ax.set_xlim(0, oracle[model] + 6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(rows, fontsize=8.5)
    axes[0].invert_yaxis()
    fig.text(0.5, -0.02, "mean corrupted accuracy (%)  -  grey: same-family baselines, "
             "colour: headline methods", ha="center", fontsize=8, color=MUTED)
    fig.suptitle("Every baseline the proposal committed to, both architectures",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_baselines.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_headline()
    fig_per_view()
    fig_baselines()
    print(f"saved figures to {OUT}/: fig_headline.png, fig_per_view.png, fig_baselines.png")
