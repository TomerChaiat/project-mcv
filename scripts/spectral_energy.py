"""Does a corruption's spectrum predict whether removing a frequency band helps?

For every (corruption, severity) and every ladder cutoff q, on test-half
images paired with their exact clean originals (data/clean_name_map.json),
in the fftshifted Fourier domain with the ladder's own Gaussian mask M_q:

  band_frac  = sum (1-M_q^2) |F(x_c - x)|^2 / sum |F(x_c - x)|^2
               fraction of the corruption's energy (corrupted minus clean)
               that the low-pass view removes (the "spectral energy of the
               corruption", as in Yin et al.)
  band_shift = log( sum (1-M_q^2) |F(x_c)|^2 / sum (1-M_q^2) |F(x)|^2 )
               how much the corruption changed the image's energy in the
               removed band: > 0 the corruption ADDED energy there (noise),
               < 0 it REMOVED energy there (blur)
  band_added = log(1 + sum (1-M_q^2) max(0, |F(x_c)|^2 - |F(x)|^2)
                       / sum (1-M_q^2) |F(x)|^2 )
               energy the corruption added in the removed band, frequency by
               frequency, relative to the clean image's energy there: counts
               energy the corruption adds (e.g. noise grain) but not detail
               it removes

and, from the spectral cache, the accuracy gain of that low-pass view over
the original on the same test half, per model. Writes
results/spectral_energy.json; plotted by scripts/make_paper_figures.py.

    python scripts/spectral_energy.py --n-per-condition 500
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import spearmanr

from src import cache
from src.cache import FIXED_CUTOFFS, load_image
from src.filters import radial_lowpass_mask
from src.splits import ALL_CORRUPTIONS, CORRUPTION_FAMILY, EXTRA_FAMILY, tune_test_masks

MODELS = ["resnet18", "deit", "convnext"]


def power(x):
    """(B,3,H,W) -> fftshifted power spectrum summed over channels, (B,H,W)."""
    f = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    return (f.abs() ** 2).sum(1)


def correlations(rows):
    """Spearman correlation, across the 75 (corruption, severity) conditions,
    between each spectral statistic at cutoff q and the gain of the view at
    that same cutoff -- per model and per cutoff. (Pooling over cutoffs would
    mostly measure 'strong cutoffs hurt'.) 'within_family' is the mean of the
    Spearman correlations computed inside each corruption family, i.e. with
    the differences between families removed."""
    summary = {}
    keys = [k for k in ("band_frac", "band_shift", "band_added") if k in rows[0]]
    families = sorted({r["family"] for r in rows})
    for m in MODELS:
        summary[m] = {}
        for key in keys:
            per_q = []
            for qi, q in enumerate(FIXED_CUTOFFS):
                x = [r[key][qi] for r in rows]
                y = [r["view_gain"][m][qi] for r in rows]
                rho, p = spearmanr(x, y)
                within = [spearmanr([r[key][qi] for r in rows if r["family"] == f],
                                    [r["view_gain"][m][qi] for r in rows if r["family"] == f])[0]
                          for f in families]
                within = [w for w in within if np.isfinite(w)]
                per_q.append({"cutoff": q, "spearman": float(rho), "p": float(p), "n": len(x),
                              "within_family": float(np.mean(within)) if within else float("nan")})
            summary[m][key] = per_q
            print(f"  {m:9s} {key:10s} " + " ".join(
                f"{d['cutoff']}:{d['spearman']:+.2f}({d['within_family']:+.2f})" for d in per_q))
    return summary


def main():
    """Measure how much corruption energy each low-pass view removes, pair it
    with the view's accuracy gain per model, and save rows + Spearman correlations.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-condition", type=int, default=500,
                    help="test-half images per (corruption, severity) for the spectra")
    ap.add_argument("--c-root", default="data/imagenet-c")
    ap.add_argument("--clean-root", default="data/imagenet/val")
    ap.add_argument("--name-map", default="data/clean_name_map.json")
    ap.add_argument("--spectral-cache", default="cache_fixed")
    ap.add_argument("--out", default="results/spectral_energy.json")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--corruptions", nargs="+", default=ALL_CORRUPTIONS,
                    help="default: the 15 standard ones; the extras are in src.splits.EXTRA_FAMILY")
    ap.add_argument("--summary-only", action="store_true",
                    help="recompute the correlations from an existing --out file")
    args = ap.parse_args()

    if args.summary_only:
        rec = json.loads(Path(args.out).read_text())
        rec["spearman"] = correlations(rec["rows"])
        Path(args.out).write_text(json.dumps(rec, indent=1))
        print("updated", args.out)
        return

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inv = {v: k for k, v in json.loads(Path(args.name_map).read_text()).items()}
    masks = torch.stack([radial_lowpass_mask(224, 224, q, device=dev) for q in FIXED_CUTOFFS])
    removed = 1.0 - masks ** 2                                    # (Q,H,W) energy removed by view q
    cache.CACHE_ROOT = Path(args.spectral_cache)
    rng = np.random.default_rng(0)

    rows = []
    family = {**CORRUPTION_FAMILY, **EXTRA_FAMILY}
    for c in args.corruptions:
        for s in range(1, 6):
            # image list and test half from the cache, so spectra and accuracies use the same images
            d0 = cache.load_condition(MODELS[0], c, s)
            _, test = tune_test_masks(d0["paths"])
            test_paths = np.asarray(d0["paths"])[test]
            pick = rng.choice(len(test_paths), size=min(args.n_per_condition, len(test_paths)), replace=False)
            frac_sum = np.zeros(len(FIXED_CUTOFFS))
            logshift_sum = np.zeros(len(FIXED_CUTOFFS))
            added_sum = np.zeros(len(FIXED_CUTOFFS))
            n = 0
            for lo in range(0, len(pick), args.batch):
                ps = [Path(str(p)) for p in test_paths[pick[lo:lo + args.batch]]]
                xc = torch.stack([load_image(p) for p in ps]).to(dev)
                x = torch.stack([load_image(Path(args.clean_root) / p.parent.name / f"{inv[p.stem]}.JPEG")
                                 for p in ps]).to(dev)
                pd, pc, p0 = power(xc - x), power(xc), power(x)            # (B,H,W)
                band_d = torch.einsum("bhw,qhw->bq", pd, removed)
                frac = band_d / pd.sum((1, 2))[:, None]
                shift = torch.log(torch.einsum("bhw,qhw->bq", pc, removed)
                                  / torch.einsum("bhw,qhw->bq", p0, removed))
                frac_sum += frac.sum(0).cpu().double().numpy()
                logshift_sum += shift.sum(0).cpu().double().numpy()
                added = torch.log1p(torch.einsum("bhw,qhw->bq", (pc - p0).clamp(min=0), removed)
                                    / torch.einsum("bhw,qhw->bq", p0, removed))
                added_sum += added.sum(0).cpu().double().numpy()
                n += len(ps)

            row = {"corruption": c, "family": family[c], "severity": s, "n_images": n,
                   "band_frac": (frac_sum / n).tolist(), "band_shift": (logshift_sum / n).tolist(),
                   "band_added": (added_sum / n).tolist(),
                   "view_gain": {}}
            for m in MODELS:
                d = cache.load_condition(m, c, s)
                _, t = tune_test_masks(d["paths"])
                hit = d["probs"][t].argmax(-1) == d["labels"][t][:, None]
                acc = hit.mean(0)
                row["view_gain"][m] = (acc[1:] - acc[0]).tolist()            # per cutoff, vs original
            rows.append(row)
            print(f"  {c:18s}/{s}: band_frac@0.45 {row['band_frac'][3]:.3f}  "
                  f"band_shift@0.45 {row['band_shift'][3]:+.3f}  "
                  + "  ".join(f"{m} {100 * row['view_gain'][m][3]:+.1f}" for m in MODELS))

    summary = correlations(rows)
    out = Path(args.out)
    out.write_text(json.dumps({"cutoffs": list(FIXED_CUTOFFS), "rows": rows, "spearman": summary}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
