"""Merge the spectral (cache_fixed) and spatial-TTA (cache_spatial) profiles
into one 15-view combined profile, per (model, corruption, severity):

    view  0      original (identical in both caches -- verified, not assumed)
    views 1-7    the fixed low-pass ladder (cutoffs 0.9 .. 0.12)
    views 8-14   the spatial-TTA views (hflip, 4 corner crops, 2 corner flips)

No GPU work: both caches already exist, this is a pure-numpy concatenation.
The point is to let the same reliability-weighted fusion mechanism arbitrate
across *both* view families instead of just one, to see whether it can reach
(or beat) the spatial-TTA baseline that plain low-pass fusion couldn't -- see
README.md "Baselines and overhead" for why that gap exists.

Read back with tune_fusion.py / evaluate.py --cache-dir cache_combined
--ladder-len 8 (the stability signal only has an ordered "ladder" for views
0-7; views 8-14 get S_k=0, see analysis.stability_from_d).

    python scripts/build_combined_cache.py
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import cache

#: the 15 standard ImageNet-C corruptions, in family order
STANDARD_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

#: number of views that form the ordered filter ladder (view 0 + 7 spectral
#: views); the remaining 7 spatial-TTA views have no rung ordering.
LADDER_LEN = 8


def merge_condition(model_name, corruption, severity, spectral_dir, spatial_dir,
                    spatial_views=range(1, 8)):
    """Combine one (model, corruption, severity) condition, or None if either
    source cache is missing it.

    Args:
        spatial_views: which of cache_spatial's view indices (1-7; 0 is the
            duplicate original) to append after the spectral ladder. Default
            all 7 (the full combined profile); pass e.g. [1] for hflip-only
            (a cheaper variant, see --out-dir cache_combined_lite).
    """
    cache.CACHE_ROOT = spectral_dir
    if not cache.has_condition(model_name, corruption, severity):
        return None
    spec = cache.load_condition(model_name, corruption, severity)

    cache.CACHE_ROOT = spatial_dir
    if not cache.has_condition(model_name, corruption, severity):
        return None
    spat = cache.load_spatial_condition(model_name, corruption, severity)

    spec_paths, spat_paths = list(spec["paths"]), list(spat["paths"])
    if spec_paths != spat_paths:
        # re-order the spatial condition to match the spectral one's image order
        pos = {p: i for i, p in enumerate(spat_paths)}
        if set(spec_paths) != set(spat_paths):
            raise ValueError(
                f"{model_name}/{corruption}/{severity}: image sets differ between "
                f"{spectral_dir} and {spatial_dir}"
            )
        idx = [pos[p] for p in spec_paths]
        spat_probs = spat["probs"][idx]
    else:
        spat_probs = spat["probs"]

    combined = np.concatenate(
        [spec["probs"][:, :LADDER_LEN], spat_probs[:, list(spatial_views)]], axis=1
    ).astype(np.float16)
    assert combined.shape[1] == LADDER_LEN + len(list(spatial_views))
    # Preserve spectral metadata so downstream plots/evaluators can label the
    # ladder correctly for either fixed or geometric caches.
    cutoffs = spec.get("cutoffs")
    ladder = spec.get("ladder")
    view_names = ["original"]
    if cutoffs is not None:
        # Fixed/geometric cutoffs are identical for every image; adaptive energy
        # caches may vary per image, so use generic spectral labels there.
        first = np.asarray(cutoffs[0] if np.ndim(cutoffs) == 2 else cutoffs)
        if np.ndim(cutoffs) == 2 and np.allclose(cutoffs, cutoffs[:1]):
            view_names += [f"lowpass_{x:.6g}" for x in first]
        else:
            view_names += [f"spectral_{i}" for i in range(1, LADDER_LEN)]
    else:
        view_names += [f"spectral_{i}" for i in range(1, LADDER_LEN)]
    spatial_names = [
        "hflip", "top_left", "top_right", "bottom_left", "bottom_right",
        "top_left_hflip", "bottom_right_hflip",
    ]
    selected_names = [spatial_names[i - 1] for i in spatial_views]
    view_names += selected_names
    return combined, spec["labels"], spec["paths"], cutoffs, ladder, view_names


def main():
    """Merge every requested (model, corruption, severity) condition and save it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"],
                    choices=["resnet18", "deit"])
    ap.add_argument("--corruptions", nargs="+",
                    default=STANDARD_CORRUPTIONS + ["clean"])
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--spectral-dir", default="cache_fixed")
    ap.add_argument("--spatial-dir", default="cache_spatial")
    ap.add_argument("--out-dir", default="cache_combined")
    ap.add_argument("--spatial-views", nargs="+", type=int, default=list(range(1, 8)),
                    help="which cache_spatial view indices (1-7) to append after "
                         "the spectral ladder; default all 7. e.g. '--spatial-views 1' "
                         "for hflip-only (pair with --out-dir cache_combined_lite)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    spectral_dir = Path(args.spectral_dir)
    spatial_dir = Path(args.spatial_dir)
    out_dir = Path(args.out_dir)

    conditions = []
    for corr in args.corruptions:
        for sev in ([0] if corr == "clean" else args.severities):
            conditions.append((corr, sev))

    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        n_ok = 0
        for corr, sev in conditions:
            cache.CACHE_ROOT = out_dir
            if not args.overwrite and cache.has_condition(model_name, corr, sev):
                print(f"  skip {corr}/{sev} (cached)")
                n_ok += 1
                continue

            merged = merge_condition(model_name, corr, sev, spectral_dir, spatial_dir,
                                     spatial_views=args.spatial_views)
            if merged is None:
                print(f"  --  {corr}/{sev}: missing in one of the source caches")
                continue
            probs, labels, paths, cutoffs, ladder, view_names = merged

            cache.CACHE_ROOT = out_dir
            out = cache.save_spatial_condition(
                model_name, corr, sev,
                probs=probs, labels=labels, paths=paths,
                cutoffs=cutoffs, ladder=ladder, ladder_len=LADDER_LEN,
                view_names=view_names,
            )
            orig_acc = (probs.astype(np.float32)[:, 0].argmax(-1) == labels).mean()
            print(f"  ok  {corr}/{sev}: {len(paths):5d} imgs  "
                  f"{probs.shape[1]} views  orig acc {100 * orig_acc:5.1f}%  "
                  f"({out.stat().st_size / 1e6:.0f} MB)")
            n_ok += 1

        print(f"  {n_ok}/{len(conditions)} conditions merged")


if __name__ == "__main__":
    main()
