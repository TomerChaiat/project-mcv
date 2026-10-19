"""GPU step, run once: compute the 8-view spatial-TTA baseline profile for
every (model, corruption, severity) and write it to cache_spatial/.

This is the "spatial TTA with equal view count" baseline from the proposal:
same number of views (8) and forward passes as the spectral profile, but the
views are crop/flip variants instead of low-pass filters. Read back with
scripts/evaluate_tta.py.

    python scripts/cache_spatial_profiles.py
    python scripts/cache_spatial_profiles.py --models resnet18 --corruptions fog

Re-run any time: conditions already cached are skipped unless --overwrite.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from src import cache

#: the 15 standard ImageNet-C corruptions, in family order (same list as
#: scripts/cache_profiles.py)
STANDARD_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def main():
    """Iterate (model, corruption, severity), compute each spatial profile, save to cache_spatial/."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"],
                    choices=["resnet18", "deit"])
    ap.add_argument("--corruptions", nargs="+",
                    default=STANDARD_CORRUPTIONS + ["clean"])
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--subset", default="data/subset_classes.json")
    ap.add_argument("--n-per-class", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefetch-batches", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cache-dir", default="cache_spatial")
    args = ap.parse_args()

    cache.CACHE_ROOT = Path(args.cache_dir)
    wnid_to_index = json.load(open(args.subset))["wnid_to_index"]
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("device:", device, "| classes:", len(wnid_to_index))

    conditions = []
    for corr in args.corruptions:
        for sev in ([0] if corr == "clean" else args.severities):
            conditions.append((corr, sev))

    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        model = cache.LOADERS[model_name](device)[0]

        for corr, sev in conditions:
            if not args.overwrite and cache.has_condition(model_name, corr, sev):
                print(f"  skip {corr}/{sev} (cached)")
                continue

            paths, labels = cache.list_condition_images(
                corr, sev, wnid_to_index, n_per_class=args.n_per_class,
            )
            if not paths:
                print(f"  --  {corr}/{sev}: no images on disk, skipping")
                continue

            t0 = time.time()
            probs = cache.compute_spatial_condition(
                model, paths, device, batch_size=args.batch_size,
                workers=args.workers, prefetch_batches=args.prefetch_batches,
            )
            out = cache.save_spatial_condition(
                model_name, corr, sev, probs=probs, labels=labels, paths=paths,
            )
            orig_acc = (probs.astype(np.float32)[:, 0].argmax(-1) == labels).mean()
            size_mb = out.stat().st_size / 1e6
            print(f"  ok  {corr}/{sev}: {len(paths):5d} imgs  "
                  f"{time.time() - t0:5.0f}s  orig acc {100 * orig_acc:5.1f}%  "
                  f"({size_mb:.0f} MB)")

    total = sum(p.stat().st_size for p in cache.CACHE_ROOT.rglob("*.npz")) / 1e9
    print(f"\ncache total: {total:.1f} GB in {cache.CACHE_ROOT}/")


if __name__ == "__main__":
    main()
