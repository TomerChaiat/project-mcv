"""Inference overhead of the K+1-view profile vs. a single forward pass.

The proposal promises this as one of the reported metrics; nothing else in the
pipeline measures wall-clock, so this is a standalone timing script (GPU
optional, but the numbers are only meaningful on the device you'd deploy on).

    python scripts/measure_overhead.py                  # both models, batch 64
    python scripts/measure_overhead.py --models resnet18 --batch-size 32
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torchvision.transforms import functional as TF

from src import cache

K = len(cache.FIXED_CUTOFFS)  # 7 filtered views -> K+1 = 8 views total


@torch.no_grad()
def time_forward(model, batch, n_reps, warmup=3):
    """Median per-call seconds of `model(batch)`, GPU-synced."""
    device = batch.device
    for _ in range(warmup):
        model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


@torch.no_grad()
def time_profile(model, images, n_reps, warmup=3):
    """Median per-call seconds of filtering + running all K+1 views."""
    device = images.device

    def one_call():
        views = cache.spectral_views_fixed(images, cache.FIXED_CUTOFFS)
        return cache.view_probs(model, views)

    for _ in range(warmup):
        one_call()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        one_call()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


@torch.no_grad()
def time_combined(model, images256, n_reps, warmup=3, spatial_views=range(1, 8)):
    """Median per-call seconds of filtering + spatial views + running the
    combined profile (see scripts/build_combined_cache.py). `spatial_views`
    selects which of the 7 non-duplicate spatial views to include -- default
    all 7 (15 views total); pass `range(1, 2)` for the hflip-only "lite"
    variant (9 views total, cache_combined_lite)."""
    device = images256.device
    spatial_views = list(spatial_views)

    def one_call():
        spatial = cache.spatial_views(images256)          # (B, 8, 3, 224, 224)
        spectral = cache.spectral_views_fixed(spatial[:, 0], cache.FIXED_CUTOFFS)
        views = torch.cat([spectral, spatial[:, spatial_views]], dim=1)
        return cache.view_probs(model, views)

    for _ in range(warmup):
        one_call()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        one_call()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def load_batch(n, device):
    """`n` real images (resized/cropped, no normalisation) from the clean val set."""
    paths = sorted(glob.glob("data/imagenet/val/*/*.JPEG"))[:n]
    if len(paths) < n:
        raise SystemExit(
            f"need {n} images under data/imagenet/val/, found {len(paths)} "
            "-- run scripts/fetch_clean_images.py first"
        )
    return torch.stack([cache.load_image(p) for p in paths]).to(device)


def load_batch_256(n, device):
    """Same `n` images as 'load_batch', but 256x256 (spatial-TTA geometry)."""
    paths = sorted(glob.glob("data/imagenet/val/*/*.JPEG"))[:n]
    return torch.stack([cache.load_image_256(p) for p in paths]).to(device)


def main():
    """Time 1-view, spectral, lite-combined, and full-combined inference per model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"],
                    choices=["resnet18", "deit"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device} | batch: {args.batch_size} | spectral views: {K + 1} | "
          f"combined views: {K + 1 + 7} | combined-lite views: {K + 1 + 1}\n")

    images = load_batch(args.batch_size, device)
    images256 = load_batch_256(args.batch_size, device)

    print(f"{'model':>10} {'1-view':>9} {'spectral':>10} {'x':>6} "
          f"{'lite (9v)':>10} {'x':>6} {'combined (15v)':>15} {'x':>6}  (ms/img)")
    for model_name in args.models:
        model = cache.LOADERS[model_name](device)[0]

        t_single = time_forward(model, normalize_batch(images), args.reps) / args.batch_size
        t_profile = time_profile(model, images, args.reps) / args.batch_size
        t_lite = time_combined(model, images256, args.reps, spatial_views=range(1, 2)) / args.batch_size
        t_combined = time_combined(model, images256, args.reps) / args.batch_size

        print(f"{model_name:>10} {t_single * 1e3:>9.2f} {t_profile * 1e3:>10.2f} "
              f"{t_profile / t_single:>5.1f}x {t_lite * 1e3:>10.2f} "
              f"{t_lite / t_single:>5.1f}x {t_combined * 1e3:>15.2f} "
              f"{t_combined / t_single:>5.1f}x")


def normalize_batch(images):
    """ImageNet-normalise a [0,1] image batch (the single-view baseline skips
    the low-pass/spatial view builders, which normalise internally via
    'cache.view_probs', so this is needed only for that one comparison)."""
    return TF.normalize(images, cache.MEAN, cache.STD)


if __name__ == "__main__":
    main()
