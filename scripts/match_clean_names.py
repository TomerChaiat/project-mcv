"""Recover which clean validation image is which ImageNet-C image.

The clean images (scripts/fetch_clean_images.py) come from a Hugging Face
re-host that renamed every file (<wnid>_val_<i>.JPEG), while ImageNet-C keeps
the original ILSVRC2012_val_<id>.JPEG names. Both are the same 50 validation
images per class, so they are matched by content: for each class, every clean
image (resize 256 + centre-crop 224, the geometry ImageNet-C was built with)
is correlated against every ImageNet-C image of a mild corruption, and the
assignment is solved one-to-one (Hungarian). A second corruption must give the
identical assignment.

Writes data/clean_name_map.json: {clean stem: ImageNet-C stem}. Used by
src/splits.image_key so a clean image falls in the same tune/test half as its
corrupted versions, and by scripts/spectral_energy.py to pair images.

    python scripts/match_clean_names.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.cache import load_image


def features(paths):
    """Grayscale 56x56, z-normalised per image -> (N, 3136)."""
    x = torch.stack([load_image(p) for p in paths]).mean(1, keepdim=True)
    x = F.avg_pool2d(x, 4).flatten(1).numpy().astype(np.float64)
    x -= x.mean(1, keepdims=True)
    x /= x.std(1, keepdims=True) + 1e-8
    return x


def match(clean_paths, corr_paths):
    a, b = features(clean_paths), features(corr_paths)
    corr = a @ b.T / a.shape[1]                      # Pearson correlation matrix
    rows, cols = linear_sum_assignment(-corr)
    s = np.sort(corr, axis=1)
    return cols[np.argsort(rows)], corr[rows, cols], s[:, -1] - s[:, -2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-root", default="data/imagenet/val")
    ap.add_argument("--c-root", default="data/imagenet-c")
    ap.add_argument("--ref", default="brightness/1")
    ap.add_argument("--check", default="jpeg_compression/1")
    ap.add_argument("--out", default="data/clean_name_map.json")
    args = ap.parse_args()

    mapping, worst_corr, worst_margin, n_disagree = {}, 1.0, 1.0, 0
    for wdir in sorted(Path(args.clean_root).iterdir()):
        clean = sorted(wdir.glob("*.JPEG"))
        ref = sorted((Path(args.c_root) / args.ref / wdir.name).glob("*.JPEG"))
        chk = sorted((Path(args.c_root) / args.check / wdir.name).glob("*.JPEG"))
        if len(clean) != len(ref) or [p.name for p in ref] != [p.name for p in chk]:
            raise SystemExit(f"{wdir.name}: image counts/names differ between sources")
        idx, c, margin = match(clean, ref)
        idx2, _, _ = match(clean, chk)
        n_disagree += int((idx != idx2).sum())
        worst_corr, worst_margin = min(worst_corr, c.min()), min(worst_margin, margin.min())
        for p, j in zip(clean, idx):
            mapping[p.stem] = ref[j].stem
    print(f"matched {len(mapping)} clean images; worst matched correlation {worst_corr:.4f}, "
          f"worst best-vs-second margin {worst_margin:.4f}; "
          f"assignments differing between {args.ref} and {args.check}: {n_disagree}")
    if n_disagree:
        raise SystemExit("the two reference corruptions disagree -- not writing a map")
    if len(set(mapping.values())) != len(mapping):
        raise SystemExit("matching is not one-to-one -- not writing a map")
    Path(args.out).write_text(json.dumps(mapping, indent=1, sort_keys=True))
    print("saved", args.out)


if __name__ == "__main__":
    main()
