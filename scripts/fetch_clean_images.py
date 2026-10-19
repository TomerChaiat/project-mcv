"""Fetch clean ImageNet-val images for the 100 subset classes.

Source: evanarlian/imagenet_1k_resized_256 (ungated re-host of the official
ImageNet-1k validation split, shorter side resized to 256 -- which is exactly
what the eval pipeline does before a 224 center crop, so no quality is lost).

The two val parquet shards (~0.9 GB) are downloaded to the HF cache, filtered
to the subset wnids, written out as JPEG under data/imagenet/val/<wnid>/, then
you can delete the HF cache if you want the space back.
"""
import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "evanarlian/imagenet_1k_resized_256"
VAL_SHARDS = [
    "data/val-00000-of-00002-b5248be478d25e41.parquet",
    "data/val-00001-of-00002-85f3d9c8fa1edb63.parquet",
]
WNID_RE = re.compile(r"n\d{8}")


def sorted_wnids(reference_dir):
    """The 1000 wnid folders in `reference_dir`, sorted (position == class index).

    Used only to assert that the parquet's integer labels agree with our
    wnid ordering before anything is written.
    """
    wnids = sorted(d for d in os.listdir(reference_dir) if WNID_RE.fullmatch(d))
    if len(wnids) != 1000:
        raise SystemExit(f"{reference_dir}: need all 1000 wnid folders")
    return wnids


def main():
    """Download the val parquet shards, keep the subset classes, write JPEGs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/subset_classes.json")
    ap.add_argument("--reference-dir", default="data/imagenet-c/gaussian_noise/1")
    ap.add_argument("--out", default="data/imagenet/val")
    ap.add_argument("--per-class", type=int, default=0,
                    help="cap images per class (0 = keep all ~50)")
    args = ap.parse_args()

    subset = json.load(open(args.subset))["wnid_to_index"]
    idx_to_wnid = {v: k for k, v in subset.items()}
    want_idx = set(idx_to_wnid)
    # sanity: the parquet label ids must line up with our wnid ordering
    all_wnids = sorted_wnids(args.reference_dir)
    for wnid, idx in subset.items():
        assert all_wnids[idx] == wnid, "label-index / wnid mismatch"

    out_root = Path(args.out)
    counts = {w: 0 for w in subset}

    for shard in VAL_SHARDS:
        path = hf_hub_download(REPO, shard, repo_type="dataset")
        print("reading", os.path.basename(path))
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=512, columns=["image", "label"]):
            d = batch.to_pydict()
            for img_obj, label in zip(d["image"], d["label"]):
                if label not in want_idx:
                    continue
                wnid = idx_to_wnid[label]
                if args.per_class and counts[wnid] >= args.per_class:
                    continue
                raw = img_obj["bytes"] if isinstance(img_obj, dict) else img_obj
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                cdir = out_root / wnid
                cdir.mkdir(parents=True, exist_ok=True)
                n = counts[wnid]
                im.save(cdir / f"{wnid}_val_{n:05d}.JPEG", quality=95)
                counts[wnid] = n + 1

    total = sum(counts.values())
    empty = [w for w, c in counts.items() if c == 0]
    print(f"\nwrote {total} images across {len(counts) - len(empty)} classes -> {out_root}")
    print(f"per class: min {min(counts.values())}, max {max(counts.values())}")
    if empty:
        print(f"WARNING: {len(empty)} classes got 0 images: {empty[:10]}")


if __name__ == "__main__":
    main()
