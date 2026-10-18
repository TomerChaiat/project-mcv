"""Fetch clean ImageNet-val images for the 100 subset classes.

Source:
    evanarlian/imagenet_1k_resized_256

The two validation parquet shards are downloaded through the Hugging Face
cache, filtered to the subset class indices stored in
data/subset_classes.json, and written as JPEG files under:

    data/imagenet/val/<wnid>/

This version does not require a full 1000-class ImageNet-C directory for
label verification. It trusts the committed wnid -> original ImageNet index
mapping in subset_classes.json.
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image


REPO = "evanarlian/imagenet_1k_resized_256"

VAL_SHARDS = [
    "data/val-00000-of-00002-b5248be478d25e41.parquet",
    "data/val-00001-of-00002-85f3d9c8fa1edb63.parquet",
]


def load_subset(path):
    """Load and validate the committed wnid -> ImageNet-index mapping."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "wnid_to_index" not in data:
        raise SystemExit(
            f"{path}: missing required 'wnid_to_index' field"
        )

    subset = data["wnid_to_index"]

    if len(subset) != 100:
        raise SystemExit(
            f"{path}: expected 100 subset classes, found {len(subset)}"
        )

    indices = list(subset.values())

    if len(set(indices)) != len(indices):
        raise SystemExit(
            f"{path}: duplicate ImageNet class indices found"
        )

    bad = [
        (wnid, idx)
        for wnid, idx in subset.items()
        if not isinstance(idx, int) or not 0 <= idx < 1000
    ]

    if bad:
        raise SystemExit(
            f"{path}: invalid ImageNet class index entries, e.g. {bad[:5]}"
        )

    return subset


def main():
    """Download clean ImageNet validation images for the selected subset."""
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--subset",
        default="data/subset_classes.json",
        help="JSON file containing wnid_to_index",
    )

    ap.add_argument(
        "--out",
        default="data/imagenet/val",
        help="output directory",
    )

    ap.add_argument(
        "--per-class",
        type=int,
        default=0,
        help="cap images per class (0 = keep all, normally 50)",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite JPEG files that already exist",
    )

    args = ap.parse_args()

    subset = load_subset(args.subset)

    idx_to_wnid = {
        idx: wnid
        for wnid, idx in subset.items()
    }

    want_idx = set(idx_to_wnid)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    counts = {wnid: 0 for wnid in subset}

    # Count already-existing files so rerunning the script is safe.
    if not args.overwrite:
        for wnid in subset:
            cdir = out_root / wnid
            if cdir.exists():
                counts[wnid] = sum(
                    1
                    for p in cdir.glob("*.JPEG")
                    if p.is_file()
                )

    print(f"subset: {len(subset)} classes")
    print(
        f"existing images: {sum(counts.values())}"
        if not args.overwrite
        else "overwrite mode enabled"
    )

    for shard in VAL_SHARDS:
        print(f"\ndownloading/checking shard: {shard}")

        path = hf_hub_download(
            repo_id=REPO,
            filename=shard,
            repo_type="dataset",
        )

        print("reading", os.path.basename(path))

        pf = pq.ParquetFile(path)

        for batch in pf.iter_batches(
            batch_size=512,
            columns=["image", "label"],
        ):
            d = batch.to_pydict()

            for img_obj, label in zip(
                d["image"],
                d["label"],
            ):
                if label not in want_idx:
                    continue

                wnid = idx_to_wnid[label]

                if (
                    args.per_class
                    and counts[wnid] >= args.per_class
                ):
                    continue

                n = counts[wnid]

                # Normal ImageNet validation has 50 images per class.
                # If files already exist, avoid rewriting them.
                out_path = (
                    out_root
                    / wnid
                    / f"{wnid}_val_{n:05d}.JPEG"
                )

                if out_path.exists() and not args.overwrite:
                    counts[wnid] += 1
                    continue

                raw = (
                    img_obj["bytes"]
                    if isinstance(img_obj, dict)
                    else img_obj
                )

                im = Image.open(
                    io.BytesIO(raw)
                ).convert("RGB")

                out_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                im.save(
                    out_path,
                    quality=95,
                )

                counts[wnid] += 1

    total = sum(counts.values())

    empty = [
        wnid
        for wnid, count in counts.items()
        if count == 0
    ]

    print()
    print(
        f"wrote/found {total} images across "
        f"{len(counts) - len(empty)} classes "
        f"-> {out_root}"
    )

    print(
        f"per class: min {min(counts.values())}, "
        f"max {max(counts.values())}"
    )

    wrong_count = {
        wnid: count
        for wnid, count in counts.items()
        if count != 50
    }

    if empty:
        print(
            f"WARNING: {len(empty)} classes got 0 images: "
            f"{empty[:10]}"
        )

    if wrong_count:
        print(
            f"WARNING: {len(wrong_count)} classes do not have "
            f"exactly 50 images"
        )
        for wnid, count in list(wrong_count.items())[:10]:
            print(f"  {wnid}: {count}")
    else:
        print(
            "clean ImageNet subset OK: "
            "100 classes x 50 images = 5000 images"
        )


if __name__ == "__main__":
    main()