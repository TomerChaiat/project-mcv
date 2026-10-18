#!/usr/bin/env python3
"""Validate the cached inputs required by the final comprehensive run."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src import cache  # noqa: E402
from src.splits import CORRUPTION_FAMILY  # noqa: E402

MODELS = ("resnet18", "deit")


def expected_paths(root: Path, model: str):
    for corruption in CORRUPTION_FAMILY:
        for severity in range(1, 6):
            yield corruption, severity, root / model / corruption / f"{severity}.npz"
    yield "clean", 0, root / model / "clean" / "0.npz"


def validate_root(root: Path, name: str):
    result = {"root": str(root), "models": {}}
    failures = []
    for model in MODELS:
        missing = []
        present = 0
        for corruption, severity, path in expected_paths(root, model):
            if path.is_file():
                present += 1
            else:
                missing.append(f"{corruption}/{severity}")
        result["models"][model] = {
            "present": present,
            "expected": 76,
            "missing": missing,
        }
        if missing:
            failures.append(f"{name}/{model}: {len(missing)} missing")
    return result, failures


def check_combined_view_count(root: Path):
    """Load only one small metadata record per model to catch the wrong cache."""
    checks = {}
    old = cache.CACHE_ROOT
    try:
        cache.CACHE_ROOT = root
        for model in MODELS:
            d = cache.load_condition(model, "gaussian_noise", 1)
            views = int(d["probs"].shape[1])
            checks[model] = views
            if views != 15:
                raise SystemExit(
                    f"combined cache {root} has {views} views for {model}; expected 15"
                )
    finally:
        cache.CACHE_ROOT = old
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-cache", required=True)
    ap.add_argument("--spatial-cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    combined = (REPO / args.combined_cache).resolve() if not Path(args.combined_cache).is_absolute() else Path(args.combined_cache)
    spatial = (REPO / args.spatial_cache).resolve() if not Path(args.spatial_cache).is_absolute() else Path(args.spatial_cache)

    combined_rec, f1 = validate_root(combined, "combined")
    spatial_rec, f2 = validate_root(spatial, "spatial")
    view_counts = check_combined_view_count(combined)

    report = {
        "repo": str(REPO),
        "combined": combined_rec,
        "spatial": spatial_rec,
        "combined_view_counts": view_counts,
        "protocol": {
            "corruptions": len(CORRUPTION_FAMILY),
            "severities": [1, 2, 3, 4, 5],
            "conditions_per_model": 75,
            "plus_clean": True,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    failures = f1 + f2
    if failures:
        raise SystemExit("cache validation failed: " + "; ".join(failures))

    print("cache validation passed: combined and spatial are 76/76 for both models")
    print("combined profile view count: 15 for both models")


if __name__ == "__main__":
    main()
