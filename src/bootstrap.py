"""Paired cluster bootstrap over images for mean-over-conditions accuracy deltas.

The headline statistic is mean accuracy over the 75 (corruption, severity)
conditions, each itself a mean over held-out images. The SAME images appear in
every condition (ImageNet-C corrupts one fixed set of validation images), so
per-image outcomes are strongly correlated across conditions: an image that
fusion fixes under gaussian_noise/3 is likely fixed under gaussian_noise/4 too.
Resampling images independently inside each condition treats those as 75
independent draws and understates the variance.

This module resamples *images* (clusters), once per bootstrap replicate, and
applies the same draw to every condition the image appears in. Per condition
the replicate statistic is the weighted mean of the paired difference
d = correct_A - correct_B over that condition's images, the weights being how
often each image was drawn; the replicate is the mean of those over conditions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def image_keys(paths):
    """Image ids shared across conditions (ImageNet-C keeps the file stem)."""
    return np.asarray([Path(str(p)).stem for p in paths])


def cluster_bootstrap(conditions, n_boot=2000, seed=0, chunk=250):
    """Paired cluster bootstrap of mean_c mean_i (a_ci - b_ci).

    Args:
        conditions: iterable of dicts with 'keys' (image ids, aligned with the
            arrays), 'a' and 'b' (per-image bool correctness of the two
            predictors on the same images).
        n_boot: bootstrap replicates.
        seed: RNG seed.
        chunk: replicates drawn per batch (bounds memory at chunk x n_images).

    Returns:
        dict with 'point', 'ci95' [lo, hi], 'se', 'p_le_zero' (fraction of
        replicates <= 0), 'n_conditions', 'n_images', 'n_boot'.
    """
    conditions = list(conditions)
    if not conditions:
        raise ValueError("no conditions to bootstrap")
    all_keys = sorted({str(k) for c in conditions for k in c["keys"]})
    index = {k: i for i, k in enumerate(all_keys)}
    n_img = len(all_keys)

    prepared = []
    points = []
    for c in conditions:
        a = np.asarray(c["a"], bool)
        b = np.asarray(c["b"], bool)
        if a.shape != b.shape or len(a) != len(c["keys"]):
            raise ValueError("a, b and keys must be aligned per condition")
        idx = np.asarray([index[str(k)] for k in c["keys"]], dtype=np.int64)
        d = a.astype(np.float64) - b.astype(np.float64)
        prepared.append((idx, d))
        points.append(d.mean())

    rng = np.random.default_rng(seed)
    reps = np.empty(int(n_boot), dtype=np.float64)
    for lo in range(0, int(n_boot), chunk):
        nb = min(chunk, int(n_boot) - lo)
        w = rng.multinomial(n_img, np.full(n_img, 1.0 / n_img), size=nb).astype(np.float64)
        acc = np.zeros(nb, dtype=np.float64)
        for idx, d in prepared:
            wc = w[:, idx]
            tot = wc.sum(axis=1)
            # A replicate that draws none of this condition's images (only
            # possible for tiny conditions) falls back to the point value.
            safe = np.where(tot > 0, tot, 1.0)
            acc += np.where(tot > 0, (wc @ d) / safe, d.mean())
        reps[lo:lo + nb] = acc / len(prepared)

    lo_q, hi_q = np.percentile(reps, [2.5, 97.5])
    return {
        "point": float(np.mean(points)),
        "ci95": [float(lo_q), float(hi_q)],
        "se": float(reps.std(ddof=1)),
        "p_le_zero": float(np.mean(reps <= 0.0)),
        "n_conditions": len(prepared),
        "n_images": n_img,
        "n_boot": int(n_boot),
    }
