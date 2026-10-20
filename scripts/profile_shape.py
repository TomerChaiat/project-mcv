"""Do CNNs and ViTs have systematically different spectral stability profiles?

For every model, corruption family and ladder view k (test half, spectral
cache): mean JSD(p_k, p_0) -- how far the prediction drifts from the original
as more high frequencies are removed -- and mean top-1 confidence of view k,
plus the same drift on clean images. Writes results/profile_shape.json;
plotted by scripts/make_paper_figures.py.

    python scripts/profile_shape.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import analysis, cache
from src.splits import ALL_CORRUPTIONS, CORRUPTION_FAMILY, FAMILIES, tune_test_masks

MODELS = ["resnet18", "deit", "convnext"]


def drift_and_conf(probs):
    """Mean JSD of each view to the original and mean max-probability per view.

    Returns:
        (jsd, conf), each of shape (n_views,).
    """
    p = probs.astype(np.float32)
    jsd = analysis.jensen_shannon(p, p[:, :1]).mean(0)          # (K+1,), view 0 -> 0
    conf = p.max(-1).mean(0)
    return jsd, conf


def main():
    """Per model and family, average the drift/confidence profile over the test
    half; print it and save a json.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache_fixed")
    ap.add_argument("--out", default="results/profile_shape.json")
    args = ap.parse_args()
    cache.CACHE_ROOT = Path(args.cache_dir)

    res = {}
    for m in MODELS:
        acc = {f: {"jsd": [], "conf": []} for f in list(FAMILIES) + ["clean"]}
        conds = [(c, s) for c in ALL_CORRUPTIONS for s in range(1, 6)] + [("clean", 0)]
        for c, s in conds:
            d = cache.load_condition(m, c, s)
            _, t = tune_test_masks(d["paths"])
            jsd, conf = drift_and_conf(d["probs"][t])
            fam = CORRUPTION_FAMILY.get(c, "clean")
            acc[fam]["jsd"].append(jsd)
            acc[fam]["conf"].append(conf)
        res[m] = {f: {k: np.mean(v, 0).tolist() for k, v in d_.items()} for f, d_ in acc.items()}
        print(m)
        for f, v in res[m].items():
            print(f"  {f:8s} JSD to original " + " ".join(f"{x:.3f}" for x in v["jsd"])
                  + "   conf " + " ".join(f"{x:.2f}" for x in v["conf"]))
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("saved", args.out)


if __name__ == "__main__":
    main()
