"""Pick the fusion method per model from TUNE-HALF accuracy only, and freeze it.

Candidates are the tuning JSONs of one cache tag: grid (tune_fusion.py) and the
CE variants (tune_fusion_ce.py). Each holds one config per leave-one-family-out
fold. The selection score is the cross-family one the LOFO protocol is about:
every corruption x severity is scored on its tune half with the config of the
fold that held its family out (the same protocol evaluate.py applies to the
test half), so each config is judged on a family it never saw and no test-half
image is touched. The in-family fit score (``held_out[family]
["mean_fused_acc"]``, accuracy on the three families a fold was trained on) is
printed alongside for reference only. The winner is copied to
``<out>/tuning_<model>_<tag>_final_<stamp>.json`` for evaluate.py --tuning.

    python scripts/select_method.py --models resnet18 deit --tag cache_combined_lofo --results results
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import cache
from src.evaluation import device_from_arg, fused_correct_many_torch
from src.metrics import FusionConfig
from src.splits import CORRUPTION_FAMILY, tune_test_masks

METHODS = ["grid", "ce", "ce_bias", "ce_unconstrained", "ce_bias_only"]


def latest(results, method, model, tag):
    """Newest tuning JSON for exactly this method (anchored: 'ce_bias' must not
    match 'ce_bias_only')."""
    rx = re.compile(rf"tuning_{re.escape(method)}_{re.escape(model)}_{re.escape(tag)}_\d{{8}}_\d{{6}}\.json$")
    files = [f for f in glob.glob(str(Path(results) / f"tuning_{method}_{model}_{tag}_*.json"))
             if rx.search(f)]
    return sorted(files)[-1] if files else None


def fit_score(rec):
    """Mean over folds of tune-half accuracy on the fold's own training families."""
    return float(np.mean([v["mean_fused_acc"] for v in rec["held_out"].values()]))


def held_out_scores(model, recs, cache_dir, device):
    """Tune-half accuracy of each method on the families its folds held out.

    Every condition is loaded once and all methods are scored from one
    signal pass. Returns {method: mean accuracy over conditions}.
    """
    meta = next(iter(recs.values()))
    for m, rec in recs.items():
        for k in ("ladder_len", "split", "tune_severities"):
            if rec.get(k) != meta.get(k):
                raise SystemExit(f"{model}: {m} has {k}={rec.get(k)!r}, others {meta.get(k)!r}")
    split = meta.get("split") or {"frac_tune": 0.5, "seed": 0}
    cfgs = {m: {fam: FusionConfig.from_dict(v["config"]) for fam, v in rec["held_out"].items()}
            for m, rec in recs.items()}

    cache.CACHE_ROOT = Path(cache_dir)
    accs = {m: [] for m in recs}
    for corruption, fam in CORRUPTION_FAMILY.items():
        for sev in meta["tune_severities"]:
            if not cache.has_condition(model, corruption, sev):
                # a partial cache would rank the methods on a subset of conditions
                raise SystemExit(f"{model}: missing cache {corruption}/{sev} in {cache_dir}")
            d = cache.load_condition(model, corruption, sev)
            tune_mask, _ = tune_test_masks(d["paths"], split["frac_tune"], split["seed"])
            ok, _ = fused_correct_many_torch(
                d["probs"][tune_mask], d["labels"][tune_mask],
                {m: c[fam] for m, c in cfgs.items()},
                ladder_len=meta.get("ladder_len"), device=device,
            )
            for m in recs:
                accs[m].append(float(ok[m].mean()))
    return {m: float(np.mean(a)) for m, a in accs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tag", required=True, help="tuning tag, e.g. cache_combined_lofo")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="where to write the frozen copy (default: --results)")
    ap.add_argument("--cache-dir", default=None,
                    help="profile cache to score on (default: the one recorded in the tuning JSONs)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    out = Path(args.out or args.results)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = device_from_arg(args.device)
    summary = {}

    for model in args.models:
        paths = {m: latest(args.results, m, model, args.tag) for m in METHODS}
        paths = {m: p for m, p in paths.items() if p}
        if not paths:
            raise SystemExit(f"no tuning JSONs for {model} tag={args.tag} in {args.results}")
        recs = {m: json.loads(Path(p).read_text()) for m, p in paths.items()}
        cache_dir = args.cache_dir or next(iter(recs.values()))["cache_dir"]
        held = held_out_scores(model, recs, cache_dir, device)
        fit = {m: fit_score(r) for m, r in recs.items()}
        winner = max(held, key=held.get)

        print(f"\n=== {model} ({args.tag}) - LOFO tune-half accuracy ===")
        print(f"  {'method':18s} {'held-out fam.':>13s} {'fit fams.':>10s}")
        for m in sorted(held, key=lambda m: -held[m]):
            print(f"  {m:18s} {100 * held[m]:12.3f}% {100 * fit[m]:9.3f}%"
                  + ("   <- selected" if m == winner else ""))
        dst = out / f"tuning_{model}_{args.tag}_final_{stamp}.json"
        shutil.copyfile(paths[winner], dst)
        print(f"  frozen: {dst}")
        summary[model] = {"selected": winner, "frozen": str(dst),
                          "criterion": "tune-half accuracy on each fold's held-out family",
                          "tune_half_acc_heldout": held,
                          "tune_half_acc_fit": fit,
                          "sources": paths}

    sp = out / f"selection_{args.tag}_{stamp}.json"
    sp.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {sp}")


if __name__ == "__main__":
    main()
