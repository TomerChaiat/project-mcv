"""How much of the oracle gap is reachable from C/A/S alone?

Optimized implementation: every cached condition is loaded once and its C/A/S
signals are computed once with the same torch/CUDA JSD backend used by the GPU
tuner.  The four leave-one-family-out folds then reuse those features.  The
scikit-learn fits remain CPU work by design; use --no-gbm when only the linear
ceiling is needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src import analysis, cache
from src.evaluation import device_from_arg
from src.metrics import accuracy, top1
from src.splits import CORRUPTION_FAMILY, leave_one_family_out, tune_test_masks
from src.evaluation import signal_variants_torch


def precompute_conditions(
    model_name,
    cache_dir,
    severities,
    frac_tune,
    split_seed,
    *,
    m,
    agreement,
    ladder_len,
    device,
    signal_chunk,
    signal_pair_batch,
):
    """Load each condition once; cache tune/test features and test probabilities."""
    cache.CACHE_ROOT = Path(cache_dir)
    out = {}
    t0 = time.time()
    for c in CORRUPTION_FAMILY:
        for sev in severities:
            if not cache.has_condition(model_name, c, sev):
                continue
            d = cache.load_condition(model_name, c, sev)
            tune_mask, test_mask = tune_test_masks(d["paths"], frac_tune, split_seed)
            probs = np.ascontiguousarray(d["probs"])
            labels = np.asarray(d["labels"], dtype=np.int64)
            probs_t = torch.as_tensor(probs, device=device, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, device=device, dtype=torch.long)
            with torch.inference_mode():
                _raw, std = signal_variants_torch(
                    probs_t, [m], [agreement], ladder_len=ladder_len,
                    chunk=signal_chunk, pair_batch=signal_pair_batch,
                )
                C, A, S = std[(m, agreement)]
                X = torch.stack([C, A, S], dim=-1).cpu().numpy().astype(np.float32, copy=False)
                correct = probs_t.argmax(-1).eq(labels_t[:, None]).cpu().numpy()
            out[(c, sev)] = {
                "X_tune": X[tune_mask].reshape(-1, 3),
                "y_tune": correct[tune_mask].reshape(-1),
                "X_test": X[test_mask].reshape(-1, 3),
                "test_shape": (int(test_mask.sum()), probs.shape[1]),
                "probs_test": np.ascontiguousarray(probs[test_mask]),
                "labels_test": labels[test_mask],
            }
            print(f"  precompute {c:18s}/{sev}  [{time.time()-t0:.1f}s cumulative]")
    print(f"  precomputed {len(out)} conditions once in {time.time()-t0:.1f}s")
    return out


def stack_train(pre, corruptions, severities):
    xs, ys = [], []
    for c in corruptions:
        for sev in severities:
            rec = pre.get((c, sev))
            if rec is not None:
                xs.append(rec["X_tune"])
                ys.append(rec["y_tune"])
    if not xs:
        return None, None
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def fused_acc_from_scores(pre, corruptions, severities, score_fn):
    accs = []
    for c in corruptions:
        for sev in severities:
            rec = pre.get((c, sev))
            if rec is None:
                continue
            scores = score_fn(rec["X_test"]).reshape(rec["test_shape"])
            w = analysis.fusion_weights(scores, tau=1.0)
            probs = rec["probs_test"].astype(np.float32)
            pred = top1(analysis.fuse(probs, w))
            accs.append(accuracy(pred, rec["labels_test"]))
    return float(np.mean(accs)) if accs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"], choices=["resnet18", "deit"])
    ap.add_argument("--cache-dir", default="cache_fixed")
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--agreement", choices=["all", "origin"], default="origin")
    ap.add_argument("--ladder-len", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--signal-chunk", type=int, default=256)
    ap.add_argument("--signal-pair-batch", type=int, default=32)
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(args.device)
    report = {}

    for model_name in args.models:
        print(f"\n=== {model_name}  ({args.cache_dir}) ===")
        pre = precompute_conditions(
            model_name, args.cache_dir, args.severities,
            args.frac_tune, args.split_seed,
            m=args.m, agreement=args.agreement, ladder_len=args.ladder_len,
            device=device, signal_chunk=args.signal_chunk,
            signal_pair_batch=args.signal_pair_batch,
        )
        per_family = {}
        for held, tune_corr, test_corr in leave_one_family_out():
            X, y = stack_train(pre, tune_corr, args.severities)
            if X is None:
                continue

            t0 = time.time()
            lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
            lr_acc = fused_acc_from_scores(pre, test_corr, args.severities, lr.decision_function)
            entry = {"logistic_acc": lr_acc}
            msg = f"  hold out {held:9s}: logistic {lr_acc*100:.1f}%"

            if not args.no_gbm:
                gb = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, subsample=0.5, random_state=0,
                ).fit(X, y)
                gb_acc = fused_acc_from_scores(pre, test_corr, args.severities, gb.decision_function)
                entry["gbm_acc"] = gb_acc
                msg += f"   gbm {gb_acc*100:.1f}%"

            print(msg + f"   (n_train_views={len(y)}, {time.time()-t0:.1f}s fit/eval)")
            per_family[held] = entry
        report[model_name] = per_family

    path = out_dir / f"signal_ceiling_{Path(args.cache_dir).name}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
