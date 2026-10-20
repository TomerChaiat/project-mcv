"""Does the fitted per-view bias track individual view accuracy, or ensemble
diversity/decorrelation? Three per-view statistics, computed on the tune half
of cache_combined (15 views: 0-7 spectral ladder, 8-14 spatial), pooled over
all 75 conditions, then correlated against the fitted per-view bias vector
("global" config) of a CE fit with a per-view bias (--method ce_bias or
ce_bias_only) for each model:

  acc_k      individual top-1 accuracy of view k alone.
  amb_k      mean Jensen-Shannon divergence of view k's prediction from the
             15-view equal-weight ensemble mean -- the "ambiguity" term of the
             Krogh-Vedelsby ambiguity decomposition. High = this view
             disagrees a lot with the consensus (diverse).
  loo_k      leave-one-out ensemble contribution: acc(14-view equal average
             without view k) subtracted from acc(15-view equal average) --
             positive means removing view k HURTS the ensemble, i.e. it's
             pulling weight even though (per acc_k) it may be individually
             weak.

If the bias correlates with amb_k/loo_k more than with acc_k, it acts as an
ensemble-diversity prior rather than a per-view trust score.

    python scripts/decorrelation_diagnostic.py --cache-dir cache_combined --method ce_bias_only
"""
import argparse
import glob
import json
import re
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import cache
from src.splits import CORRUPTION_FAMILY, tune_test_masks

EPS = 1e-12


def latest_tuning(method, model_name, cache_dir, results_dir="results"):
    """Path of the newest LOFO tuning json for (method, model, cache), or exit if none."""
    tag = Path(cache_dir).name + "_lofo"
    rx = re.compile(rf"tuning_{method}_{model_name}_{tag}_\d{{8}}_\d{{6}}\.json$")
    files = sorted(f for f in glob.glob(f"{results_dir}/tuning_{method}_{model_name}_{tag}_*.json")
                   if rx.search(f))
    if not files:
        raise SystemExit(f"no tuning_{method}_{model_name}_{tag}_*.json in {results_dir} -- "
                         f"run scripts/tune_fusion_ce.py first")
    return files[-1]


def jsd_to_mean(probs_t, mean_t):
    """Base-2 JSD(p_k, p_mean) per image, per view. probs_t (N,V,C), mean_t (N,C)."""
    p = probs_t
    q = mean_t[:, None, :].expand_as(p)
    m = 0.5 * (p + q)
    pa, qa, ma = p + EPS, q + EPS, m + EPS
    kl_pm = (pa * (torch.log(pa) - torch.log(ma))).sum(-1)
    kl_qm = (qa * (torch.log(qa) - torch.log(ma))).sum(-1)
    return (0.5 * kl_pm + 0.5 * kl_qm) / float(np.log(2.0))


def main():
    """Correlate each view's fitted CE bias with its accuracy, ambiguity and
    leave-one-out contribution on the tune half; print and save a json report.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit", "convnext"])
    ap.add_argument("--cache-dir", default="cache_combined")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--method", choices=["ce_bias", "ce_bias_only"], default="ce_bias_only")
    ap.add_argument("--ladder-len", type=int, default=8)
    ap.add_argument("--out", default=None, help="JSON report (default: results/view_bias_diagnostic_<method>.json)")
    args = ap.parse_args()
    report = {"method": args.method, "cache_dir": args.cache_dir, "models": {}}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name in args.models:
        bias_json = latest_tuning(args.method, model_name, args.cache_dir, args.results_dir)
        rec = json.loads(Path(bias_json).read_text())
        bias = np.asarray(rec["global"]["config"]["view_bias"], dtype=np.float64)
        n_views = len(bias)

        cache.CACHE_ROOT = Path(args.cache_dir)
        correct_sum = np.zeros(n_views)
        loo_correct_sum = np.zeros(n_views)  # 14-view equal avg without k
        full_correct_sum = 0.0
        amb_sum = np.zeros(n_views)
        n_total = 0

        for corruption in CORRUPTION_FAMILY:
            for sev in (1, 2, 3, 4, 5):
                if not cache.has_condition(model_name, corruption, sev):
                    continue
                d = cache.load_condition(model_name, corruption, sev)
                tune_mask, _ = tune_test_masks(d["paths"], 0.5, 0)
                probs = np.ascontiguousarray(d["probs"][tune_mask])
                labels = np.asarray(d["labels"][tune_mask], dtype=np.int64)
                probs_t = torch.from_numpy(probs).to(device=device, dtype=torch.float32)
                labels_t = torch.from_numpy(labels).to(device=device)
                n = probs_t.shape[0]

                pred_k = probs_t.argmax(-1)                      # (N, V)
                correct_k = pred_k.eq(labels_t[:, None])          # (N, V)
                correct_sum += correct_k.sum(0).cpu().numpy()

                mean_all = probs_t.mean(1)                        # (N, C)
                full_correct_sum += mean_all.argmax(-1).eq(labels_t).sum().item()

                amb = jsd_to_mean(probs_t, mean_all)               # (N, V)
                amb_sum += amb.sum(0).cpu().numpy()

                # leave-one-out equal-average: (sum_all - p_k) / (V-1)
                sum_all = probs_t.sum(1, keepdim=True)             # (N,1,C)
                loo_mean = (sum_all - probs_t) / (n_views - 1)      # (N,V,C)
                loo_pred = loo_mean.argmax(-1)                      # (N,V)
                loo_correct_sum += loo_pred.eq(labels_t[:, None]).sum(0).cpu().numpy()

                n_total += n

        acc_k = correct_sum / n_total
        amb_k = amb_sum / n_total
        full_acc = full_correct_sum / n_total
        loo_acc_k = loo_correct_sum / n_total          # acc of ensemble w/o view k
        contribution_k = full_acc - loo_acc_k           # >0: removing k hurts -> k helps

        print(f"\n=== {model_name}  (n_tune_images={n_total}, bias from {bias_json}) ===")
        print(f"{'view':>4} {'bias':>8} {'acc_k':>8} {'ambig':>8} {'loo_contrib':>12}")
        for k in range(n_views):
            print(f"{k:>4} {bias[k]:>8.3f} {100*acc_k[k]:>7.2f}% {amb_k[k]:>8.4f} {100*contribution_k[k]:>11.3f}pt")

        def pearson(x, y):
            """Pearson correlation of x and y, as a float."""
            x = np.asarray(x); y = np.asarray(y)
            return float(np.corrcoef(x, y)[0, 1])

        def spearman(x, y):
            """Spearman rank correlation of x and y, as a float."""
            x = np.asarray(x); y = np.asarray(y)
            rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
            return pearson(rx, ry)

        print(f"  corr(bias, acc_k)         pearson={pearson(bias, acc_k):+.3f}  spearman={spearman(bias, acc_k):+.3f}")
        print(f"  corr(bias, ambiguity)     pearson={pearson(bias, amb_k):+.3f}  spearman={spearman(bias, amb_k):+.3f}")
        print(f"  corr(bias, loo_contrib)   pearson={pearson(bias, contribution_k):+.3f}  spearman={spearman(bias, contribution_k):+.3f}")
        report["models"][model_name] = {
            "tuning": bias_json, "bias": bias.tolist(), "view_acc": acc_k.tolist(),
            "ambiguity": amb_k.tolist(), "loo_contribution": contribution_k.tolist(),
            "corr": {k: {"pearson": pearson(bias, v), "spearman": spearman(bias, v)}
                     for k, v in (("view_acc", acc_k), ("ambiguity", amb_k),
                                  ("loo_contribution", contribution_k))}}

    out = Path(args.out or f"{args.results_dir}/view_bias_diagnostic_{args.method}.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
