"""Does adding per-image ambiguity (D_k) as a 4th CE-fitted signal beat the
existing 3-signal ce_bias result? Same LOFO discipline as the main pipeline:
fit on tune half (per held-out family), select nothing extra (bias is always
on), evaluate on the test half, bootstrap 4-signal vs 3-signal.

D_k(image) = JSD(p_k, mean_over_views(p))  -- the ambiguity term of the
Krogh-Vedelsby ensemble decomposition. decorrelation_diagnostic.py shows the
fitted ce_bias bias vector correlates far more with each view's leave-one-out
ensemble contribution than with its individual accuracy; D is the natural
per-image signal that turns that diagnostic into something CE can actually
fit on.

(m, agreement) per model are taken from that model's CE + bias fit ("global"
config), and both the 3- and the 4-signal model are fitted here with the same
settings, so the only difference between them is the presence of D. The
test-half difference gets a paired cluster bootstrap over images.

    python scripts/experiment_ambiguity.py --cache-dir cache_combined --out results
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import analysis, cache
from src.bootstrap import cluster_bootstrap, image_keys
from src.ce_fusion import fit_coefficients_torch
from src.evaluation import device_from_arg, signal_variants_torch, pairwise_jsd_torch
from src.splits import CORRUPTION_FAMILY, leave_one_family_out, tune_test_masks

EPS = 1e-12


def ce_bias_settings(model_name, cache_dir, results_dir):
    """(m, agreement) of the newest CE + bias fit for this model and cache."""
    import glob, re
    tag = Path(cache_dir).name + "_lofo"
    rx = re.compile(rf"tuning_ce_bias_{model_name}_{tag}_\d{{8}}_\d{{6}}\.json$")
    files = sorted(f for f in glob.glob(f"{results_dir}/tuning_ce_bias_{model_name}_{tag}_*.json")
                   if rx.search(f))
    if not files:
        raise SystemExit(f"no CE + bias tuning for {model_name} in {results_dir}")
    cfg = json.loads(Path(files[-1]).read_text())["global"]["config"]
    return int(cfg["m"]), str(cfg["agreement"])


def zscore_t(x, eps=1e-8):
    """Z-score `x` along the last dim (sd < eps treated as 1). Returns a tensor of the same shape."""
    mu = x.mean(dim=-1, keepdim=True)
    sd = torch.sqrt(((x - mu) ** 2).mean(dim=-1, keepdim=True))
    denom = torch.where(sd < eps, torch.ones_like(sd), sd)
    return (x - mu) / denom


def ambiguity_t(probs_t):
    """D_k(image) = JSD(p_k, mean_over_views(p)), base-2, per image per view."""
    mean_t = probs_t.mean(dim=1, keepdim=True)  # (N,1,C)
    p = probs_t
    q = mean_t.expand_as(p)
    m = 0.5 * (p + q)
    pa, qa, ma = p + EPS, q + EPS, m + EPS
    kl_pm = (pa * (torch.log(pa) - torch.log(ma))).sum(-1)
    kl_qm = (qa * (torch.log(qa) - torch.log(ma))).sum(-1)
    return (0.5 * kl_pm + 0.5 * kl_qm) / float(np.log(2.0))


def load_condition_signals(model_name, corruption, sev, m, agreement, device, half_fn):
    """Load one condition's half (selected by `half_fn`) and compute its signals.

    Returns:
        (probs_t, labels_t, X3, X4, p_true, keys): X3 = (C, A, S), X4 adds the
        ambiguity signal D; p_true is each view's probability of the true label.
    """
    d = cache.load_condition(model_name, corruption, sev)
    mask = half_fn(d["paths"])
    probs = np.ascontiguousarray(d["probs"][mask])
    labels = np.asarray(d["labels"][mask], dtype=np.int64)
    probs_t = torch.from_numpy(probs).to(device=device, dtype=torch.float32)
    labels_t = torch.from_numpy(labels).to(device=device)
    with torch.inference_mode():
        _raw, std = signal_variants_torch(probs_t, [m], [agreement], ladder_len=8)
        C, A, S = std[(m, agreement)]
        D = zscore_t(ambiguity_t(probs_t))
        X3 = torch.stack([C, A, S], dim=-1)          # (N, V, 3)
        X4 = torch.stack([C, A, S, D], dim=-1)        # (N, V, 4)
        p_true = probs_t.gather(2, labels_t[:, None, None].expand(-1, probs_t.shape[1], 1)).squeeze(2)
    return probs_t, labels_t, X3, X4, p_true, image_keys(np.asarray(d["paths"])[mask])


def tune_half(paths):
    """Bool mask of the tune half (frac 0.5, seed 0) for `paths`."""
    m, _ = tune_test_masks(paths, 0.5, 0)
    return m


def test_half(paths):
    """Bool mask of the test half (frac 0.5, seed 0) for `paths`."""
    _, m = tune_test_masks(paths, 0.5, 0)
    return m


def fit_and_score(model_name, device, cache_dir, out_dir, results_dir="results", n_boot=2000):
    """Fit 3-signal and 4-signal (with ambiguity) CE + bias weights per LOFO
    fold on the tune half, score both on the test half and bootstrap the
    difference; saves per-image correctness.

    Returns:
        Results dict: per-fold fits, test mCA of both variants and the
        4 - 3 signal bootstrap.
    """
    m, agreement = ce_bias_settings(model_name, cache_dir, results_dir)
    cache.CACHE_ROOT = Path(cache_dir)

    # ---- tune half: build per-condition X3/X4/p_true, cache in RAM ----
    tune_data = {}
    for corruption in CORRUPTION_FAMILY:
        for sev in (1, 2, 3, 4, 5):
            if not cache.has_condition(model_name, corruption, sev):
                continue
            probs_t, labels_t, X3, X4, p_true, _ = load_condition_signals(
                model_name, corruption, sev, m, agreement, device, tune_half
            )
            tune_data[(corruption, sev)] = dict(X3=X3, X4=X4, p_true=p_true)

    n_views = tune_data[next(iter(tune_data))]["X3"].shape[1]
    results = {"model": model_name, "m": m, "agreement": agreement, "held_out": {}}

    for held, tune_corr, _ in leave_one_family_out():
        keys = [(c, s) for c in tune_corr for s in (1, 2, 3, 4, 5) if (c, s) in tune_data]
        X3 = torch.cat([tune_data[k]["X3"] for k in keys], dim=0)
        X4 = torch.cat([tune_data[k]["X4"] for k in keys], dim=0)
        p_true = torch.cat([tune_data[k]["p_true"] for k in keys], dim=0)

        t0 = time.time()
        theta3, bias3, loss3, _ = fit_coefficients_torch(
            X3, p_true, view_bias=True, nonnegative=True, random_restarts=4, device=device
        )
        theta4, bias4, loss4, _ = fit_coefficients_torch(
            X4, p_true, view_bias=True, nonnegative=True, random_restarts=4, device=device
        )
        results["held_out"][held] = {
            "theta3": theta3.tolist(), "bias3": bias3.tolist(), "ce_nll3": loss3,
            "theta4": theta4.tolist(), "bias4": bias4.tolist(), "ce_nll4": loss4,
            "n_samples": int(len(X3)), "fit_time_s": time.time() - t0,
        }
        print(f"  [{model_name}] hold out {held:9s}: "
              f"3-sig theta={np.round(theta3,3)} NLL={loss3:.4f}  |  "
              f"4-sig theta={np.round(theta4,3)} NLL={loss4:.4f}  D_weight={theta4[3]:+.4f}  "
              f"[{time.time()-t0:.1f}s]")

    del tune_data

    # ---- test half: evaluate per-corruption using the held-out fold's fitted config ----
    per_cond_cache = {}
    fam_theta3 = {h: results["held_out"][h]["theta3"] for h in results["held_out"]}
    fam_bias3 = {h: results["held_out"][h]["bias3"] for h in results["held_out"]}
    fam_theta4 = {h: results["held_out"][h]["theta4"] for h in results["held_out"]}
    fam_bias4 = {h: results["held_out"][h]["bias4"] for h in results["held_out"]}

    # per-condition scoring using each condition's OWN held-out-family config
    def score_lofo(theta_by_fam, bias_by_fam, sig_key):
        """Test-half accuracy with each condition scored by its family's fold weights.

        Returns:
            (mean accuracy over conditions, {"<corruption>__<sev>": per-image correctness}).
        """
        acc_by_cond = []
        correctness = {}
        for corruption in CORRUPTION_FAMILY:
            fam = CORRUPTION_FAMILY[corruption]
            th = torch.as_tensor(theta_by_fam[fam], device=device, dtype=torch.float32)
            bs = torch.as_tensor(bias_by_fam[fam], device=device, dtype=torch.float32)
            for sev in (1, 2, 3, 4, 5):
                if not cache.has_condition(model_name, corruption, sev):
                    continue
                key = (corruption, sev)
                if key not in per_cond_cache:
                    probs_t, labels_t, X3, X4, p_true, keys = load_condition_signals(
                        model_name, corruption, sev, m, agreement, device, test_half
                    )
                    # kept in host memory: all 75 test conditions do not fit on the GPU
                    per_cond_cache[key] = (probs_t.half().cpu(), labels_t.cpu(), X3.cpu(), X4.cpu(), keys)
                probs_t, labels_t, X3, X4, _ = per_cond_cache[key]
                probs_t, labels_t = probs_t.to(device).float(), labels_t.to(device)
                X3, X4 = X3.to(device), X4.to(device)
                X = X4 if sig_key == 4 else X3
                scores = torch.einsum("nvd,d->nv", X, th) + bs[None, :]
                w = torch.softmax(scores, dim=1)
                fused = torch.bmm(w[:, None, :], probs_t).squeeze(1)
                pred = fused.argmax(-1)
                ok = pred.eq(labels_t)
                acc_by_cond.append(ok.float().mean().item())
                correctness[f"{corruption}__{sev}"] = ok.cpu().numpy()
        return float(np.mean(acc_by_cond)), correctness

    mca3, corr3 = score_lofo(fam_theta3, fam_bias3, 3)
    mca4, corr4 = score_lofo(fam_theta4, fam_bias4, 4)

    print(f"  [{model_name}] TEST HALF  mCA 3-signal(C,A,S)+bias = {100*mca3:.3f}%   "
          f"4-signal(C,A,S,D)+bias = {100*mca4:.3f}%   delta = {100*(mca4-mca3):+.3f} pts")

    bs = cluster_bootstrap(
        [{"keys": per_cond_cache[k][4], "a": corr4[f"{k[0]}__{k[1]}"], "b": corr3[f"{k[0]}__{k[1]}"]}
         for k in per_cond_cache], n_boot=n_boot, seed=0)
    print(f"  [{model_name}] 4-signal - 3-signal = {100*bs['point']:+.3f} pts  "
          f"95% CI [{100*bs['ci95'][0]:+.3f}, {100*bs['ci95'][1]:+.3f}]")
    results["test_mca_3signal"] = mca3
    results["test_mca_4signal"] = mca4
    results["four_minus_three"] = bs
    np.savez_compressed(
        out_dir / f"ambiguity_correctness_{model_name}.npz",
        **{f"corr3__{k}": v for k, v in corr3.items()},
        **{f"corr4__{k}": v for k, v in corr4.items()},
    )
    return results


def main():
    """Run the ambiguity experiment for every model and save ambiguity_experiment.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit", "convnext"])
    ap.add_argument("--cache-dir", default="cache_combined")
    ap.add_argument("--results-dir", default="results", help="where the CE + bias tuning JSONs are")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg("auto")
    all_results = {}
    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        all_results[model_name] = fit_and_score(model_name, device, args.cache_dir, out_dir,
                                                results_dir=args.results_dir)
    (out_dir / "ambiguity_experiment.json").write_text(json.dumps(all_results, indent=2))
    print("\nsaved", out_dir / "ambiguity_experiment.json")


if __name__ == "__main__":
    main()
