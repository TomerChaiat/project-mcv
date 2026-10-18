"""GPU evaluation helpers for cached multi-view probability profiles.

Same definitions as the NumPy ``metrics`` path (cross-checked in
test/test_evaluation.py), with the heavy work in torch, and able to score
*many* fusion configurations from one pairwise-JSD pass per condition.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from src import analysis
from src.constants import PROB_EPS
from src.metrics import BASELINE_TAU, FusionConfig



def _zscore_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-image z-score matching NumPy's population-standard-deviation rule."""
    mu = x.mean(dim=-1, keepdim=True)
    sd = torch.sqrt(((x - mu) ** 2).mean(dim=-1, keepdim=True))
    denom = torch.where(sd < eps, torch.ones_like(sd), sd)
    return (x - mu) / denom


def pairwise_jsd_torch(
    probs: torch.Tensor, *, pair_batch: int = 32, eps: float = analysis.EPS
) -> torch.Tensor:
    """Pairwise base-2 JSD without materializing an N x V x V x C tensor."""
    if probs.ndim != 3:
        raise ValueError(f"probs must have shape (N,V,C), got {tuple(probs.shape)}")
    if pair_batch < 1:
        raise ValueError("pair_batch must be >= 1")
    n, v, _ = probs.shape
    d = torch.zeros((n, v, v), device=probs.device, dtype=probs.dtype)
    if v <= 1:
        return d
    ii, jj = torch.triu_indices(v, v, offset=1, device=probs.device)
    ln2 = float(np.log(2.0))
    for lo in range(0, ii.numel(), pair_batch):
        i = ii[lo : lo + pair_batch]
        j = jj[lo : lo + pair_batch]
        p = probs[:, i, :]
        q = probs[:, j, :]
        m = 0.5 * (p + q)
        pa, qa, ma = p + eps, q + eps, m + eps
        js = 0.5 * (
            (pa * (torch.log(pa) - torch.log(ma))).sum(dim=-1)
            + (qa * (torch.log(qa) - torch.log(ma))).sum(dim=-1)
        ) / ln2
        d[:, i, j] = js
        d[:, j, i] = js
    return d


def _stability_from_d_torch(d: torch.Tensor, *, m: int, ladder_len=None) -> torch.Tensor:
    nviews = d.shape[-1]
    L = nviews if ladder_len is None else int(ladder_len)
    if not 0 <= L <= nviews:
        raise ValueError(f"ladder_len must be in [0,{nviews}], got {L}")
    out = torch.zeros(d.shape[:-1], device=d.device, dtype=d.dtype)
    for k in range(L):
        lo, hi = max(0, k - m), min(L - 1, k + m)
        idx = [j for j in range(lo, hi + 1) if j != k]
        if idx:
            out[..., k] = -d[..., k, idx].mean(dim=-1)
    if 0 < L < nviews:
        # non-ladder views are neutral: mean ladder S (z-score 0), see
        # analysis.stability_from_d
        out[..., L:] = out[..., :L].mean(dim=-1, keepdim=True)
    return out


def signal_variants_torch(
    probs_t: torch.Tensor, m_values, agreement_modes, *, ladder_len=None,
    chunk=256, pair_batch=32
):
    """Compute requested confidence/agreement/stability variants on one device."""
    if probs_t.dtype != torch.float32:
        probs_t = probs_t.float()
    n, v, _ = probs_t.shape
    C = probs_t.max(dim=-1).values
    A = {a: torch.empty((n, v), device=probs_t.device, dtype=torch.float32) for a in agreement_modes}
    S = {m: torch.empty((n, v), device=probs_t.device, dtype=torch.float32) for m in m_values}
    for start in range(0, n, chunk):
        p = probs_t[start : start + chunk]
        d = pairwise_jsd_torch(p, pair_batch=pair_batch)
        stop = start + len(p)
        if "all" in A:
            A["all"][start:stop] = -d.sum(dim=-1) / (v - 1)
        if "origin" in A:
            A["origin"][start:stop] = -d[:, :, 0]
        for radius in m_values:
            S[radius][start:stop] = _stability_from_d_torch(d, m=radius, ladder_len=ladder_len)
    C_std = _zscore_torch(C)
    A_std = {k: _zscore_torch(x) for k, x in A.items()}
    S_std = {k: _zscore_torch(x) for k, x in S.items()}
    raw, std = {}, {}
    for radius in m_values:
        for agreement in agreement_modes:
            raw[(radius, agreement)] = (C, A[agreement], S[radius])
            std[(radius, agreement)] = (C_std, A_std[agreement], S_std[radius])
    return raw, std

def device_from_arg(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def _ece_from_conf_pred(
    conf: torch.Tensor,
    pred: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Torch equivalent of ``src.metrics.ece`` from confidence + argmax.

    Uses one bucket/scatter reduction instead of 15 ``mask.any().item()`` CUDA
    synchronizations.  ``bucketize(..., right=False)`` has the desired
    interval convention for interior boundaries: (lo, hi].
    """
    correct = pred.eq(labels).to(torch.float32)
    valid = conf > 0.0  # metrics.ece excludes conf == 0 from every (lo, hi] bin
    if not bool(valid.any().item()):
        return 0.0
    c = conf[valid]
    y = correct[valid]
    boundaries = torch.linspace(
        0.0, 1.0, n_bins + 1, device=conf.device, dtype=conf.dtype
    )[1:-1]
    bins = torch.bucketize(c, boundaries, right=False)
    counts = torch.bincount(bins, minlength=n_bins).to(torch.float32)
    conf_sum = torch.bincount(bins, weights=c.to(torch.float32), minlength=n_bins)
    corr_sum = torch.bincount(bins, weights=y, minlength=n_bins)
    nz = counts > 0
    gap = torch.zeros_like(counts)
    gap[nz] = torch.abs(corr_sum[nz] / counts[nz] - conf_sum[nz] / counts[nz])
    out = (counts / float(conf.numel()) * gap).sum()
    return float(out.item())


def _nll_from_probs(probs: torch.Tensor, labels: torch.Tensor, eps=PROB_EPS) -> float:
    p_true = probs.gather(1, labels[:, None]).squeeze(1)
    return float((-torch.log(torch.clamp(p_true, min=float(eps), max=1.0))).mean().item())


def _calibrated_fused_stats(
    fused: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    *,
    eps=PROB_EPS,
):
    """Return pred, ECE, NLL after probability-temperature calibration.

    For T != 1 this matches ``metrics.temperature_scale_probs`` without
    materialising a second N x C calibrated probability tensor.  Exact zero
    probabilities stay zero; the project-wide epsilon is applied only when NLL
    is measured, matching ``metrics.nll``.
    """
    T = float(temperature)
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"output temperature must be positive, got {temperature}")

    pred = fused.argmax(dim=1)
    if abs(T - 1.0) < 1e-12:
        conf = fused.max(dim=1).values
        return pred, _ece_from_conf_pred(conf, pred, labels), _nll_from_probs(fused, labels, eps)

    neg_inf = torch.full((), -torch.inf, device=fused.device, dtype=fused.dtype)
    logp = torch.where(fused > 0, torch.log(fused), neg_inf)
    scaled = logp / T
    log_den = torch.logsumexp(scaled, dim=1)

    pred_scaled_logp = scaled.gather(1, pred[:, None]).squeeze(1) - log_den
    conf = torch.exp(pred_scaled_logp)

    true_scaled_logp = scaled.gather(1, labels[:, None]).squeeze(1) - log_den
    log_eps = float(np.log(float(eps)))
    nll = (-torch.clamp(true_scaled_logp, min=log_eps)).mean()
    return pred, _ece_from_conf_pred(conf, pred, labels), float(nll.item())


def _transfer_rates_torch(orig_pred, new_pred, labels):
    orig_ok = orig_pred.eq(labels)
    new_ok = new_pred.eq(labels)
    n_orig_ok = int(orig_ok.sum().item())
    n_orig_bad = int((~orig_ok).sum().item())
    broke = int((orig_ok & ~new_ok).sum().item())
    fixed = int((~orig_ok & new_ok).sum().item())
    return {
        "negative_transfer": broke / n_orig_ok if n_orig_ok else 0.0,
        "positive_transfer": fixed / n_orig_bad if n_orig_bad else 0.0,
        "n_broken": broke,
        "n_fixed": fixed,
    }


def _fuse_from_cfg(
    probs_t: torch.Tensor,
    triplet,
    cfg: FusionConfig,
) -> torch.Tensor:
    C, A, S = triplet
    scores = float(cfg.alpha) * C + float(cfg.beta) * A + float(cfg.gamma) * S
    if cfg.view_bias:
        bias = torch.as_tensor(cfg.view_bias, device=probs_t.device, dtype=torch.float32)
        if tuple(bias.shape) != (probs_t.shape[1],):
            raise ValueError(
                f"view_bias has length {len(bias)} but profile has {probs_t.shape[1]} views"
            )
        scores = scores + bias[None, :]
    weights = torch.softmax(scores / float(cfg.tau), dim=1)
    return torch.bmm(weights[:, None, :], probs_t).squeeze(1)


def fused_correct_many_torch(
    probs_views,
    labels,
    configs: Mapping[str, FusionConfig],
    *,
    ladder_len=None,
    device="auto",
    signal_chunk=256,
    signal_pair_batch=32,
):
    """Per-image correctness for several fused configs from one JSD pass.

    Returns ``(correctness, orig_correct)`` where correctness maps method names
    to CPU bool arrays.  This is the lightweight path used by paired bootstrap.
    """
    dev = device_from_arg(device) if not isinstance(device, torch.device) else device
    probs_t = torch.as_tensor(np.ascontiguousarray(probs_views), device=dev, dtype=torch.float32)
    labels_t = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=dev)

    if not configs:
        orig = probs_t[:, 0].argmax(1).eq(labels_t).cpu().numpy()
        return {}, orig

    m_values = sorted({int(c.m) for c in configs.values()})
    agreements = sorted(set(["all"] + [str(c.agreement) for c in configs.values()]))
    with torch.inference_mode():
        raw, std = signal_variants_torch(
            probs_t,
            m_values,
            agreements,
            ladder_len=ladder_len,
            chunk=signal_chunk,
            pair_batch=signal_pair_batch,
        )
        out = {}
        for name, cfg in configs.items():
            triplet = (std if cfg.standardize else raw)[(cfg.m, cfg.agreement)]
            fused = _fuse_from_cfg(probs_t, triplet, cfg)
            out[name] = fused.argmax(1).eq(labels_t).cpu().numpy()
        orig = probs_t[:, 0].argmax(1).eq(labels_t).cpu().numpy()
    return out, orig


def fit_vanilla_temperature(
    cache_dir,
    model_name,
    corruptions,
    severities,
    *,
    frac_tune=0.5,
    split_seed=0,
    device="auto",
    temperatures=None,
    eps=PROB_EPS,
):
    """Fit one scalar output temperature for the raw, unfused view-0 predictor.

    Standard post-hoc temperature scaling (Guo et al. 2017): minimizes NLL of
    the original softmax on the tune half only, pooled across every cached
    corruption/severity for this model -- the same half, same NLL objective,
    and same grid search that ``tune_output_temperature`` uses to calibrate
    the *fused* predictor. This gives a fair "orig, also calibrated" baseline
    instead of comparing a calibrated fused predictor to an uncalibrated raw
    one.
    """
    from src import cache as cache_mod
    from src.splits import tune_test_masks

    dev = device_from_arg(device) if not isinstance(device, torch.device) else device
    if temperatures is None:
        temperatures = np.geomspace(0.25, 4.0, 49).astype(np.float32)
    temperatures = np.asarray(temperatures, dtype=np.float32)
    temperatures_t = torch.as_tensor(temperatures, device=dev, dtype=torch.float32)
    nll_sums = torch.zeros(len(temperatures), device=dev, dtype=torch.float64)
    n_total = 0

    prev_root = cache_mod.CACHE_ROOT
    cache_mod.CACHE_ROOT = Path(cache_dir)
    try:
        for corruption in corruptions:
            for sev in severities:
                if not cache_mod.has_condition(model_name, corruption, sev):
                    continue
                d = cache_mod.load_condition(model_name, corruption, sev)
                tune_mask, _ = tune_test_masks(d["paths"], frac_tune, split_seed)
                probs = np.ascontiguousarray(d["probs"][tune_mask][:, 0])
                labels = np.asarray(d["labels"][tune_mask], dtype=np.int64)
                orig = torch.as_tensor(probs, device=dev, dtype=torch.float32)
                labels_t = torch.as_tensor(labels, device=dev)
                with torch.inference_mode():
                    logp = torch.where(
                        orig > 0, torch.log(orig), torch.full_like(orig, -torch.inf)
                    )
                    # Floor only the true-class term (matches
                    # tune_output_temperature / metrics.nll): a float16-cached
                    # probability can round the true class to exactly 0, which
                    # would make every temperature's NLL +inf and turn the grid
                    # search into "pick the first entry" instead of a real fit.
                    true_prob = orig.gather(1, labels_t[:, None]).squeeze(1)
                    true_logp = torch.log(torch.clamp(true_prob, min=float(eps)))
                    for i in range(len(temperatures)):
                        T = temperatures_t[i]
                        z = logp / T
                        log_den = torch.logsumexp(z, dim=1)
                        true_scaled = true_logp / T
                        nll_sums[i] += (log_den - true_scaled).sum()
                n_total += len(labels)
    finally:
        cache_mod.CACHE_ROOT = prev_root

    if n_total == 0:
        raise RuntimeError(f"no tune-half data found for {model_name} in {cache_dir}")
    means = (nll_sums / n_total).cpu().numpy()
    idx = int(np.argmin(means))
    return float(temperatures[idx]), float(means[idx])


def score_many_torch(
    probs_views,
    labels,
    configs: Mapping[str, FusionConfig],
    *,
    ladder_len=None,
    device="auto",
    signal_chunk=256,
    signal_pair_batch=32,
    return_correctness=False,
    orig_temperature=1.0,
):
    """Score many fusion configs while sharing one condition's JSD/signals.

    The returned row schema matches ``src.metrics.score(..., full=True)``.
    Baselines are computed once and copied into every method row.
    """
    if not configs:
        return ({}, {}) if return_correctness else {}

    dev = device_from_arg(device) if not isinstance(device, torch.device) else device
    probs_t = torch.as_tensor(np.ascontiguousarray(probs_views), device=dev, dtype=torch.float32)
    labels_t = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=dev)
    n = int(labels_t.numel())
    n_views = int(probs_t.shape[1])

    m_values = sorted({int(c.m) for c in configs.values()})
    agreements = sorted(set(["all"] + [str(c.agreement) for c in configs.values()]))

    with torch.inference_mode():
        raw, std = signal_variants_torch(
            probs_t,
            m_values,
            agreements,
            ladder_len=ladder_len,
            chunk=signal_chunk,
            pair_batch=signal_pair_batch,
        )

        # C and A_all do not depend on m; select any requested m for the fixed
        # confidence/agreement-only baselines.
        C_std, A_all_std, _ = std[(m_values[0], "all")]

        orig = probs_t[:, 0]
        orig_pred = orig.argmax(1)
        equal_pred = probs_t.mean(1).argmax(1)

        conf_w = torch.softmax(C_std / float(BASELINE_TAU), dim=1)
        conf_pred = torch.bmm(conf_w[:, None, :], probs_t).squeeze(1).argmax(1)
        agree_w = torch.softmax(A_all_std / float(BASELINE_TAU), dim=1)
        agree_pred = torch.bmm(agree_w[:, None, :], probs_t).squeeze(1).argmax(1)

        view1_idx = 1 if n_views > 1 else 0
        view1_pred = probs_t[:, view1_idx].argmax(1)
        oracle_ok = probs_t.argmax(2).eq(labels_t[:, None]).any(1)

        orig_conf = orig.max(1).values
        # Calibrated-vanilla baseline: same temperature-scaling machinery as the
        # fused predictor's cfg.output_temperature, fit separately on view 0
        # alone (see fit_vanilla_temperature). Defaults to 1.0 (a no-op), so
        # existing callers that never pass orig_temperature are unaffected.
        _, orig_ece_t, orig_nll_t = _calibrated_fused_stats(
            orig, labels_t, orig_temperature
        )
        shared = {
            "n": n,
            "orig_acc": float(orig_pred.eq(labels_t).float().mean().item()),
            "equal_acc": float(equal_pred.eq(labels_t).float().mean().item()),
            "conf_acc": float(conf_pred.eq(labels_t).float().mean().item()),
            "agree_acc": float(agree_pred.eq(labels_t).float().mean().item()),
            "view1_acc": float(view1_pred.eq(labels_t).float().mean().item()),
            "oracle_acc": float(oracle_ok.float().mean().item()),
            "orig_ece": _ece_from_conf_pred(orig_conf, orig_pred, labels_t),
            "orig_nll": _nll_from_probs(orig, labels_t),
            "orig_temperature": float(orig_temperature),
            "orig_ece_calibrated": orig_ece_t,
            "orig_nll_calibrated": orig_nll_t,
        }

        rows = {}
        correctness = {}
        for name, cfg in configs.items():
            triplet = (std if cfg.standardize else raw)[(cfg.m, cfg.agreement)]
            fused = _fuse_from_cfg(probs_t, triplet, cfg)
            fused_pred, fused_ece, fused_nll = _calibrated_fused_stats(
                fused, labels_t, cfg.output_temperature
            )
            row = dict(shared)
            row.update(
                fused_acc=float(fused_pred.eq(labels_t).float().mean().item()),
                fused_ece=fused_ece,
                fused_nll=fused_nll,
            )
            row.update(_transfer_rates_torch(orig_pred, fused_pred, labels_t))
            rows[name] = row
            if return_correctness:
                correctness[name] = fused_pred.eq(labels_t).cpu().numpy()

        if return_correctness:
            correctness["__orig__"] = orig_pred.eq(labels_t).cpu().numpy()
            return rows, correctness
        return rows
