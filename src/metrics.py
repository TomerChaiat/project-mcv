"""Metrics and fusion over cached multi-view prediction profiles.

The main fusion configuration may vary by held-out corruption family, but the
reported confidence-only and agreement-only baselines are deliberately fixed:
per-image standardized signal, temperature 1.0, and agreement-to-all for the
agreement baseline.  This makes baseline columns comparable across grid, CE,
and refined-CE runs.
"""
from dataclasses import dataclass

import numpy as np

from src import analysis
from src.constants import PROB_EPS


BASELINE_TAU = 1.0


@dataclass(frozen=True)
class FusionConfig:
    """One setting of the fusion hyper-parameters.

    ``view_bias`` is an optional additive prior over views.  ``output_temperature``
    calibrates the already-fused class distribution and therefore does not change
    its argmax prediction.
    """
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    tau: float = 1.0
    standardize: bool = True
    m: int = 1
    agreement: str = "all"
    view_bias: tuple = ()
    output_temperature: float = 1.0

    def as_dict(self):
        return dict(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            tau=self.tau,
            standardize=self.standardize,
            m=self.m,
            agreement=self.agreement,
            view_bias=list(self.view_bias),
            output_temperature=self.output_temperature,
        )

    @classmethod
    def from_dict(cls, d):
        allowed = {
            "alpha", "beta", "gamma", "tau", "standardize", "m", "agreement",
            "view_bias", "output_temperature",
        }
        kwargs = {k: d[k] for k in allowed if k in d}
        if "view_bias" in kwargs and kwargs["view_bias"] is not None:
            kwargs["view_bias"] = tuple(float(x) for x in kwargs["view_bias"])
        return cls(**kwargs)


def top1(probs):
    return probs.argmax(axis=-1)


def accuracy(pred, labels):
    return float(np.mean(pred == labels))


def ece(probs2d, labels, n_bins=15):
    conf = probs2d.max(axis=-1)
    pred = probs2d.argmax(axis=-1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            out += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(out)


def nll(probs2d, labels, eps=PROB_EPS):
    """Mean negative log-likelihood using the project-wide probability floor."""
    probs2d = np.asarray(probs2d, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    p_true = probs2d[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))


def temperature_scale_probs(probs2d, temperature=1.0, eps=PROB_EPS):
    """Apply post-fusion temperature scaling to a probability distribution.

    This is equivalent to softmax(log(p) / T).  Positive T never changes the
    class ordering, so top-1 accuracy is unchanged apart from exact numerical
    ties.
    """
    T = float(temperature)
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"output temperature must be positive, got {temperature}")
    probs2d = np.asarray(probs2d, dtype=np.float32)
    if abs(T - 1.0) < 1e-12:
        return probs2d
    # Preserve exact zeros as zero mass.  The separate NLL function floors only
    # the true-class probability, so T=1 remains exactly the original fused
    # distribution rather than a renormalized all-class epsilon floor.
    z = np.full_like(probs2d, -np.inf, dtype=np.float32)
    positive = probs2d > 0
    z[positive] = np.log(probs2d[positive]) / T
    z -= np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def transfer_rates(orig_pred, new_pred, labels):
    orig_ok = orig_pred == labels
    new_ok = new_pred == labels
    n_orig_ok = int(orig_ok.sum())
    n_orig_bad = int((~orig_ok).sum())
    broke = int((orig_ok & ~new_ok).sum())
    fixed = int((~orig_ok & new_ok).sum())
    return {
        "negative_transfer": broke / n_orig_ok if n_orig_ok else 0.0,
        "positive_transfer": fixed / n_orig_bad if n_orig_bad else 0.0,
        "n_broken": broke,
        "n_fixed": fixed,
    }


def precompute(probs_views, m=1, agreement="all", ladder_len=None):
    """Do config-independent work for one condition.

    Agreement-to-all is always computed because it defines the canonical
    agreement-only baseline.  If the main method requests agreement-to-origin,
    that cheaper vector is computed in addition.
    """
    probs_views = np.asarray(probs_views, dtype=np.float32)
    C, A_all, S = analysis.profile_signals(
        probs_views,
        m=m,
        standardize=False,
        agreement="all",
        ladder_len=ladder_len,
    )
    if agreement == "all":
        A = A_all
    elif agreement == "origin":
        A = analysis.agreement_to_origin(probs_views).astype(np.float32)
    else:
        raise ValueError(f"agreement must be 'all' or 'origin', got {agreement!r}")

    return {
        "probs": probs_views,
        "C": C,
        "A": A,
        "A_baseline_all": A_all,
        "S": S,
        "orig": probs_views[:, 0],
        "equal_pred": top1(probs_views.mean(axis=1)),
    }


def _view_scores(pre, cfg: FusionConfig):
    C, A, S = pre["C"], pre["A"], pre["S"]
    if cfg.standardize:
        C, A, S = analysis.zscore(C), analysis.zscore(A), analysis.zscore(S)
    scores = analysis.reliability(C, A, S, cfg.alpha, cfg.beta, cfg.gamma)
    if cfg.view_bias:
        bias = np.asarray(cfg.view_bias, dtype=np.float32)
        if bias.shape != (scores.shape[1],):
            raise ValueError(
                f"view_bias has length {len(bias)}, but profile has {scores.shape[1]} views"
            )
        scores = scores + bias[None, :]
    return scores


def fused_probs(pre, cfg: FusionConfig):
    scores = _view_scores(pre, cfg)
    w = analysis.fusion_weights(scores, tau=cfg.tau)
    fused = analysis.fuse(pre["probs"], w)
    return temperature_scale_probs(fused, cfg.output_temperature)


def score(pre, labels, cfg: FusionConfig, full=True):
    labels = np.asarray(labels)
    fused = fused_probs(pre, cfg)
    fused_pred = top1(fused)
    fused_acc = accuracy(fused_pred, labels)
    if not full:
        return fused_acc

    orig_pred = top1(pre["orig"])

    # Canonical baselines: fixed definition, independent of the main config.
    conf_signal = analysis.zscore(pre["C"])
    agree_signal = analysis.zscore(pre["A_baseline_all"])
    conf_only = analysis.fuse(
        pre["probs"], analysis.fusion_weights(conf_signal, tau=BASELINE_TAU)
    )
    agree_only = analysis.fuse(
        pre["probs"], analysis.fusion_weights(agree_signal, tau=BASELINE_TAU)
    )

    row = {
        "n": int(len(labels)),
        "orig_acc": accuracy(orig_pred, labels),
        "equal_acc": accuracy(pre["equal_pred"], labels),
        "conf_acc": accuracy(top1(conf_only), labels),
        "agree_acc": accuracy(top1(agree_only), labels),
        "view1_acc": accuracy(top1(pre["probs"][:, 1]), labels),
        "fused_acc": fused_acc,
        "oracle_acc": float(np.mean((top1(pre["probs"]) == labels[:, None]).any(axis=1))),
        "orig_ece": ece(pre["orig"], labels),
        "fused_ece": ece(fused, labels),
        "orig_nll": nll(pre["orig"], labels),
        "fused_nll": nll(fused, labels),
    }
    row.update(transfer_rates(orig_pred, fused_pred, labels))
    return row


def evaluate(probs_views, labels, cfg: FusionConfig):
    return score(
        precompute(probs_views, m=cfg.m, agreement=cfg.agreement),
        labels,
        cfg,
        full=True,
    )
