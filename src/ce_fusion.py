"""Cross-entropy fitting for reliability coefficients, with optional view bias.

For standardized per-image/per-view signal vectors x_k = (C_k, A_k, S_k):

    r_k = theta^T x_k + b_k
    w_k = softmax(r)_k
    p*(y) = sum_k w_k p_k(y)

The base fit may constrain theta=(alpha,beta,gamma) to be non-negative or may
leave it unconstrained for the diagnostic requested in the review.  Optional
per-view biases are represented in the identifiable mean-zero subspace.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp

from src.constants import PROB_EPS

DEFAULT_EPS = PROB_EPS


def loss_and_grad(theta, X, p_true, l2=1e-3, eps=DEFAULT_EPS):
    """CE objective/gradient for theta only (kept for backward compatibility)."""
    theta = np.asarray(theta, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)

    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError(f"X must have shape (N, V, 3), got {X.shape}")
    if p_true.shape != X.shape[:2]:
        raise ValueError(f"p_true must have shape {X.shape[:2]}, got {p_true.shape}")

    log_py = np.log(np.clip(p_true, eps, 1.0))
    r = np.einsum("nvd,d->nv", X, theta)
    log_den = logsumexp(r, axis=1)
    log_num = logsumexp(r + log_py, axis=1)

    nll = np.mean(log_den - log_num)
    loss = float(nll + l2 * np.dot(theta, theta))

    w = np.exp(r - log_den[:, None])
    q = np.exp(r + log_py - log_num[:, None])
    grad = np.mean(np.sum((w - q)[..., None] * X, axis=1), axis=0)
    grad = grad + 2.0 * l2 * theta
    return loss, np.asarray(grad, dtype=np.float64)


def mean_zero_bias_from_free(free_bias, n_views):
    """Map V-1 free coordinates to an exactly mean-zero V-vector."""
    free_bias = np.asarray(free_bias, dtype=np.float64)
    if n_views <= 1:
        return np.zeros(n_views, dtype=np.float64)
    if free_bias.shape != (n_views - 1,):
        raise ValueError(f"free bias must have shape {(n_views - 1,)}, got {free_bias.shape}")
    out = np.empty(n_views, dtype=np.float64)
    out[:-1] = free_bias
    out[-1] = -free_bias.sum()
    return out


def _free_bias_from_mean_zero(bias):
    bias = np.asarray(bias, dtype=np.float64)
    centered = bias - bias.mean()
    return centered[:-1].copy()


def loss_and_grad_with_bias(
    params,
    X,
    p_true,
    *,
    l2_theta=1e-3,
    l2_bias=1e-3,
    eps=DEFAULT_EPS,
):
    """CE objective/gradient for theta plus an identifiable mean-zero view bias."""
    X = np.asarray(X, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError(f"X must have shape (N, V, 3), got {X.shape}")
    if p_true.shape != X.shape[:2]:
        raise ValueError(f"p_true must have shape {X.shape[:2]}, got {p_true.shape}")

    n_views = X.shape[1]
    params = np.asarray(params, dtype=np.float64)
    expected = 3 + max(0, n_views - 1)
    if params.shape != (expected,):
        raise ValueError(f"params must have shape {(expected,)}, got {params.shape}")

    theta = params[:3]
    bias = mean_zero_bias_from_free(params[3:], n_views)

    log_py = np.log(np.clip(p_true, eps, 1.0))
    r = np.einsum("nvd,d->nv", X, theta) + bias[None, :]
    log_den = logsumexp(r, axis=1)
    log_num = logsumexp(r + log_py, axis=1)

    nll = np.mean(log_den - log_num)
    loss = float(
        nll + l2_theta * np.dot(theta, theta) + l2_bias * np.dot(bias, bias)
    )

    w = np.exp(r - log_den[:, None])
    q = np.exp(r + log_py - log_num[:, None])
    diff = w - q

    grad_theta = np.mean(np.sum(diff[..., None] * X, axis=1), axis=0)
    grad_theta += 2.0 * l2_theta * theta

    grad_bias_full = np.mean(diff, axis=0) + 2.0 * l2_bias * bias
    # b[-1] = -sum(free), hence db_j/df_j = 1 and db_last/df_j = -1.
    grad_bias_free = grad_bias_full[:-1] - grad_bias_full[-1]
    grad = np.concatenate([grad_theta, grad_bias_free])
    return loss, grad.astype(np.float64)


def fit_coefficients(
    X,
    p_true,
    *,
    l2=1e-3,
    bias_l2=1e-3,
    eps=DEFAULT_EPS,
    starts=None,
    random_restarts=4,
    seed=0,
    maxiter=500,
    nonnegative=True,
    view_bias=False,
):
    """Fit theta and optional mean-zero view bias with L-BFGS-B.

    Returns:
        theta, bias, best_loss, scipy_result
    """
    # SciPy calls the objective many times.  Convert once here so each callback
    # does not allocate a fresh float64 copy of the whole tune set.
    X = np.asarray(X, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)
    n_views = X.shape[1]

    def make_param(theta, bias=None):
        theta = np.asarray(theta, dtype=float)
        if not view_bias:
            return theta.copy()
        if bias is None:
            bias = np.zeros(n_views, dtype=float)
        return np.concatenate([theta, _free_bias_from_mean_zero(bias)])

    base_starts = []
    if starts is not None:
        for s in starts:
            if isinstance(s, dict):
                base_starts.append(make_param(s["theta"], s.get("bias")))
            else:
                arr = np.asarray(s, dtype=float)
                if arr.shape == (3,):
                    base_starts.append(make_param(arr))
                else:
                    base_starts.append(arr.copy())

    base_starts.extend([make_param(np.zeros(3)), make_param(np.ones(3))])

    rng = np.random.default_rng(seed)
    for _ in range(random_restarts):
        theta0 = rng.uniform(0.0, 2.0, size=3) if nonnegative else rng.normal(0.0, 1.0, size=3)
        bias0 = rng.normal(0.0, 0.1, size=n_views) if view_bias else None
        base_starts.append(make_param(theta0, bias0))

    unique = []
    seen = set()
    for p in base_starts:
        p = np.asarray(p, dtype=float)
        if nonnegative:
            p = p.copy()
            p[:3] = np.maximum(p[:3], 0.0)
        key = tuple(np.round(p, 12))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    if view_bias:
        def fun(params):
            return loss_and_grad_with_bias(
                params, X, p_true, l2_theta=l2, l2_bias=bias_l2, eps=eps
            )
        n_params = 3 + n_views - 1
    else:
        def fun(params):
            return loss_and_grad(params, X, p_true, l2=l2, eps=eps)
        n_params = 3

    bounds = None
    if nonnegative:
        bounds = [(0.0, None)] * 3 + [(None, None)] * (n_params - 3)

    best = None
    for x0 in unique:
        res = minimize(
            fun,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8},
        )
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError("no CE optimization starts were generated")

    theta = np.asarray(best.x[:3], dtype=float)
    if view_bias:
        bias = mean_zero_bias_from_free(best.x[3:], n_views).astype(float)
    else:
        bias = np.zeros(n_views, dtype=float)
    return theta, bias, float(best.fun), best



# --------------------------------------------------------------------------- #
# CUDA/torch-backed optimizer objective
# --------------------------------------------------------------------------- #

def _as_torch_float(x, *, device, dtype):
    """Move an array/tensor once to the requested torch device/dtype."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.ascontiguousarray(x), device=device, dtype=dtype)


def loss_and_grad_torch(
    params,
    X,
    p_true,
    *,
    l2_theta=1e-3,
    l2_bias=1e-3,
    eps=DEFAULT_EPS,
    view_bias=False,
    device="cuda",
    dtype=torch.float32,
):
    """Torch equivalent of the CE objective and analytic gradient.

    The large N x V x 3 signal tensor and N x V true-class probabilities are
    evaluated on ``device``.  SciPy still owns the tiny (3 or 3+V-1)
    L-BFGS-B parameter vector, so bounds/restarts and the numerical method stay
    the same as the NumPy implementation while the expensive reductions run on
    CUDA.

    This function is intentionally public so tests can compare it against the
    canonical NumPy implementation.
    """
    X_t = _as_torch_float(X, device=torch.device(device), dtype=dtype)
    p_t = _as_torch_float(p_true, device=X_t.device, dtype=dtype)
    if X_t.ndim != 3 or X_t.shape[-1] != 3:
        raise ValueError(f"X must have shape (N, V, 3), got {tuple(X_t.shape)}")
    if tuple(p_t.shape) != tuple(X_t.shape[:2]):
        raise ValueError(f"p_true must have shape {tuple(X_t.shape[:2])}, got {tuple(p_t.shape)}")

    log_py_t = torch.log(torch.clamp(p_t, min=float(eps), max=1.0))
    return _loss_and_grad_torch_preloaded(
        params,
        X_t,
        log_py_t,
        l2_theta=l2_theta,
        l2_bias=l2_bias,
        view_bias=view_bias,
    )


def _loss_and_grad_torch_preloaded(
    params,
    X_t,
    log_py_t,
    *,
    l2_theta,
    l2_bias,
    view_bias,
):
    """Objective callback for already-resident torch tensors."""
    params = np.asarray(params, dtype=np.float64)
    n_views = X_t.shape[1]
    expected = 3 + (n_views - 1 if view_bias else 0)
    if params.shape != (expected,):
        raise ValueError(f"params must have shape {(expected,)}, got {params.shape}")

    with torch.inference_mode():
        theta = torch.as_tensor(params[:3], device=X_t.device, dtype=X_t.dtype)
        if view_bias:
            free = torch.as_tensor(params[3:], device=X_t.device, dtype=X_t.dtype)
            bias = torch.cat([free, -free.sum().reshape(1)])
        else:
            bias = torch.zeros(n_views, device=X_t.device, dtype=X_t.dtype)

        r = torch.einsum("nvd,d->nv", X_t, theta) + bias[None, :]
        r_true = r + log_py_t
        log_den = torch.logsumexp(r, dim=1)
        log_num = torch.logsumexp(r_true, dim=1)
        nll = (log_den - log_num).mean()
        loss = nll + float(l2_theta) * torch.dot(theta, theta)
        if view_bias:
            loss = loss + float(l2_bias) * torch.dot(bias, bias)

        # Analytic derivative.  Avoid autograd graph construction inside every
        # SciPy callback; these are exactly the w and q terms from the paper
        # derivation used by the NumPy implementation above.
        diff = torch.softmax(r, dim=1) - torch.softmax(r_true, dim=1)
        grad_theta = torch.einsum("nv,nvd->d", diff, X_t) / X_t.shape[0]
        grad_theta = grad_theta + 2.0 * float(l2_theta) * theta

        if view_bias:
            grad_bias_full = diff.mean(dim=0) + 2.0 * float(l2_bias) * bias
            grad_bias_free = grad_bias_full[:-1] - grad_bias_full[-1]
            grad = torch.cat([grad_theta, grad_bias_free])
        else:
            grad = grad_theta

        # .item() / .cpu() synchronize CUDA, which is exactly what SciPy needs
        # before it can choose the next tiny parameter vector.
        return float(loss.item()), grad.double().cpu().numpy()


def fit_coefficients_torch(
    X,
    p_true,
    *,
    l2=1e-3,
    bias_l2=1e-3,
    eps=DEFAULT_EPS,
    starts=None,
    random_restarts=4,
    seed=0,
    maxiter=500,
    nonnegative=True,
    view_bias=False,
    device="cuda",
    dtype="float32",
):
    """L-BFGS-B fit with the expensive CE objective evaluated by torch.

    The optimizer, bounds, starts and mean-zero bias parameterization match
    :func:`fit_coefficients`.  Only the objective/gradient backend changes.
    ``float32`` is the intended CUDA mode; ``float64`` is available as a parity
    diagnostic but is much slower on consumer GPUs.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA CE backend requested but torch.cuda.is_available() is False")
    torch_dtype = torch.float32 if str(dtype).lower() in {"float32", "fp32", "torch.float32"} else torch.float64

    X_t = _as_torch_float(X, device=dev, dtype=torch_dtype)
    p_t = _as_torch_float(p_true, device=dev, dtype=torch_dtype)
    if X_t.ndim != 3 or X_t.shape[-1] != 3:
        raise ValueError(f"X must have shape (N, V, 3), got {tuple(X_t.shape)}")
    if tuple(p_t.shape) != tuple(X_t.shape[:2]):
        raise ValueError(f"p_true must have shape {tuple(X_t.shape[:2])}, got {tuple(p_t.shape)}")

    n_views = X_t.shape[1]
    log_py_t = torch.log(torch.clamp(p_t, min=float(eps), max=1.0))

    def make_param(theta, bias=None):
        theta = np.asarray(theta, dtype=float)
        if not view_bias:
            return theta.copy()
        if bias is None:
            bias = np.zeros(n_views, dtype=float)
        return np.concatenate([theta, _free_bias_from_mean_zero(bias)])

    base_starts = []
    if starts is not None:
        for s in starts:
            if isinstance(s, dict):
                base_starts.append(make_param(s["theta"], s.get("bias")))
            else:
                arr = np.asarray(s, dtype=float)
                if arr.shape == (3,):
                    base_starts.append(make_param(arr))
                else:
                    base_starts.append(arr.copy())
    base_starts.extend([make_param(np.zeros(3)), make_param(np.ones(3))])

    rng = np.random.default_rng(seed)
    for _ in range(random_restarts):
        theta0 = (
            rng.uniform(0.0, 2.0, size=3)
            if nonnegative
            else rng.normal(0.0, 1.0, size=3)
        )
        bias0 = rng.normal(0.0, 0.1, size=n_views) if view_bias else None
        base_starts.append(make_param(theta0, bias0))

    unique = []
    seen = set()
    for p in base_starts:
        p = np.asarray(p, dtype=float)
        if nonnegative:
            p = p.copy()
            p[:3] = np.maximum(p[:3], 0.0)
        key = tuple(np.round(p, 12))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    n_params = 3 + (n_views - 1 if view_bias else 0)
    bounds = None
    if nonnegative:
        bounds = [(0.0, None)] * 3 + [(None, None)] * (n_params - 3)

    def fun(params):
        return _loss_and_grad_torch_preloaded(
            params,
            X_t,
            log_py_t,
            l2_theta=l2,
            l2_bias=bias_l2,
            view_bias=view_bias,
        )

    best = None
    # fp32 objectives do not contain 1e-12 of meaningful precision.  Tolerances
    # below are tight enough to reproduce the fitted direction while avoiding
    # extra L-BFGS-B evaluations chasing float32 noise.
    ftol = 1e-10 if torch_dtype == torch.float32 else 1e-12
    gtol = 1e-6 if torch_dtype == torch.float32 else 1e-8
    for x0 in unique:
        res = minimize(
            fun,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": ftol, "gtol": gtol},
        )
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError("no CE optimization starts were generated")

    theta = np.asarray(best.x[:3], dtype=float)
    if view_bias:
        bias = mean_zero_bias_from_free(best.x[3:], n_views).astype(float)
    else:
        bias = np.zeros(n_views, dtype=float)
    return theta, bias, float(best.fun), best

def fit_nonnegative(
    X,
    p_true,
    *,
    l2=1e-3,
    eps=DEFAULT_EPS,
    starts=None,
    random_restarts=4,
    seed=0,
    maxiter=500,
):
    """Backward-compatible theta-only non-negative fit."""
    theta, _bias, loss, res = fit_coefficients(
        X,
        p_true,
        l2=l2,
        eps=eps,
        starts=starts,
        random_restarts=random_restarts,
        seed=seed,
        maxiter=maxiter,
        nonnegative=True,
        view_bias=False,
    )
    return theta, loss, res
