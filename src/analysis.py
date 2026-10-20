"""The spectral stability profile maths: the three reliability signals and the
fusion, over a whole condition at once.

The array 'probs' has shape (N, K+1, C): N images, K+1 views each (view 0 is the
unfiltered image), C classes -- exactly what 'src/cache.py' stores. This is the
single implementation of the profile maths; 'src/metrics.py' and the scripts
build everything on top of it.

Everything here is plain NumPy except 'profile_signals': its pairwise-JSD step
builds an (N, K+1, K+1, C) intermediate, which is the dominant cost of a full
tune/evaluate run (it is called once per condition, often per grid point), so
that one function runs on torch tensors -- GPU when available -- and converts
back to NumPy at the end. Callers are unaffected; the public inputs/outputs are
NumPy arrays either way.
"""

import numpy as np
import torch

EPS = 1e-12


def kl(a, b):
    """KL(a || b) in bits, summed over the last (class) axis; 'EPS' avoids log 0."""
    a = a + EPS
    b = b + EPS
    return np.sum(a * np.log2(a / b), axis=-1)


def jensen_shannon(p, q):
    """Jensen-Shannon divergence (base 2) between two probability arrays.

    JSD(p, q) = 0.5 KL(p || m) + 0.5 KL(q || m) with m = 0.5 (p + q). Symmetric,
    and bounded in [0, 1] for valid distributions.

    Args:
        p, q: arrays that broadcast against each other, last axis = classes.

    Returns:
        array of the broadcast shape without the class axis.
    """
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def confidence(probs):
    """C_k = max_c p_k(c): the top class probability of every view.

    Args:
        probs: (..., K+1, C) array.

    Returns:
        (..., K+1) array in [0, 1].
    """
    return probs.max(axis=-1)


def pairwise_jsd(probs):
    """Jensen-Shannon divergence between every pair of views.

    Args:
        probs: (M, K+1, C) array.

    Returns:
        (M, K+1, K+1) array, base-2, in [0, 1]; the diagonal is 0.
    """
    p = probs[:, :, None, :]                     # (M, K+1, 1, C)
    q = probs[:, None, :, :]                     # (M, 1, K+1, C)
    return jensen_shannon(p, q)


def agreement_from_d(d):
    """A_k from a precomputed pairwise-JSD matrix 'd' (M, K+1, K+1).

    A_k = -(1/K) sum_{j != k} JSD(p_k, p_j). The diagonal of 'd' is 0, so
    summing the whole row and dividing by K (= n - 1) is the same thing.
    """
    n = d.shape[-1]
    return -d.sum(axis=-1) / (n - 1)


def stability_from_d(d, m=1, ladder_len=None):
    """S_k from a precomputed pairwise-JSD matrix 'd' (M, K+1, K+1).

    S_k = - mean JSD from view k to the views within 'm' ladder rungs (k
    itself excluded).

    Args:
        d: (..., K+1, K+1) pairwise-JSD array.
        m: neighbourhood radius in ladder rungs.
        ladder_len: number of leading views that form an ordered filter
            ladder (view 0 = original, 1..ladder_len-1 = increasing filter
            strength). Defaults to the whole profile (a single ladder).
            Views at index >= ladder_len have no
            rung ordering (e.g. spatial-TTA crop/flip views concatenated
            after the ladder, see scripts/build_combined_cache.py) get the
            image's mean ladder S: 'neighbouring filtering levels' isn't
            defined for them, so they are kept neutral: with the mean, their
            per-image z-score is exactly 0. (Since every ladder S_k <= 0, a
            constant 0 would instead rank them as the most stable views.)
    """
    n = d.shape[-1]
    L = n if ladder_len is None else ladder_len
    out = np.zeros(d.shape[:-1], dtype=d.dtype)
    for k in range(L):
        lo, hi = max(0, k - m), min(L - 1, k + m)
        idx = [j for j in range(lo, hi + 1) if j != k]
        if idx:                      # a 1-view "ladder" has no neighbours: S = 0
            out[..., k] = -d[..., k, idx].mean(axis=-1)
    if L < n:
        out[..., L:] = out[..., :L].mean(axis=-1, keepdims=True)
    return out


def agreement_to_origin(probs):
    """A0_k = -JSD(p_k, p_0): agreement with the original view only.

    The proposal's planned ablation for a specific failure mode of
    'agreement()': nested low-passes are correlated by construction, so a
    heavily-filtered view can find itself in a self-agreeing cluster with
    other heavily-filtered views even when all of them are wrong (measured:
    per-view correctness correlates more with this signal than with
    agreement-to-all-views, e.g. +0.355 vs +0.281 for the weather family on
    ResNet-18/fixed-ladder). Anchoring agreement to the unfiltered view
    removes that cluster effect. A0_0 is always 0 (a view compared to
    itself), same convention as 'agreement_from_d's diagonal.

    Args:
        probs: (..., K+1, C) array; index -2 is the view axis, view 0 first.

    Returns:
        (..., K+1) array.
    """
    p0 = probs[..., :1, :]
    return -jensen_shannon(probs, p0)


def zscore(x, axis=-1, eps=1e-8):
    """Standardise 'x' to mean 0, std 1 along 'axis'.

    Applied per image to each signal so C_k, A_k, S_k (which have different
    natural ranges) become comparable before they are weighted. A near-constant
    signal (std < eps) is only recentred, not scaled.
    """
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / np.where(sd < eps, 1.0, sd)


def _default_device():
    """'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _kl_t(a, b):
    """torch equivalent of 'kl', used inside the GPU path of 'profile_signals'."""
    a = a + EPS
    b = b + EPS
    return torch.sum(a * torch.log2(a / b), dim=-1)


def _jsd_t(p, q):
    """torch equivalent of 'jensen_shannon'."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_t(p, m) + 0.5 * _kl_t(q, m)


def _pairwise_jsd_t(probs):
    """torch equivalent of 'pairwise_jsd'; probs is (M, K+1, C)."""
    p = probs[:, :, None, :]
    q = probs[:, None, :, :]
    return _jsd_t(p, q)


def _agreement_from_d_t(d):
    """torch equivalent of 'agreement_from_d'."""
    n = d.shape[-1]
    return -d.sum(dim=-1) / (n - 1)


def _stability_from_d_t(d, m=1, ladder_len=None):
    """torch equivalent of 'stability_from_d'."""
    n = d.shape[-1]
    L = n if ladder_len is None else ladder_len
    out = torch.zeros(d.shape[:-1], dtype=d.dtype, device=d.device)
    for k in range(L):
        lo, hi = max(0, k - m), min(L - 1, k + m)
        idx = [j for j in range(lo, hi + 1) if j != k]
        if idx:
            out[..., k] = -d[..., k, idx].mean(dim=-1)
    if L < n:
        out[..., L:] = out[..., :L].mean(dim=-1, keepdim=True)
    return out


def _agreement_to_origin_t(probs):
    """torch equivalent of 'agreement_to_origin'."""
    p0 = probs[..., :1, :]
    return -_jsd_t(probs, p0)


def profile_signals(probs, m=1, standardize=True, chunk=1024, agreement="all",
                    ladder_len=None, device=None):
    """Compute C_k, A_k, S_k for every image in a condition.

    The pairwise-JSD step (an (chunk, K+1, K+1, C) intermediate per chunk) runs
    as torch tensors on 'device' -- GPU when available -- since it dominates
    the runtime of a full tune/evaluate pass; everything else is cheap enough
    to stay on the host. Inputs and outputs are NumPy arrays regardless of
    'device'.

    Args:
        probs: (N, K+1, C) array of softmax outputs.
        m: neighbourhood radius for S_k (ladder rungs).
        standardize: if True, z-score each signal per image (see 'zscore').
        chunk: number of images processed per batch, to bound the
            (chunk, K+1, K+1, C) JSD intermediate.
        agreement: "all" (agreement_from_d, the default definition) or
            "origin" (agreement_to_origin, the proposal's planned ablation
            for the self-agreeing-cluster failure mode).
        ladder_len: passed to 'stability_from_d' -- see there. None (default)
            treats the whole profile as one ordered ladder.
        device: torch device for the pairwise-JSD step. None (default) picks
            CUDA if available, else CPU.

    Returns:
        (C, A, S), each a (N, K+1) float32 array.
    """
    if agreement not in ("all", "origin"):
        raise ValueError(f"agreement must be 'all' or 'origin', got {agreement!r}")
    if device is None:
        device = _default_device()
    probs_np = np.ascontiguousarray(np.asarray(probs, dtype=np.float32))
    n = probs_np.shape[0]
    probs_t = torch.from_numpy(probs_np).to(device)

    C = probs_t.max(dim=-1).values
    A = torch.empty(probs_t.shape[:2], dtype=probs_t.dtype, device=device)
    S = torch.empty(probs_t.shape[:2], dtype=probs_t.dtype, device=device)
    for i in range(0, n, chunk):
        chunk_probs = probs_t[i:i + chunk]
        d = _pairwise_jsd_t(chunk_probs)
        A[i:i + chunk] = (_agreement_to_origin_t(chunk_probs) if agreement == "origin"
                          else _agreement_from_d_t(d))
        S[i:i + chunk] = _stability_from_d_t(d, m=m, ladder_len=ladder_len)

    C, A, S = C.cpu().numpy(), A.cpu().numpy(), S.cpu().numpy()
    if standardize:
        C, A, S = zscore(C), zscore(A), zscore(S)
    return C, A, S


def reliability(C, A, S, alpha=1.0, beta=1.0, gamma=1.0):
    """Linear combination of the signals: r = alpha*C + beta*A + gamma*S.

    Args:
        C, A, S: (N, K+1) arrays.
        alpha, beta, gamma: weight per signal (0 drops it).

    Returns:
        (N, K+1) array of reliability scores.
    """
    return alpha * C + beta * A + gamma * S


def fusion_weights(scores, tau=1.0):
    """Row-wise softmax of 'scores / tau' over the view axis.

    Args:
        scores: (N, K+1) array.
        tau: temperature; small -> near-hard selection of one view.

    Returns:
        (N, K+1) array of weights, each row non-negative and summing to 1.
    """
    z = scores / tau
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fuse(probs, weights):
    """Per-image weighted average of the view probability vectors.

    Args:
        probs: (N, K+1, C) array.
        weights: (N, K+1) array (rows should sum to 1).

    Returns:
        (N, C) array of fused predictions p*.
    """
    probs = np.asarray(probs, dtype=np.float32)
    return np.einsum("nkc,nk->nc", probs, weights.astype(np.float32))
