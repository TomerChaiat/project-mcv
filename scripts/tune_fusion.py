"""CUDA-accelerated grid search for fusion hyperparameters.

This script searches the *same* grid and writes the same tuning JSON schema,
but removes the two dominant sources of wasted work in the legacy tuner:

1. Pairwise JSD is computed once per cached condition.  From that one matrix we
   derive both agreement variants and every requested stability radius.
2. Every (config, condition) accuracy is computed exactly once.  The four LOFO
   folds and the global winner are then just means over columns of that score
   matrix, instead of re-fusing the same probabilities five times.

The expensive fusion/argmax work is batched on CUDA when available.  Cached
probabilities are moved to VRAM one condition at a time and reused for every
configuration before being released, so VRAM stays bounded while the GPU is
kept busy.

The implementation computes both C/A/S signals and grid scoring in PyTorch on CUDA when available.  The NumPy signal path is retained as
a parity/debug fallback (`--signal-backend numpy`).

The GPU JSD kernel evaluates only the upper-triangle view pairs, in small pair
batches, then mirrors them.  This avoids materialising the full
(N, V, V, C) broadcast tensor while preserving the project's base-2 JSD
definition.

Examples
--------
    python scripts/tune_fusion.py --models resnet18 \
        --cache-dir cache_fixed_fresh --tag cache_fixed_fresh \
        --out results_fast --tune-severities 1 3 5

    # Larger config batches use more VRAM but can be faster on a 12 GB GPU.
    python scripts/tune_fusion.py --config-batch 96 ...
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import analysis
from src.metrics import FusionConfig
from src.evaluation import device_from_arg, signal_variants_torch
from src.splits import CORRUPTION_FAMILY, leave_one_family_out, tune_test_masks


# Canonical fusion grid used throughout the experiments.
ALPHAS = (0.0, 1.0)
BETAS = (0.0, 0.25, 0.5, 1.0)
GAMMAS = (0.0, 0.25, 0.5, 1.0)
TAUS = (1.0, 0.5, 0.2, 0.1, 0.05)
STANDARDIZE = (True, False)
M_VALUES = (1, 2, 3)
AGREEMENT_MODES = ("all", "origin")


def grid(m: int, agreement: str = "all"):
    """Yield configs in the exact same order as the legacy tuner."""
    for a, b, g, tau, std in itertools.product(
        ALPHAS, BETAS, GAMMAS, TAUS, STANDARDIZE
    ):
        if a == 0.0 and b == 0.0 and g == 0.0:
            continue
        yield FusionConfig(
            alpha=a,
            beta=b,
            gamma=g,
            tau=tau,
            standardize=std,
            m=m,
            agreement=agreement,
        )


@dataclass
class CandidateBlock:
    key: tuple[int, str]
    start: int
    stop: int
    configs: list[FusionConfig]


def build_candidates(m_values, agreement_modes):
    """Flat candidate list + contiguous ranges for every (m, agreement)."""
    flat: list[FusionConfig] = []
    blocks: list[CandidateBlock] = []
    for m in m_values:
        for agreement in agreement_modes:
            cfgs = list(grid(m, agreement))
            start = len(flat)
            flat.extend(cfgs)
            blocks.append(CandidateBlock((m, agreement), start, len(flat), cfgs))
    return flat, blocks


def _zscore_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Exact project z-score convention, kept local to avoid repeated calls."""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return ((x - mu) / np.where(sd < eps, 1.0, sd)).astype(np.float32, copy=False)


def _to_device_float32(x, device: torch.device) -> torch.Tensor:
    """Return x as contiguous float32 tensor on device without needless copies."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    return torch.from_numpy(np.ascontiguousarray(x)).to(
        device=device, dtype=torch.float32
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def signal_variants_one_pass(
    probs: np.ndarray,
    m_values,
    agreement_modes,
    *,
    ladder_len=None,
    chunk=256,
):
    """Compute every requested C/A/S variant with one pairwise-JSD pass.

    Returns raw and per-image-standardized arrays.  Unlike the legacy tuner,
    probabilities themselves are *not* duplicated six times.
    """
    probs = np.asarray(probs, dtype=np.float32)
    n, v, _ = probs.shape

    C = analysis.confidence(probs).astype(np.float32, copy=False)
    A = {a: np.empty((n, v), dtype=np.float32) for a in agreement_modes}
    S = {m: np.empty((n, v), dtype=np.float32) for m in m_values}

    for i in range(0, n, chunk):
        p = probs[i : i + chunk]
        d = analysis.pairwise_jsd(p)
        if "all" in A:
            A["all"][i : i + len(p)] = analysis.agreement_from_d(d)
        if "origin" in A:
            # Reuse the same JSD matrix instead of recomputing JSD(p_k, p_0).
            A["origin"][i : i + len(p)] = -d[:, :, 0]
        for m in m_values:
            S[m][i : i + len(p)] = analysis.stability_from_d(
                d, m=m, ladder_len=ladder_len
            )

    raw = {}
    std = {}
    C_std = _zscore_np(C)
    A_std = {k: _zscore_np(x) for k, x in A.items()}
    S_std = {k: _zscore_np(x) for k, x in S.items()}

    for m in m_values:
        for agreement in agreement_modes:
            raw[(m, agreement)] = (C, A[agreement], S[m])
            std[(m, agreement)] = (C_std, A_std[agreement], S_std[m])
    return raw, std


def score_block_torch(
    probs_t: torch.Tensor,
    labels_t: torch.Tensor,
    raw_triplet,
    std_triplet,
    configs: list[FusionConfig],
    *,
    device: torch.device,
    config_batch: int,
) -> np.ndarray:
    """Accuracy of every config in one (m, agreement) block.

    The fused class distributions are materialized only for `config_batch`
    configurations at once.  With N~=2500, V=8, C=1000, a batch of 64 uses
    about 640 MB for the largest fused tensor, which fits comfortably on a
    12 GB card while leaving room for inputs and CUDA workspaces.
    """
    C_raw, A_raw, S_raw = [_to_device_float32(x, device) for x in raw_triplet]
    C_std, A_std, S_std = [_to_device_float32(x, device) for x in std_triplet]

    out = np.empty(len(configs), dtype=np.float32)
    n = probs_t.shape[0]

    for lo in range(0, len(configs), config_batch):
        sub = configs[lo : lo + config_batch]
        b = len(sub)

        a = torch.tensor([c.alpha for c in sub], device=device, dtype=torch.float32)[:, None, None]
        beta = torch.tensor([c.beta for c in sub], device=device, dtype=torch.float32)[:, None, None]
        g = torch.tensor([c.gamma for c in sub], device=device, dtype=torch.float32)[:, None, None]
        tau = torch.tensor([c.tau for c in sub], device=device, dtype=torch.float32)[:, None, None]
        use_std = torch.tensor(
            [c.standardize for c in sub], device=device, dtype=torch.bool
        )[:, None, None]

        raw_scores = (
            a * C_raw[None] + beta * A_raw[None] + g * S_raw[None]
        )
        std_scores = (
            a * C_std[None] + beta * A_std[None] + g * S_std[None]
        )
        scores = torch.where(use_std, std_scores, raw_scores) / tau
        weights = torch.softmax(scores, dim=-1)  # (B, N, V)

        # bmm works condition-wise: (N,B,V) @ (N,V,C) -> (N,B,C)
        fused = torch.bmm(weights.permute(1, 0, 2).contiguous(), probs_t)
        pred = fused.argmax(dim=-1)  # (N, B)
        acc = (pred == labels_t[:, None]).float().mean(dim=0)
        out[lo : lo + b] = acc.detach().cpu().numpy()

        # Drop the largest temporaries before the next batch.
        del raw_scores, std_scores, scores, weights, fused, pred, acc

    return out


def score_condition(
    probs: np.ndarray,
    labels: np.ndarray,
    blocks: list[CandidateBlock],
    *,
    m_values,
    agreement_modes,
    ladder_len,
    signal_chunk,
    signal_pair_batch,
    signal_backend,
    device,
    config_batch,
):
    """Return all candidate accuracies + baselines for one condition."""
    labels = np.asarray(labels, dtype=np.int64)

    # Move the cached probabilities once.  float16 caches are expanded directly
    # into float32 VRAM, avoiding a second full float32 host copy.
    probs_t = torch.from_numpy(np.ascontiguousarray(probs)).to(
        device=device, dtype=torch.float32
    )
    labels_t = torch.from_numpy(labels).to(device=device, dtype=torch.long)

    use_torch_signals = signal_backend == "torch"
    if signal_backend == "auto":
        use_torch_signals = device.type == "cuda"

    _sync(device)
    t0 = time.time()
    with torch.inference_mode():
        if use_torch_signals:
            raw, std = signal_variants_torch(
                probs_t,
                m_values,
                agreement_modes,
                ladder_len=ladder_len,
                chunk=signal_chunk,
                pair_batch=signal_pair_batch,
            )
        else:
            # Debug/parity fallback.  This intentionally mirrors the previous
            # GPU tuner and therefore creates a host float32 view.
            probs32 = np.asarray(probs, dtype=np.float32)
            raw, std = signal_variants_one_pass(
                probs32,
                m_values,
                agreement_modes,
                ladder_len=ladder_len,
                chunk=signal_chunk,
            )
    _sync(device)
    signal_seconds = time.time() - t0

    all_acc = np.empty(blocks[-1].stop if blocks else 0, dtype=np.float32)
    _sync(device)
    t1 = time.time()
    with torch.inference_mode():
        for block in blocks:
            all_acc[block.start : block.stop] = score_block_torch(
                probs_t,
                labels_t,
                raw[block.key],
                std[block.key],
                block.configs,
                device=device,
                config_batch=config_batch,
            )
    _sync(device)
    score_seconds = time.time() - t1

    # Baselines on the same device; transfer only two scalar results.
    with torch.inference_mode():
        orig_acc = float(
            (probs_t[:, 0].argmax(dim=-1) == labels_t).float().mean().item()
        )
        equal_acc = float(
            (probs_t.mean(dim=1).argmax(dim=-1) == labels_t).float().mean().item()
        )

    del probs_t, labels_t, raw, std
    if not use_torch_signals and "probs32" in locals():
        del probs32

    return all_acc, orig_acc, equal_acc, signal_seconds, score_seconds


def mean_for_corruptions(matrix, cond_keys, corruptions, severities):
    cols = [
        i
        for i, (c, s) in enumerate(cond_keys)
        if c in corruptions and s in severities
    ]
    if not cols:
        return None
    return matrix[:, cols].mean(axis=1)


def baseline_for_corruptions(base_meta, corruptions, severities):
    vals = [
        base_meta[(c, s)]
        for c in corruptions
        for s in severities
        if (c, s) in base_meta
    ]
    if not vals:
        return float("nan"), float("nan")
    return (
        float(np.mean([x[0] for x in vals])),
        float(np.mean([x[1] for x in vals])),
    )


def main():
    from src import cache

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"], choices=["resnet18", "deit"])
    ap.add_argument("--tune-severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--m-values", nargs="+", type=int, default=list(M_VALUES))
    ap.add_argument("--agreement-modes", nargs="+", choices=["all", "origin"], default=list(AGREEMENT_MODES))
    ap.add_argument("--ladder-len", type=int, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="auto", help="auto, cuda, cpu, or explicit torch device")
    ap.add_argument("--config-batch", type=int, default=64, help="configs fused simultaneously; raise if VRAM allows")
    ap.add_argument("--signal-chunk", type=int, default=256,
                    help="images per JSD chunk (CPU or GPU)")
    ap.add_argument("--signal-pair-batch", type=int, default=32,
                    help="unique view pairs evaluated together by the torch JSD kernel")
    ap.add_argument("--signal-backend", choices=["auto", "torch", "numpy"], default="auto",
                    help="auto uses torch on CUDA and NumPy otherwise")
    ap.add_argument(
        "--tf32",
        action="store_true",
        help="allow TF32 matmul on NVIDIA for extra speed; off by default for closer numerical parity",
    )
    args = ap.parse_args()

    if args.config_batch < 1:
        raise SystemExit("--config-batch must be >= 1")

    cache.CACHE_ROOT = Path(args.cache_dir)
    tag = args.tag or (Path(args.cache_dir).name if args.cache_dir != "cache" else None)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_from_arg(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        # cuDNN is not central here, but keeping the same policy is clearer.
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    candidates, blocks = build_candidates(args.m_values, args.agreement_modes)
    print(
        f"Grid: {len(candidates)} total configs | "
        f"{len(blocks)} (m,agreement) blocks | device={device} | "
        f"config_batch={args.config_batch} | signal_chunk={args.signal_chunk} | "
        f"signal_pair_batch={args.signal_pair_batch} | signal_backend={args.signal_backend} | "
        f"tf32={args.tf32} | cache={args.cache_dir}"
    )
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        t_all = time.time()

        cond_keys = []
        acc_columns = []
        base_meta = {}
        signal_total = 0.0
        scoring_total = 0.0

        for corruption in CORRUPTION_FAMILY:
            for sev in args.tune_severities:
                if not cache.has_condition(model_name, corruption, sev):
                    print(f"  missing cache: {corruption}/{sev}, skipping")
                    continue

                d = cache.load_condition(model_name, corruption, sev)
                tune_mask, _ = tune_test_masks(d["paths"], args.frac_tune, args.split_seed)
                probs = d["probs"][tune_mask]
                labels = d["labels"][tune_mask]

                t0 = time.time()
                accs, orig, equal, sig_s, score_s = score_condition(
                    probs,
                    labels,
                    blocks,
                    m_values=args.m_values,
                    agreement_modes=args.agreement_modes,
                    ladder_len=args.ladder_len,
                    signal_chunk=args.signal_chunk,
                    signal_pair_batch=args.signal_pair_batch,
                    signal_backend=args.signal_backend,
                    device=device,
                    config_batch=args.config_batch,
                )
                elapsed = time.time() - t0
                cond_keys.append((corruption, sev))
                acc_columns.append(accs)
                base_meta[(corruption, sev)] = (orig, equal)
                signal_total += sig_s
                scoring_total += score_s
                print(
                    f"  {corruption:18s}/{sev}: n={len(labels):4d}  "
                    f"signals {sig_s:5.1f}s  grid {score_s:5.1f}s  total {elapsed:5.1f}s"
                )

        if not acc_columns:
            print("  no tune conditions; skipping model")
            continue

        # (num_candidates, num_conditions), tiny compared with probability caches.
        acc_matrix = np.stack(acc_columns, axis=1)
        print(
            f"  scored {len(cond_keys)} conditions x {len(candidates)} configs ONCE "
            f"in {time.time() - t_all:.0f}s "
            f"(signals {signal_total:.0f}s, GPU grid {scoring_total:.0f}s)"
        )

        record = {
            "model": model_name,
            "cache_dir": args.cache_dir,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "grid_size": len(candidates),
            "tune_severities": args.tune_severities,
            "split": {"frac_tune": args.frac_tune, "seed": args.split_seed},
            "m_values": args.m_values,
            "ladder_len": args.ladder_len,
            "agreement_modes": args.agreement_modes,
            "held_out": {},
            "global": {},
            "backend": {
                "name": "fast_torch_grid",
                "device": str(device),
                "config_batch": args.config_batch,
                "signal_chunk": args.signal_chunk,
                "signal_pair_batch": args.signal_pair_batch,
                "signal_backend": args.signal_backend,
                "tf32": bool(args.tf32),
            },
        }

        def tune_on(name, tune_corr):
            means = mean_for_corruptions(
                acc_matrix, cond_keys, tune_corr, args.tune_severities
            )
            if means is None:
                print(f"  {name:14s}: no cached tune conditions, skipping")
                return None
            # np.argmax returns the first maximum, matching the old nested loops'
            # strict '>' tie-breaking because candidate order is identical.
            idx = int(np.argmax(means))
            cfg = candidates[idx]
            acc = float(means[idx])
            orig, equal = baseline_for_corruptions(
                base_meta, tune_corr, args.tune_severities
            )
            print(
                f"  {name:14s}: fused {acc*100:.1f}%  "
                f"(orig {orig*100:.1f}, equal {equal*100:.1f})  "
                f"a={cfg.alpha} b={cfg.beta} g={cfg.gamma} tau={cfg.tau} "
                f"std={cfg.standardize} m={cfg.m} agree={cfg.agreement}"
            )
            return {
                "config": cfg.as_dict(),
                "mean_fused_acc": acc,
                "mean_orig_acc": orig,
                "mean_equal_acc": equal,
            }

        for held, tune_corr, _ in leave_one_family_out():
            entry = tune_on(f"hold out {held}", tune_corr)
            if entry:
                record["held_out"][held] = entry
        record["global"] = tune_on("global", list(CORRUPTION_FAMILY)) or {}

        name = (
            f"tuning_grid_{model_name}"
            + (f"_{tag}" if tag else "")
            + f"_{record['timestamp']}.json"
        )
        out = out_dir / name
        out.write_text(json.dumps(record, indent=2))
        print("  saved", out)


if __name__ == "__main__":
    main()
