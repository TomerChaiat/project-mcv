"""CE direction fitting + accuracy scale selection + calibration.

The implementation keeps the scientific protocol while using the GPU and shared caches efficiently:

1. CE objective/gradient reductions run in torch on CUDA (SciPy still owns the
   tiny L-BFGS-B parameter vector and bounds).
2. Every tune-half probability tensor is loaded from the compressed cache once
   and kept in CPU RAM, so scale and temperature selection never re-open .npz
   files.
3. Multiple CE variants (base, view-bias, unconstrained) can be run in one
   process and share the same GPU signal precomputation / RAM cache.

No model inference or image recaching is performed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import cache
from src.ce_fusion import DEFAULT_EPS, fit_coefficients, fit_coefficients_torch
from src.metrics import FusionConfig
from src.splits import CORRUPTION_FAMILY, leave_one_family_out, tune_test_masks
from scripts.tune_fusion import AGREEMENT_MODES, M_VALUES
from src.evaluation import device_from_arg, signal_variants_torch


VARIANTS = {
    "base": ("ce", False, False),
    "bias": ("ce_bias", True, False),
    "unconstrained": ("ce_unconstrained", False, True),
}


def latest_grid_tuning(model_name, tag, results_dir):
    patterns = [
        Path(results_dir) / f"tuning_grid_{model_name}_{tag}_*.json",
        Path(results_dir) / f"tuning_fast_{model_name}_{tag}_*.json",  # development-era compatibility
        Path(results_dir) / f"tuning_{model_name}_{tag}_*.json",
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(str(p)))
    return sorted(files)[-1] if files else None


def load_grid_record(spec, model_name, tag, results_dir):
    if not spec:
        return None, None
    path = latest_grid_tuning(model_name, tag, results_dir) if spec == "auto" else spec
    if not path:
        print("  grid start: none found")
        return None, None
    rec = json.loads(Path(path).read_text())
    print("  grid start:", path)
    return rec, path


def grid_effective_start(grid_record, held_family=None):
    """Convert grid alpha/beta/gamma,tau into the equivalent tau=1 vector."""
    if not grid_record:
        return None
    try:
        entry = grid_record["global"] if held_family is None else grid_record["held_out"][held_family]
        c = entry["config"]
        tau = float(c.get("tau", 1.0))
        return np.array([c["alpha"], c["beta"], c["gamma"]], dtype=float) / tau
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _condition_keys(corruptions, severities, by_key, key):
    return [
        (c, s)
        for c in corruptions
        for s in severities
        if (c, s) in by_key[key]
    ]


def concatenate_conditions(by_key, conditions, corruptions, severities, key):
    """Build one CE fit tensor from already-cached RAM arrays."""
    cond_keys = _condition_keys(corruptions, severities, by_key, key)
    if not cond_keys:
        return None, None
    X = np.concatenate([by_key[key][cs] for cs in cond_keys], axis=0)
    p_true = np.concatenate([conditions[cs]["p_true"] for cs in cond_keys], axis=0)
    return X, p_true


def baseline_means(base_meta, corruptions, severities):
    recs = [
        base_meta[(c, s)]
        for c in corruptions
        for s in severities
        if (c, s) in base_meta
    ]
    if not recs:
        return float("nan"), float("nan")
    return (
        float(np.mean([r["orig_acc"] for r in recs])),
        float(np.mean([r["equal_acc"] for r in recs])),
    )


def precompute_all_conditions(
    model_name,
    tune_severities,
    keys,
    *,
    frac_tune,
    split_seed,
    ladder_len,
    signal_chunk,
    signal_pair_batch,
    device,
    resident_probs="auto",
):
    """Load each tune condition once and retain reusable data.

    On CUDA, ``resident_probs=auto`` keeps the float16 probability cache in
    VRAM when at least 8 GiB is free.  The 8-view tune half is ~3 GiB and the
    15-view combined tune half is ~5.7 GiB, both suitable for a 12 GiB card
    with room left for the per-condition fp32 fusion temporaries.  This removes
    the repeated 40--75 MB host-to-device copy for every condition, holdout,
    scale grid, temperature grid and CE variant.  Use ``cpu`` to force the
    older RAM-resident path if VRAM is shared with another process.
    """
    by_key = {key: {} for key in keys}
    conditions = {}
    base_meta = {}
    n_conditions = 0
    n_prob_bytes = 0
    t0 = time.time()

    m_values = sorted({k[0] for k in keys})
    agreements = sorted({k[1] for k in keys})

    if resident_probs == "cuda" and device.type != "cuda":
        raise ValueError("resident_probs=cuda requires a CUDA device")
    keep_probs_cuda = resident_probs == "cuda"
    if resident_probs == "auto" and device.type == "cuda":
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        keep_probs_cuda = free_bytes >= 8 * 2**30
    resident_where = "cuda-fp16" if keep_probs_cuda else "cpu-fp16"

    for corruption in CORRUPTION_FAMILY:
        for sev in tune_severities:
            if not cache.has_condition(model_name, corruption, sev):
                print(f"  missing cache: {corruption}/{sev}, skipping")
                continue

            d = cache.load_condition(model_name, corruption, sev)
            tune_mask, _ = tune_test_masks(d["paths"], frac_tune, split_seed)
            # Preserve the on-disk float16 probabilities in RAM.  Convert only
            # the current condition to fp32 when it enters the GPU.
            probs = np.ascontiguousarray(d["probs"][tune_mask])
            labels = np.asarray(d["labels"][tune_mask], dtype=np.int64)
            n_prob_bytes += probs.nbytes

            probs_t = torch.from_numpy(probs).to(device=device, dtype=torch.float32)
            labels_t = torch.from_numpy(labels).to(device=device, dtype=torch.long)
            with torch.inference_mode():
                _raw, std = signal_variants_torch(
                    probs_t,
                    m_values,
                    agreements,
                    ladder_len=ladder_len,
                    chunk=signal_chunk,
                    pair_batch=signal_pair_batch,
                )
                p_true_t = probs_t.gather(
                    2, labels_t[:, None, None].expand(-1, probs_t.shape[1], 1)
                ).squeeze(2)
                p_true = p_true_t.cpu().numpy().astype(np.float32, copy=False)

                for key in keys:
                    C, A, S = std[key]
                    by_key[key][(corruption, sev)] = (
                        torch.stack([C, A, S], dim=-1)
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )

                base_meta[(corruption, sev)] = {
                    "orig_acc": float((probs_t[:, 0].argmax(-1) == labels_t).float().mean().item()),
                    "equal_acc": float((probs_t.mean(1).argmax(-1) == labels_t).float().mean().item()),
                }

            prob_store = probs_t.to(dtype=torch.float16) if keep_probs_cuda else probs
            conditions[(corruption, sev)] = {
                "probs": prob_store,
                "labels": labels,
                "labels_t": labels_t if keep_probs_cuda else None,
                "p_true": p_true,
            }
            # labels_t/prob_store remain referenced only in CUDA-resident mode.
            del probs_t, d
            if not keep_probs_cuda:
                del labels_t
            n_conditions += 1

    elapsed = time.time() - t0
    print(
        f"  precomputed+resident-cached {n_conditions} tune conditions on {device} in "
        f"{elapsed:.1f}s  (prob cache {n_prob_bytes / 2**30:.2f} GiB, {resident_where})"
    )
    return by_key, conditions, base_meta


def _condition_to_device(cond, device):
    if torch.is_tensor(cond["probs"]):
        probs_t = cond["probs"].to(device=device, dtype=torch.float32)
    else:
        probs_t = torch.from_numpy(cond["probs"]).to(device=device, dtype=torch.float32)
    if cond.get("labels_t") is not None:
        labels_t = cond["labels_t"].to(device=device, dtype=torch.long)
    else:
        labels_t = torch.from_numpy(cond["labels"]).to(device=device, dtype=torch.long)
    return probs_t, labels_t


def tune_scale_accuracy(
    conditions,
    by_key,
    key,
    theta,
    bias,
    corruptions,
    severities,
    scales,
    *,
    device,
    batch_scales=32,
):
    """Choose score scale by mean condition accuracy, without cache re-reads."""
    scales = np.asarray(scales, dtype=np.float32)
    acc_sum = np.zeros(len(scales), dtype=np.float64)
    n_conditions = 0
    theta_t = torch.tensor(theta, device=device, dtype=torch.float32)
    bias_t = torch.tensor(bias, device=device, dtype=torch.float32)

    for cs in _condition_keys(corruptions, severities, by_key, key):
        cond = conditions[cs]
        probs_t, labels_t = _condition_to_device(cond, device)
        X_t = torch.from_numpy(by_key[key][cs]).to(device=device, dtype=torch.float32)

        with torch.inference_mode():
            base = torch.einsum("nvd,d->nv", X_t, theta_t) + bias_t[None, :]
            cond_acc = []
            for lo in range(0, len(scales), batch_scales):
                sb = torch.as_tensor(
                    scales[lo:lo + batch_scales], device=device, dtype=torch.float32
                )[:, None, None]
                weights = torch.softmax(sb * base[None, :, :], dim=-1)
                fused = torch.bmm(weights.permute(1, 0, 2).contiguous(), probs_t)
                pred = fused.argmax(-1)
                acc = (pred == labels_t[:, None]).float().mean(0)
                cond_acc.append(acc.cpu().numpy())
            acc_sum += np.concatenate(cond_acc)
        n_conditions += 1
        del probs_t, labels_t, X_t

    if n_conditions == 0:
        raise RuntimeError("no conditions available for scale tuning")
    means = acc_sum / n_conditions
    idx = int(np.argmax(means))
    return float(scales[idx]), float(means[idx]), means


def tune_output_temperature(
    conditions,
    by_key,
    key,
    theta,
    bias,
    score_scale,
    corruptions,
    severities,
    temperatures,
    *,
    device,
    eps,
    temperature_batch=49,
):
    """Choose post-fusion T by tune-half NLL, reusing the RAM probability cache."""
    temperatures = np.asarray(temperatures, dtype=np.float32)
    nll_sums = np.zeros(len(temperatures), dtype=np.float64)
    n_total = 0
    theta_t = torch.tensor(theta, device=device, dtype=torch.float32)
    bias_t = torch.tensor(bias, device=device, dtype=torch.float32)

    for cs in _condition_keys(corruptions, severities, by_key, key):
        cond = conditions[cs]
        probs_t, labels_t = _condition_to_device(cond, device)
        X_t = torch.from_numpy(by_key[key][cs]).to(device=device, dtype=torch.float32)

        with torch.inference_mode():
            scores = torch.einsum("nvd,d->nv", X_t, theta_t) + bias_t[None, :]
            weights = torch.softmax(float(score_scale) * scores, dim=-1)
            fused = torch.bmm(weights[:, None, :], probs_t).squeeze(1)

            # Preserve exact zeros as -inf.  Floor only the true class, matching
            # src.metrics.nll and the CE fitting convention.
            logp = torch.where(
                fused > 0,
                torch.log(fused),
                torch.full_like(fused, -torch.inf),
            )
            true_prob = fused.gather(1, labels_t[:, None]).squeeze(1)
            true_logp = torch.log(torch.clamp(true_prob, min=float(eps)))

            for lo in range(0, len(temperatures), temperature_batch):
                tb = torch.as_tensor(
                    temperatures[lo:lo + temperature_batch],
                    device=device,
                    dtype=torch.float32,
                )[:, None, None]
                z = logp[None, :, :] / tb
                log_den = torch.logsumexp(z, dim=-1)
                true_scaled = true_logp[None, :] / tb.squeeze(-1)
                sums = (log_den - true_scaled).sum(dim=1)
                nll_sums[lo:lo + len(sums)] += sums.cpu().numpy()

        n_total += len(cond["labels"])
        del probs_t, labels_t, X_t

    if n_total == 0:
        raise RuntimeError("no conditions available for temperature tuning")
    means = nll_sums / n_total
    idx = int(np.argmin(means))
    return float(temperatures[idx]), float(means[idx]), means


def _variant_list(args):
    if args.variants:
        # Preserve user order while removing duplicates.
        return list(dict.fromkeys(args.variants))
    # Backward-compatible single-variant behavior for old commands.
    if args.view_bias and args.allow_negative:
        raise SystemExit("legacy flags do not define bias+unconstrained; use --variants")
    if args.view_bias:
        return ["bias"]
    if args.allow_negative:
        return ["unconstrained"]
    return ["base"]


def _fit_one_key(
    X,
    p_true,
    *,
    starts,
    args,
    view_bias,
    allow_negative,
    device,
):
    backend = args.optimizer_backend
    if backend == "auto":
        backend = "torch" if device.type == "cuda" else "numpy"

    common = dict(
        l2=args.l2,
        bias_l2=args.bias_l2,
        eps=args.eps,
        starts=starts,
        random_restarts=args.random_restarts,
        seed=args.optimizer_seed,
        maxiter=args.maxiter,
        nonnegative=not allow_negative,
        view_bias=view_bias,
    )
    if backend == "torch":
        return fit_coefficients_torch(
            X,
            p_true,
            device=device,
            dtype=args.ce_dtype,
            **common,
        )
    return fit_coefficients(X, p_true, **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"], choices=["resnet18", "deit"])
    ap.add_argument("--cache-dir", default="cache_fixed")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tune-severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--m-values", nargs="+", type=int, default=list(M_VALUES))
    ap.add_argument("--agreement-modes", nargs="+", choices=["all", "origin"], default=list(AGREEMENT_MODES))
    ap.add_argument("--ladder-len", type=int, default=None)
    ap.add_argument("--grid-tuning", default="auto")
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--bias-l2", type=float, default=1e-3)
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS)
    ap.add_argument("--random-restarts", type=int, default=4)
    ap.add_argument("--optimizer-seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=500)

    # Old single-variant flags remain accepted.
    ap.add_argument("--allow-negative", action="store_true")
    ap.add_argument("--view-bias", action="store_true")
    # New mode: one process, one signal/RAM precompute, many CE variants.
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None)

    ap.add_argument("--optimizer-backend", choices=["auto", "torch", "numpy"], default="auto")
    ap.add_argument("--ce-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--no-scale-tuning", action="store_true")
    ap.add_argument("--scale-min", type=float, default=0.25)
    ap.add_argument("--scale-max", type=float, default=32.0)
    ap.add_argument("--scale-points", type=int, default=15)
    ap.add_argument("--scale-batch", type=int, default=32)
    ap.add_argument("--no-output-temperature", action="store_true")
    ap.add_argument("--temperature-min", type=float, default=0.25)
    ap.add_argument("--temperature-max", type=float, default=4.0)
    ap.add_argument("--temperature-points", type=int, default=49)
    ap.add_argument("--temperature-batch", type=int, default=49)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--signal-chunk", type=int, default=256)
    ap.add_argument("--signal-pair-batch", type=int, default=32)
    ap.add_argument(
        "--resident-probs", choices=["auto", "cpu", "cuda"], default="auto",
        help="keep tune probabilities in VRAM between holdouts when possible",
    )
    args = ap.parse_args()

    cache.CACHE_ROOT = Path(args.cache_dir)
    tag = args.tag or Path(args.cache_dir).name
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(args.device)
    keys = [(m, a) for m in args.m_values for a in args.agreement_modes]
    variants = _variant_list(args)

    scales = np.array([1.0], dtype=np.float32) if args.no_scale_tuning else np.geomspace(
        args.scale_min, args.scale_max, args.scale_points
    ).astype(np.float32)
    temperatures = np.array([1.0], dtype=np.float32) if args.no_output_temperature else np.geomspace(
        args.temperature_min, args.temperature_max, args.temperature_points
    ).astype(np.float32)

    backend = args.optimizer_backend
    if backend == "auto":
        backend = "torch" if device.type == "cuda" else "numpy"
    print(
        f"CE tuning: variants={variants} severities={args.tune_severities} "
        f"device={device} optimizer={backend}/{args.ce_dtype} "
        f"scale_points={len(scales)} temperature_points={len(temperatures)}"
    )

    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        grid_record, grid_path = load_grid_record(args.grid_tuning, model_name, tag, args.out)
        by_key, conditions, base_meta = precompute_all_conditions(
            model_name,
            args.tune_severities,
            keys,
            frac_tune=args.frac_tune,
            split_seed=args.split_seed,
            ladder_len=args.ladder_len,
            signal_chunk=args.signal_chunk,
            signal_pair_batch=args.signal_pair_batch,
            device=device,
            resident_probs=args.resident_probs,
        )

        for variant in variants:
            method, view_bias, allow_negative = VARIANTS[variant]
            print(f"\n--- {method} ---")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record = {
                "model": model_name,
                "method": method,
                "cache_dir": args.cache_dir,
                "timestamp": stamp,
                "tune_severities": args.tune_severities,
                "split": {"frac_tune": args.frac_tune, "seed": args.split_seed},
                "m_values": args.m_values,
                "ladder_len": args.ladder_len,
                "agreement_modes": args.agreement_modes,
                "standardize": True,
                "nonnegative": not allow_negative,
                "view_bias": bool(view_bias),
                "l2": args.l2,
                "bias_l2": args.bias_l2,
                "eps": args.eps,
                "grid_start": grid_path,
                "optimizer_backend": backend,
                "ce_dtype": args.ce_dtype,
                "ram_probability_cache": True,
                "scale_grid": scales.tolist(),
                "temperature_grid": temperatures.tolist(),
                "held_out": {},
                "global": {},
            }

            def tune_on(name, tune_corr, held_family=None):
                t0 = time.time()
                grid_start = grid_effective_start(grid_record, held_family)
                best = None

                for key in keys:
                    X, p_true = concatenate_conditions(
                        by_key, conditions, tune_corr, args.tune_severities, key
                    )
                    if X is None:
                        continue
                    starts = [grid_start] if grid_start is not None else None
                    theta, bias, objective, res = _fit_one_key(
                        X,
                        p_true,
                        starts=starts,
                        args=args,
                        view_bias=view_bias,
                        allow_negative=allow_negative,
                        device=device,
                    )
                    # Avoid another full-data CE pass: objective is exactly
                    # NLL + the two quadratic penalties.
                    ce_nll = float(
                        objective
                        - args.l2 * float(np.dot(theta, theta))
                        - (args.bias_l2 * float(np.dot(bias, bias)) if view_bias else 0.0)
                    )
                    cand = {
                        "key": key,
                        "theta": theta,
                        "bias": bias,
                        "objective": float(objective),
                        "ce_nll": ce_nll,
                        "success": bool(res.success),
                        "message": str(res.message),
                        "nit": int(res.nit),
                        "n_samples": int(len(X)),
                    }
                    if best is None or cand["objective"] < best["objective"]:
                        best = cand
                    del X, p_true

                if best is None:
                    return None

                key = best["key"]
                scale, tune_acc, _ = tune_scale_accuracy(
                    conditions,
                    by_key,
                    key,
                    best["theta"],
                    best["bias"],
                    tune_corr,
                    args.tune_severities,
                    scales,
                    device=device,
                    batch_scales=args.scale_batch,
                )
                out_temp, calibrated_nll, _ = tune_output_temperature(
                    conditions,
                    by_key,
                    key,
                    best["theta"],
                    best["bias"],
                    scale,
                    tune_corr,
                    args.tune_severities,
                    temperatures,
                    device=device,
                    eps=args.eps,
                    temperature_batch=args.temperature_batch,
                )

                m, agreement = key
                scaled_theta = scale * best["theta"]
                scaled_bias = scale * best["bias"]
                cfg = FusionConfig(
                    alpha=float(scaled_theta[0]),
                    beta=float(scaled_theta[1]),
                    gamma=float(scaled_theta[2]),
                    tau=1.0,
                    standardize=True,
                    m=int(m),
                    agreement=agreement,
                    view_bias=tuple(float(x) for x in scaled_bias),
                    output_temperature=float(out_temp),
                )
                orig, equal = baseline_means(base_meta, tune_corr, args.tune_severities)
                print(
                    f"  {name:14s}: CE-NLL {best['ce_nll']:.5f}  scale {scale:.3g}  "
                    f"acc {100*tune_acc:.2f}%  T={out_temp:.3g}  cal-NLL {calibrated_nll:.5f}  "
                    f"m={m} agree={agreement}  [{time.time()-t0:.1f}s]"
                )
                return {
                    "config": cfg.as_dict(),
                    "ce_direction": [float(x) for x in best["theta"]],
                    "ce_view_bias": [float(x) for x in best["bias"]],
                    "selection_scale": float(scale),
                    "output_temperature": float(out_temp),
                    "mean_fused_acc": float(tune_acc),
                    "mean_orig_acc": orig,
                    "mean_equal_acc": equal,
                    "mean_ce_nll": best["ce_nll"],
                    "mean_calibrated_nll": float(calibrated_nll),
                    "objective": best["objective"],
                    "optimizer": {
                        "success": best["success"],
                        "message": best["message"],
                        "nit": best["nit"],
                        "n_samples": best["n_samples"],
                        "backend": backend,
                        "dtype": args.ce_dtype,
                    },
                }

            for held, tune_corr, _ in leave_one_family_out():
                entry = tune_on(f"hold out {held}", tune_corr, held_family=held)
                if entry:
                    record["held_out"][held] = entry
            record["global"] = tune_on("global", list(CORRUPTION_FAMILY), held_family=None) or {}

            out = out_dir / f"tuning_{method}_{model_name}_{tag}_{stamp}.json"
            out.write_text(json.dumps(record, indent=2))
            print("  saved", out)


if __name__ == "__main__":
    main()
