#!/usr/bin/env python3
"""Benchmark vanilla 1-view inference vs the 15-view CE+bias method.

Disk I/O is excluded deliberately: the same already-loaded 256x256 image batch
is used by both methods.  Timings therefore compare model/view/fusion compute,
which is the deployment overhead attributable to the approach itself.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

from src import cache  # noqa: E402
from src.evaluation import signal_variants_torch  # noqa: E402
from src.metrics import FusionConfig  # noqa: E402


def load_global_cfg(path: str) -> FusionConfig:
    rec = json.loads(Path(path).read_text())
    if rec.get("held_out"):
        raise SystemExit(
            f"{path} still contains held_out configs; expected a global-only tuning file"
        )
    return FusionConfig.from_dict(rec["global"]["config"])


def load_batch_256(n: int, device: torch.device) -> torch.Tensor:
    paths = sorted(glob.glob(str(REPO / "data/imagenet/val/*/*.JPEG")))
    if len(paths) < n:
        raise SystemExit(
            f"need at least {n} clean images under data/imagenet/val; found {len(paths)}"
        )
    return torch.stack([cache.load_image_256(p) for p in paths[:n]]).to(device)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fused_probs(probs: torch.Tensor, cfg: FusionConfig) -> torch.Tensor:
    modes = [cfg.agreement]
    raw, std = signal_variants_torch(
        probs,
        [int(cfg.m)],
        modes,
        ladder_len=8,
        chunk=256,
        pair_batch=32,
    )
    C, A, S = (std if cfg.standardize else raw)[(int(cfg.m), cfg.agreement)]
    scores = float(cfg.alpha) * C + float(cfg.beta) * A + float(cfg.gamma) * S
    if cfg.view_bias:
        bias = torch.as_tensor(cfg.view_bias, device=probs.device, dtype=torch.float32)
        if bias.numel() != probs.shape[1]:
            raise RuntimeError(
                f"view_bias length {bias.numel()} != number of views {probs.shape[1]}"
            )
        scores = scores + bias[None, :]
    weights = torch.softmax(scores / float(cfg.tau), dim=1)
    fused = torch.bmm(weights[:, None, :], probs).squeeze(1)

    T = float(cfg.output_temperature)
    if abs(T - 1.0) > 1e-12:
        neg_inf = torch.full((), -torch.inf, device=fused.device, dtype=fused.dtype)
        logp = torch.where(fused > 0, torch.log(fused), neg_inf)
        fused = torch.softmax(logp / T, dim=1)
    return fused


def bench(fn, device: torch.device, warmup: int, reps: int):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            sync(device)
            times.append(time.perf_counter() - t0)
        peak = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else None
        )
    return {
        "median_s": float(statistics.median(times)),
        "min_s": float(min(times)),
        "max_s": float(max(times)),
        "peak_cuda_alloc_mb": peak,
    }


def center_crop_224(images256: torch.Tensor) -> torch.Tensor:
    h, w = images256.shape[-2:]
    y, x = (h - 224) // 2, (w - 224) // 2
    return images256[:, :, y:y + 224, x:x + 224]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet-tuning", required=True)
    ap.add_argument("--deit-tuning", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    configs = {
        "resnet18": load_global_cfg(args.resnet_tuning),
        "deit": load_global_cfg(args.deit_tuning),
    }
    images256 = load_batch_256(args.batch_size, device)
    out_rows = []

    for model_name in ("resnet18", "deit"):
        model = cache.LOADERS[model_name](device)[0]
        cfg = configs[model_name]

        def vanilla():
            x = center_crop_224(images256)
            x = TF.normalize(x, cache.MEAN, cache.STD)
            return torch.softmax(model(x), dim=-1)

        def ours():
            spatial = cache.spatial_views(images256)
            spectral = cache.spectral_views_fixed(spatial[:, 0], cache.FIXED_CUTOFFS)
            views = torch.cat([spectral, spatial[:, 1:]], dim=1)  # 8 spectral + 7 nonduplicate spatial = 15
            probs = cache.view_probs(model, views)
            return fused_probs(probs, cfg)

        vanilla_stats = bench(vanilla, device, args.warmup, args.reps)
        ours_stats = bench(ours, device, args.warmup, args.reps)

        v_ms = vanilla_stats["median_s"] * 1000.0 / args.batch_size
        o_ms = ours_stats["median_s"] * 1000.0 / args.batch_size
        overhead = o_ms / v_ms

        for method, views, stats, ms in (
            ("vanilla", 1, vanilla_stats, v_ms),
            ("final_combined", 15, ours_stats, o_ms),
        ):
            out_rows.append({
                "model": model_name,
                "method": method,
                "views": views,
                "batch_size": args.batch_size,
                "reps": args.reps,
                "device": str(device),
                "median_batch_ms": stats["median_s"] * 1000.0,
                "ms_per_image": ms,
                "images_per_second": 1000.0 / ms,
                "overhead_vs_vanilla": 1.0 if method == "vanilla" else overhead,
                "peak_cuda_alloc_mb": stats["peak_cuda_alloc_mb"],
            })

        print(
            f"{model_name}: vanilla {v_ms:.3f} ms/img | final {o_ms:.3f} ms/img | "
            f"overhead {overhead:.2f}x"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "inference_timing.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    meta = {
        "definition": "GPU compute from an already-loaded 256x256 tensor; disk/JPEG I/O excluded for both approaches",
        "vanilla": "center crop + ImageNet normalization + one model forward + softmax",
        "final_combined": "8 fixed spectral views + 7 additional spatial views + 15 forwards + confidence/agreement/stability + CE+bias fusion/calibration",
        "batch_size": args.batch_size,
        "reps": args.reps,
        "warmup": args.warmup,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "rows": out_rows,
    }
    (out_dir / "inference_timing.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
