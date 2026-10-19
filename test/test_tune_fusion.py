import numpy as np
import torch

from src.metrics import precompute, score
from scripts.tune_fusion import (
    grid,
    signal_variants_one_pass,
    score_block_torch,
)


def _random_probs(rng, n=17, v=8, c=19):
    x = rng.random((n, v, c), dtype=np.float32)
    x /= x.sum(axis=-1, keepdims=True)
    return x.astype(np.float32)


def test_one_pass_signals_match_project_definitions():
    rng = np.random.default_rng(3)
    probs = _random_probs(rng)
    raw, _ = signal_variants_one_pass(
        probs, [1, 2, 3], ["all", "origin"], chunk=5
    )
    for m in (1, 2, 3):
        for agreement in ("all", "origin"):
            pre = precompute(probs, m=m, agreement=agreement)
            C, A, S = raw[(m, agreement)]
            np.testing.assert_allclose(C, pre["C"], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(A, pre["A"], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(S, pre["S"], rtol=1e-6, atol=1e-7)


def test_gpu_scoring_matches_reference_accuracy_cpu():
    rng = np.random.default_rng(7)
    probs = _random_probs(rng, n=23, v=8, c=31)
    labels = rng.integers(0, probs.shape[-1], size=len(probs), dtype=np.int64)

    raw, std = signal_variants_one_pass(probs, [2], ["origin"], chunk=7)
    cfgs = list(grid(2, "origin"))[:40]

    probs_t = torch.from_numpy(probs)
    labels_t = torch.from_numpy(labels)
    gpu = score_block_torch(
        probs_t,
        labels_t,
        raw[(2, "origin")],
        std[(2, "origin")],
        cfgs,
        device=torch.device("cpu"),
        config_batch=9,
    )

    legacy = np.array([
        score(precompute(probs, m=2, agreement="origin"), labels, cfg, full=False)
        for cfg in cfgs
    ], dtype=np.float32)
    np.testing.assert_allclose(gpu, legacy, rtol=0, atol=0)


def test_torch_pairwise_jsd_matches_numpy_cpu():
    from src.evaluation import pairwise_jsd_torch
    from src import analysis

    rng = np.random.default_rng(11)
    probs = _random_probs(rng, n=13, v=8, c=23)
    # Include exact zeros because the real float16 cache can underflow to zero.
    probs[0, 1, 0] = 0.0
    probs[0, 1] /= probs[0, 1].sum()

    expected = analysis.pairwise_jsd(probs)
    got = pairwise_jsd_torch(torch.from_numpy(probs), pair_batch=5).numpy()
    np.testing.assert_allclose(got, expected, rtol=3e-6, atol=3e-7)


def test_torch_signal_variants_match_numpy_cpu():
    from src.evaluation import signal_variants_torch

    rng = np.random.default_rng(13)
    probs = _random_probs(rng, n=19, v=8, c=29)

    raw_np, std_np = signal_variants_one_pass(
        probs, [1, 2, 3], ["all", "origin"], chunk=6
    )
    raw_t, std_t = signal_variants_torch(
        torch.from_numpy(probs),
        [1, 2, 3],
        ["all", "origin"],
        chunk=6,
        pair_batch=4,
    )

    for key in raw_np:
        for a, b in zip(raw_np[key], raw_t[key]):
            np.testing.assert_allclose(b.numpy(), a, rtol=5e-6, atol=5e-7)
        for a, b in zip(std_np[key], std_t[key]):
            np.testing.assert_allclose(b.numpy(), a, rtol=2e-5, atol=5e-6)


def test_torch_signals_preserve_grid_accuracy_cpu():
    from src.evaluation import signal_variants_torch

    rng = np.random.default_rng(17)
    probs = _random_probs(rng, n=29, v=8, c=37)
    labels = rng.integers(0, probs.shape[-1], size=len(probs), dtype=np.int64)
    cfgs = list(grid(2, "origin"))

    raw_np, std_np = signal_variants_one_pass(
        probs, [2], ["origin"], chunk=9
    )
    raw_t, std_t = signal_variants_torch(
        torch.from_numpy(probs), [2], ["origin"], chunk=9, pair_batch=5
    )
    probs_t = torch.from_numpy(probs)
    labels_t = torch.from_numpy(labels)

    acc_np = score_block_torch(
        probs_t, labels_t, raw_np[(2, "origin")], std_np[(2, "origin")],
        cfgs, device=torch.device("cpu"), config_batch=31
    )
    acc_t = score_block_torch(
        probs_t, labels_t, raw_t[(2, "origin")], std_t[(2, "origin")],
        cfgs, device=torch.device("cpu"), config_batch=31
    )
    np.testing.assert_array_equal(acc_t, acc_np)
