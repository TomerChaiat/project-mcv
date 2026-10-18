import numpy as np

from src.evaluation import score_many_torch, fused_correct_many_torch
from src.metrics import FusionConfig, precompute, score


def _probs(seed=0, n=37, v=8, c=19):
    rng = np.random.default_rng(seed)
    x = rng.gamma(shape=1.5, scale=1.0, size=(n, v, c)).astype(np.float32)
    x /= x.sum(axis=-1, keepdims=True)
    labels = rng.integers(0, c, size=n, dtype=np.int64)
    return x.astype(np.float16), labels


def test_gpu_eval_matches_numpy_multiple_configs_cpu():
    probs, labels = _probs()
    cfgs = {
        "a": FusionConfig(1.0, 0.5, 0.25, 0.5, True, 2, "all"),
        "b": FusionConfig(
            0.3, 0.7, 0.2, 1.0, True, 3, "origin",
            view_bias=tuple(np.linspace(-0.2, 0.2, probs.shape[1])),
            output_temperature=1.3,
        ),
        "c": FusionConfig(1.0, 0.0, 1.0, 0.2, False, 1, "all"),
    }
    got = score_many_torch(
        probs, labels, cfgs, device="cpu", signal_chunk=13, signal_pair_batch=7
    )
    for name, cfg in cfgs.items():
        want = score(
            precompute(probs, m=cfg.m, agreement=cfg.agreement),
            labels,
            cfg,
            full=True,
        )
        for key in [
            "orig_acc", "equal_acc", "conf_acc", "agree_acc", "view1_acc",
            "fused_acc", "oracle_acc", "orig_ece", "fused_ece",
            "orig_nll", "fused_nll", "negative_transfer", "positive_transfer",
        ]:
            assert np.isclose(got[name][key], want[key], atol=3e-5, rtol=3e-5), (name, key, got[name][key], want[key])


def test_correctness_matches_score():
    probs, labels = _probs(seed=3)
    cfgs = {
        "a": FusionConfig(1.0, 1.0, 0.5, 0.5, True, 1, "origin"),
        "b": FusionConfig(0.2, 0.0, 0.8, 1.0, True, 3, "all"),
    }
    rows, correctness = score_many_torch(
        probs, labels, cfgs, device="cpu", return_correctness=True,
        signal_chunk=11, signal_pair_batch=5,
    )
    corr2, orig2 = fused_correct_many_torch(
        probs, labels, cfgs, device="cpu", signal_chunk=11, signal_pair_batch=5
    )
    for name in cfgs:
        assert np.array_equal(correctness[name], corr2[name])
        assert np.isclose(correctness[name].mean(), rows[name]["fused_acc"])
    assert np.array_equal(correctness["__orig__"], orig2)
