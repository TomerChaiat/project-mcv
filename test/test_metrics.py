import numpy as np

from src.metrics import FusionConfig, precompute, score, temperature_scale_probs


def _probs(seed=0, n=31, v=8, c=17):
    rng = np.random.default_rng(seed)
    x = rng.random((n, v, c), dtype=np.float32)
    x /= x.sum(axis=-1, keepdims=True)
    return x


def test_output_temperature_preserves_argmax():
    p = _probs(v=1)[:, 0]
    a = p.argmax(-1)
    for t in (0.3, 0.7, 1.0, 2.0, 4.0):
        q = temperature_scale_probs(p, t)
        np.testing.assert_array_equal(q.argmax(-1), a)
        np.testing.assert_allclose(q.sum(axis=-1), 1.0, atol=1e-6)


def test_baselines_do_not_depend_on_main_tau_standardization_or_agreement():
    probs = _probs()
    labels = np.arange(len(probs)) % probs.shape[-1]

    rows = []
    for cfg in (
        FusionConfig(alpha=1, beta=0, gamma=1, tau=0.1, standardize=True, m=1, agreement="all"),
        FusionConfig(alpha=1, beta=1, gamma=0, tau=1.0, standardize=True, m=2, agreement="origin"),
        FusionConfig(alpha=1, beta=0.5, gamma=0.5, tau=0.5, standardize=False, m=3, agreement="origin"),
    ):
        pre = precompute(probs, m=cfg.m, agreement=cfg.agreement)
        rows.append(score(pre, labels, cfg, full=True))

    assert len({r["conf_acc"] for r in rows}) == 1
    assert len({r["agree_acc"] for r in rows}) == 1


def test_view_bias_length_checked():
    probs = _probs(v=4)
    labels = np.arange(len(probs)) % probs.shape[-1]
    pre = precompute(probs, m=1, agreement="all")
    cfg = FusionConfig(view_bias=(0.1, 0.2))
    try:
        score(pre, labels, cfg, full=False)
    except ValueError as e:
        assert "view_bias" in str(e)
    else:
        raise AssertionError("expected view_bias length error")
