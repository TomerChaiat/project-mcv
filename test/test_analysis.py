"""Unit tests for src/analysis.py -- the profile maths the pipeline runs on.
No GPU or data needed.

    PYTHONPATH=. python test/test_analysis.py
"""
import numpy as np

from src import analysis


def test_jsd_identical_is_zero():
    """JSD of a distribution with itself is 0."""
    p = np.array([0.2, 0.3, 0.5])
    assert abs(analysis.jensen_shannon(p, p)) < 1e-9


def test_jsd_disjoint_is_one():
    """With base-2 logs, JSD of two disjoint one-hots is exactly 1."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(analysis.jensen_shannon(a, b) - 1.0) < 1e-6


def test_jsd_symmetric():
    """JSD(x, y) == JSD(y, x)."""
    x = np.array([0.1, 0.6, 0.3])
    y = np.array([0.5, 0.4, 0.1])
    assert abs(analysis.jensen_shannon(x, y) - analysis.jensen_shannon(y, x)) < 1e-9


def test_identical_stack_has_zero_signals():
    """When every view predicts the same thing, agreement and stability are 0."""
    probs = np.full((1, 4, 3), 1 / 3, dtype=np.float32)
    C, A, S = analysis.profile_signals(probs, m=1, standardize=False)
    assert np.allclose(A, 0.0, atol=1e-6)
    assert np.allclose(S, 0.0, atol=1e-6)


def test_signal_shapes():
    """profile_signals returns one value per view for each of C, A, S."""
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((5, 6, 100))
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    C, A, S = analysis.profile_signals(probs, m=1)
    assert C.shape == A.shape == S.shape == (5, 6)


def test_fuse_uniform_weights_is_mean():
    """Fusing with equal weights is the plain average of the views."""
    rng = np.random.default_rng(1)
    probs = rng.random((3, 4, 10)).astype(np.float32)
    w = np.full((3, 4), 0.25, dtype=np.float32)
    assert np.allclose(analysis.fuse(probs, w), probs.mean(axis=1), atol=1e-6)


def test_fusion_weights_sum_to_one():
    """Each row of the softmax weights sums to 1."""
    scores = np.array([[1.0, 2.0, 0.5, -1.0]])
    w = analysis.fusion_weights(scores, tau=0.3)
    assert np.allclose(w.sum(axis=-1), 1.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all passed")
