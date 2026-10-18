import numpy as np

from src.ce_fusion import fit_nonnegative, loss_and_grad


def test_theta_zero_is_equal_weight_nll():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 5, 3)).astype(np.float32)
    p_true = rng.uniform(0.01, 0.9, size=(7, 5)).astype(np.float32)

    loss, _ = loss_and_grad(np.zeros(3), X, p_true, l2=0.0)
    expected = -np.mean(np.log(p_true.mean(axis=1)))
    assert np.allclose(loss, expected, rtol=1e-7, atol=1e-7)


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(11, 6, 3)).astype(np.float64)
    p_true = rng.uniform(0.02, 0.95, size=(11, 6)).astype(np.float64)
    theta = np.array([0.7, 0.4, 1.2], dtype=np.float64)

    loss, grad = loss_and_grad(theta, X, p_true, l2=1e-3)
    h = 1e-6
    fd = np.zeros_like(theta)
    for j in range(3):
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += h
        minus[j] -= h
        lp, _ = loss_and_grad(plus, X, p_true, l2=1e-3)
        lm, _ = loss_and_grad(minus, X, p_true, l2=1e-3)
        fd[j] = (lp - lm) / (2 * h)

    assert np.isfinite(loss)
    assert np.allclose(grad, fd, rtol=2e-5, atol=2e-6)


def test_optimizer_respects_nonnegative_constraint():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 4, 3)).astype(np.float32)
    p_true = rng.uniform(0.02, 0.95, size=(20, 4)).astype(np.float32)

    theta, loss, res = fit_nonnegative(
        X, p_true, l2=1e-3, random_restarts=1, seed=0, maxiter=100
    )
    assert np.all(theta >= -1e-12)
    assert np.isfinite(loss)
    assert res.x.shape == (3,)


def test_bias_gradient_matches_finite_difference():
    from src.ce_fusion import loss_and_grad_with_bias

    rng = np.random.default_rng(9)
    X = rng.normal(size=(13, 5, 3)).astype(np.float64)
    p_true = rng.uniform(0.02, 0.95, size=(13, 5)).astype(np.float64)
    params = rng.normal(size=3 + 4).astype(np.float64)

    loss, grad = loss_and_grad_with_bias(
        params, X, p_true, l2_theta=1e-3, l2_bias=2e-3
    )
    h = 1e-6
    fd = np.zeros_like(params)
    for j in range(len(params)):
        plus = params.copy(); plus[j] += h
        minus = params.copy(); minus[j] -= h
        lp, _ = loss_and_grad_with_bias(
            plus, X, p_true, l2_theta=1e-3, l2_bias=2e-3
        )
        lm, _ = loss_and_grad_with_bias(
            minus, X, p_true, l2_theta=1e-3, l2_bias=2e-3
        )
        fd[j] = (lp - lm) / (2 * h)

    assert np.isfinite(loss)
    np.testing.assert_allclose(grad, fd, rtol=3e-5, atol=3e-6)


def test_mean_zero_bias_parameterization():
    from src.ce_fusion import mean_zero_bias_from_free

    b = mean_zero_bias_from_free(np.array([0.3, -0.2, 0.7]), 4)
    assert b.shape == (4,)
    assert abs(b.sum()) < 1e-12


def test_bias_fit_returns_mean_zero_bias():
    from src.ce_fusion import fit_coefficients

    rng = np.random.default_rng(12)
    X = rng.normal(size=(25, 5, 3)).astype(np.float32)
    p_true = rng.uniform(0.02, 0.95, size=(25, 5)).astype(np.float32)
    theta, bias, loss, _ = fit_coefficients(
        X, p_true, view_bias=True, nonnegative=True,
        random_restarts=0, maxiter=50,
    )
    assert theta.shape == (3,)
    assert bias.shape == (5,)
    assert np.all(theta >= -1e-10)
    assert abs(bias.sum()) < 1e-8
    assert np.isfinite(loss)


def test_torch_objective_matches_numpy_theta_only_cpu():
    from src.ce_fusion import loss_and_grad_torch

    rng = np.random.default_rng(21)
    X = rng.normal(size=(41, 8, 3)).astype(np.float64)
    p_true = rng.uniform(1e-4, 0.9, size=(41, 8)).astype(np.float64)
    theta = np.array([0.7, 0.2, 1.1], dtype=np.float64)

    loss_np, grad_np = loss_and_grad(theta, X, p_true, l2=1e-3)
    loss_t, grad_t = loss_and_grad_torch(
        theta,
        X,
        p_true,
        l2_theta=1e-3,
        device="cpu",
        dtype=__import__("torch").float64,
    )
    np.testing.assert_allclose(loss_t, loss_np, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(grad_t, grad_np, rtol=1e-10, atol=1e-10)


def test_torch_objective_matches_numpy_with_bias_cpu():
    from src.ce_fusion import loss_and_grad_torch, loss_and_grad_with_bias
    import torch

    rng = np.random.default_rng(22)
    X = rng.normal(size=(37, 6, 3)).astype(np.float64)
    p_true = rng.uniform(1e-4, 0.95, size=(37, 6)).astype(np.float64)
    params = rng.normal(size=3 + 5).astype(np.float64)

    loss_np, grad_np = loss_and_grad_with_bias(
        params, X, p_true, l2_theta=1e-3, l2_bias=2e-3
    )
    loss_t, grad_t = loss_and_grad_torch(
        params,
        X,
        p_true,
        l2_theta=1e-3,
        l2_bias=2e-3,
        view_bias=True,
        device="cpu",
        dtype=torch.float64,
    )
    np.testing.assert_allclose(loss_t, loss_np, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(grad_t, grad_np, rtol=1e-10, atol=1e-10)


def test_torch_optimizer_matches_numpy_optimizer_cpu_float64():
    from src.ce_fusion import fit_coefficients, fit_coefficients_torch

    rng = np.random.default_rng(23)
    X = rng.normal(size=(80, 5, 3)).astype(np.float64)
    p_true = rng.uniform(0.01, 0.95, size=(80, 5)).astype(np.float64)

    common = dict(
        l2=1e-3,
        random_restarts=0,
        maxiter=100,
        nonnegative=True,
        view_bias=False,
    )
    theta_np, bias_np, loss_np, _ = fit_coefficients(X, p_true, **common)
    theta_t, bias_t, loss_t, _ = fit_coefficients_torch(
        X, p_true, device="cpu", dtype="float64", **common
    )
    np.testing.assert_allclose(theta_t, theta_np, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(bias_t, bias_np, rtol=0, atol=0)
    np.testing.assert_allclose(loss_t, loss_np, rtol=1e-9, atol=1e-10)


def test_free_signals_hold_excluded_coefficients_at_zero():
    """Fitting with a signal held out equals fitting without that signal."""
    from src.ce_fusion import fit_coefficients, fit_coefficients_torch

    rng = np.random.default_rng(31)
    X = rng.normal(size=(120, 6, 3)).astype(np.float64)
    # make A (index 1) informative, so holding it at 0 really constrains the fit
    p_true = 1.0 / (1.0 + np.exp(-(X[..., 1] + 0.5 * X[..., 2])))
    common = dict(l2=1e-3, bias_l2=1e-3, random_restarts=2, maxiter=200,
                  nonnegative=True, view_bias=True)

    theta, bias, loss, _ = fit_coefficients(X, p_true, free_signals=[True, False, True], **common)
    assert theta[1] == 0.0
    theta_red, bias_red, loss_red, _ = fit_coefficients(X[..., [0, 2]], p_true, **common)
    np.testing.assert_allclose(theta[[0, 2]], theta_red, atol=1e-5)
    np.testing.assert_allclose(bias, bias_red, atol=1e-5)
    np.testing.assert_allclose(loss, loss_red, rtol=1e-8)

    theta_full, _, loss_full, _ = fit_coefficients(X, p_true, **common)
    assert theta_full[1] > 0.1 and loss_full < loss

    theta_t, _, _, _ = fit_coefficients_torch(
        X, p_true, free_signals=[True, False, True], device="cpu", dtype="float64", **common)
    assert theta_t[1] == 0.0
    np.testing.assert_allclose(theta_t, theta, atol=1e-5)


def test_no_free_signals_is_the_bias_only_fit():
    from src.ce_fusion import fit_coefficients

    rng = np.random.default_rng(32)
    X = rng.normal(size=(40, 5, 3))
    p_true = rng.uniform(0.02, 0.95, size=(40, 5))
    common = dict(view_bias=True, random_restarts=3, maxiter=100, starts=[np.ones(3)])
    a = fit_coefficients(X, p_true, free_signals=[False, False, False], **common)
    b = fit_coefficients(X, p_true, theta_fixed_zero=True, **common)
    assert np.all(a[0] == 0.0)
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]
    try:
        fit_coefficients(X, p_true, free_signals=[False, False, False], view_bias=False)
    except ValueError:
        pass
    else:
        raise AssertionError("no free coefficient and no view bias must be rejected")
