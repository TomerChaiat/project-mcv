import numpy as np

from src.bootstrap import cluster_bootstrap, image_keys


def _cond(keys, a, b):
    return {"keys": np.asarray(keys), "a": np.asarray(a, bool), "b": np.asarray(b, bool)}


def test_point_is_mean_of_condition_deltas():
    keys = [f"img{i}" for i in range(4)]
    conds = [
        _cond(keys, [1, 1, 0, 0], [0, 1, 0, 0]),   # +0.25
        _cond(keys, [1, 1, 1, 1], [1, 1, 1, 1]),   #  0
    ]
    out = cluster_bootstrap(conds, n_boot=200, seed=0)
    assert np.isclose(out["point"], 0.125)
    assert out["n_images"] == 4 and out["n_conditions"] == 2


def test_identical_predictors_give_zero_interval():
    rng = np.random.default_rng(0)
    keys = [f"img{i}" for i in range(50)]
    a = rng.random(50) < 0.5
    out = cluster_bootstrap([_cond(keys, a, a)] * 5, n_boot=300, seed=1)
    assert out["ci95"] == [0.0, 0.0] and out["point"] == 0.0


def test_correlated_conditions_widen_interval_vs_independent():
    """The same per-image effect repeated over many conditions must not shrink
    the CI the way 'independent conditions' resampling would: the cluster SE
    of C identical copies equals the SE of one copy."""
    rng = np.random.default_rng(0)
    n = 400
    keys = [f"img{i}" for i in range(n)]
    a = rng.random(n) < 0.55
    b = rng.random(n) < 0.50
    one = cluster_bootstrap([_cond(keys, a, b)], n_boot=2000, seed=2)
    many = cluster_bootstrap([_cond(keys, a, b)] * 20, n_boot=2000, seed=2)
    assert np.isclose(one["se"], many["se"], rtol=0.02)


def test_alignment_by_key_not_position():
    keys = ["x", "y", "z"]
    c1 = _cond(keys, [1, 0, 0], [0, 0, 0])
    c2 = _cond(keys[::-1], [0, 0, 1], [0, 0, 0])  # same image "x" gains in both
    out = cluster_bootstrap([c1, c2], n_boot=2000, seed=3)
    assert np.isclose(out["point"], 1 / 3)
    assert out["n_images"] == 3


def test_image_keys_use_stem():
    assert list(image_keys(["a/b/ILSVRC_1.JPEG", "c/ILSVRC_1.JPEG"])) == ["ILSVRC_1", "ILSVRC_1"]


def test_weighted_condition_auroc_matches_resampled_mann_whitney():
    """Image weights give the AUROC of the resampled data, per condition, ties = 1/2."""
    import torch
    from scripts.signal_ablation import ConditionAUROC

    def mann_whitney(score, y):
        pos, neg = score[y == 1], score[y == 0]
        diff = pos[:, None] - neg[None, :]
        return ((diff > 0) + 0.5 * (diff == 0)).mean()

    rng = np.random.default_rng(5)
    n_img, n_cond = 40, 3
    img = np.tile(np.arange(n_img), n_cond)
    cond = np.repeat(np.arange(n_cond), n_img)
    score = np.round(rng.normal(size=len(img)), 1)          # rounding creates ties
    y = (rng.random(len(img)) < 0.4 + 0.1 * score).astype(int)
    calc = ConditionAUROC(score, y, cond, img, n_cond, "cpu")
    W = np.vstack([np.ones(n_img, int), rng.multinomial(n_img, np.ones(n_img) / n_img, size=3)])
    got = calc(torch.as_tensor(W, dtype=torch.float32)).numpy()
    for b, w in enumerate(W):
        for c in range(n_cond):
            rows = np.where(cond == c)[0]
            rep = np.repeat(rows, w[img[rows]])
            assert np.isclose(got[b, c], mann_whitney(score[rep], y[rep]))
