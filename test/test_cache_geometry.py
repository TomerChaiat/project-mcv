import json

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src import cache


def _save(tmp_path, name, w, h, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    p = tmp_path / name
    Image.fromarray(arr).save(p)  # PNG: lossless
    return p, arr


def test_224_image_is_used_as_is(tmp_path):
    """ImageNet-C images are already 224x224 and are used as-is."""
    p, arr = _save(tmp_path, "c.png", 224, 224)
    x = cache.load_image(p)
    assert x.shape == (3, 224, 224)
    assert torch.equal(x, TF.to_tensor(Image.fromarray(arr)))


def test_full_size_image_gets_resize_and_center_crop(tmp_path):
    p, _ = _save(tmp_path, "v.png", 341, 256)
    x = cache.load_image(p)
    ref = TF.to_tensor(TF.center_crop(TF.resize(Image.open(p).convert("RGB"), 256), [224, 224]))
    assert torch.equal(x, ref)


def test_spatial_view0_is_input_and_shapes():
    imgs = torch.rand(2, 3, 224, 224)
    v = cache.spatial_views(imgs)
    assert v.shape == (2, 8, 3, 224, 224)
    assert torch.equal(v[:, 0], imgs)
    assert torch.equal(v[:, 1], torch.flip(imgs, dims=[-1]))
    assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0


def test_default_corner_views_are_resized_196_crops():
    """Default ("resize") corners: 196 px corner crops resized back to 224 -
    the geometry of the cache_spatial/ baseline."""
    imgs = torch.rand(2, 3, 224, 224)
    v = cache.spatial_views(imgs)
    ref = TF.resize(imgs[:, :, :196, 28:], [224, 224], antialias=True)   # top-right
    assert torch.equal(v[:, 3], ref.clamp(0.0, 1.0))


def test_pad_corner_views_are_unresampled_shifts():
    """"pad" corners (control only) are 224 windows of the 256 reflect-padded image: each is
    the input shifted by 16 px with pixels copied 1:1 (no interpolation, so
    no smoothing), and the mirrored border is continuous."""
    imgs = torch.rand(2, 3, 224, 224)
    v = cache.spatial_views(imgs, mode="pad")
    tl, tr, bl, br = v[:, 2], v[:, 3], v[:, 4], v[:, 5]
    assert torch.equal(tl[..., 16:, 16:], imgs[..., :208, :208])
    assert torch.equal(tr[..., 16:, :208], imgs[..., :208, 16:])
    assert torch.equal(bl[..., :208, 16:], imgs[..., 16:, :208])
    assert torch.equal(br[..., :208, :208], imgs[..., 16:, 16:])
    assert torch.equal(tl[..., 15, 16:], imgs[..., 1, :208])   # reflection, edge not repeated
    assert torch.equal(v[:, 6], torch.flip(tl, dims=[-1]))
    assert torch.equal(v[:, 7], torch.flip(br, dims=[-1]))


def test_virtual_combined_cache_matches_merge(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    paths = np.asarray([f"x/ILSVRC_{i}.JPEG" for i in range(5)])
    labels = np.arange(5, dtype=np.int32)
    spec = rng.random((5, 8, 10)).astype(np.float16)
    spat = rng.random((5, 8, 10)).astype(np.float16)
    spat[:, 0] = spec[:, 0]
    for root, probs, order in [("spec", spec, np.arange(5)), ("spat", spat, np.arange(5)[::-1])]:
        monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / root)
        cache.save_spatial_condition("m", "fog", 3, probs=probs[order], labels=labels[order],
                                     paths=paths[order])
    vdir = tmp_path / "virt"
    vdir.mkdir()
    (vdir / cache.VIRTUAL_SPEC).write_text(json.dumps(
        {"spectral": str(tmp_path / "spec"), "spatial": str(tmp_path / "spat")}))
    monkeypatch.setattr(cache, "CACHE_ROOT", vdir)
    assert cache.has_condition("m", "fog", 3)
    assert not cache.has_condition("m", "fog", 4)
    d = cache.load_condition("m", "fog", 3)
    assert d["probs"].shape == (5, 15, 10)
    assert np.array_equal(d["probs"][:, :8], spec)
    assert np.array_equal(d["probs"][:, 8:], spat[:, 1:])  # re-aligned by path
    assert list(d["paths"]) == list(paths) and d["ladder_len"] == 8
