import json

from src import splits


def test_clean_names_map_to_imagenet_c_key(tmp_path, monkeypatch):
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"n01443537_val_00000": "ILSVRC2012_val_00000236"}))
    monkeypatch.setattr(splits, "CLEAN_NAME_MAP", m)
    monkeypatch.setattr(splits, "_clean_map", None)
    clean = "data/imagenet/val/n01443537/n01443537_val_00000.JPEG"
    corr = "data/imagenet-c/fog/3/n01443537/ILSVRC2012_val_00000236.JPEG"
    assert splits.image_key(clean) == splits.image_key(corr) == "ILSVRC2012_val_00000236"
    assert splits.in_tune_half(clean) == splits.in_tune_half(corr)
    # unmapped names fall back to their own stem
    assert splits.image_key("x/other_name.JPEG") == "other_name"


def test_extra_corruptions_never_reach_tuning():
    """The 4 held-out extras map onto existing families but are kept out of
    every list the tuners and selection iterate over."""
    from src.splits import ALL_CORRUPTIONS, CORRUPTION_FAMILY, EXTRA_FAMILY, FAMILIES
    assert set(EXTRA_FAMILY).isdisjoint(CORRUPTION_FAMILY)
    assert set(EXTRA_FAMILY).isdisjoint(ALL_CORRUPTIONS)
    assert set(EXTRA_FAMILY.values()) == set(FAMILIES)
