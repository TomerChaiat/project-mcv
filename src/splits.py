"""Corruption families and the train/test splits used for tuning.

- Hyper-parameters are tuned leave-one-corruption-family-out, so the reported
  numbers measure generalisation across corruption *types*.
- Tuning and evaluation also use disjoint image halves, assigned per image so
  the same picture is always in the same half in every corruption/severity.
"""
import hashlib
from pathlib import Path
import numpy as np

#: The 15 standard ImageNet-C corruptions grouped into the 4 families used for
#: the leave-one-family-out split.
FAMILIES = {
    "noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}

#: corruption name -> family name
CORRUPTION_FAMILY = {c: fam for fam, cs in FAMILIES.items() for c in cs}
#: all 15 corruption names, family order
ALL_CORRUPTIONS = [c for cs in FAMILIES.values() for c in cs]


def leave_one_family_out():
    """Iterate the 4 leave-one-family-out folds.

    Yields:
        (held_out_family, tune_corruptions, test_corruptions) for each family:
        the 11 or 12 corruptions of the other families to tune on, and the 3 or
        4 corruptions of the held-out family to report.
    """
    for held in FAMILIES:
        tune = [c for fam, cs in FAMILIES.items() if fam != held for c in cs]
        yield held, tune, list(FAMILIES[held])


def image_key(path):
    """Stable id for an image, shared across every corruption/severity.

    ImageNet-C keeps the original file name (ILSVRC2012_val_XXXXXXXX.JPEG), so
    the stem is the same picture everywhere.
    """
    return Path(str(path)).stem


def in_tune_half(path, frac_tune=0.5, seed=0):
    """Whether an image belongs to the tuning half.

    Decided by hashing the image key, so the assignment is deterministic,
    platform-independent, and identical for the same picture in every
    corruption and severity.

    Args:
        path: image path (only its 'image_key' matters).
        frac_tune: target fraction assigned to the tuning half.
        seed: changes the split without changing 'frac_tune'.

    Returns:
        True for the tuning half, False for the test half.
    """
    h = hashlib.md5(f"{seed}:{image_key(path)}".encode()).hexdigest()
    return (int(h[:8], 16) % 10_000) < frac_tune * 10_000


def tune_test_masks(paths, frac_tune=0.5, seed=0):
    """Boolean masks splitting 'paths' into the tuning and test halves.

    Args:
        paths: sequence of image paths.
        frac_tune, seed: passed to 'in_tune_half'.

    Returns:
        (tune_mask, test_mask), each a bool array aligned with 'paths';
        'test_mask' is the complement of 'tune_mask'.
    """
    tune = np.array([in_tune_half(p, frac_tune, seed) for p in paths], dtype=bool)
    return tune, ~tune
