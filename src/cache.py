"""Profile cache: run each (model, corruption, severity) through the models once
and store the per-view probability vectors S(x) on disk. Every experiment after
that is NumPy over these arrays -- see src/analysis.py, src/metrics.py.

Two view families (ladders) are supported, selected by scripts/cache_profiles.py
--ladder and written to separate cache roots so they never mix:
    energy  (default) per-image adaptive cutoffs -- see `spectral_views`
    fixed   the same absolute hand-spaced cutoffs for every image
    geometric the same endpoints/K as fixed, but constant-ratio spacing

Layout (CACHE_ROOT is set by the caller, e.g. "cache" or "cache_fixed"):
    <CACHE_ROOT>/<model>/<corruption>/<severity>.npz
    <CACHE_ROOT>/<model>/clean/0.npz

Each .npz holds:
    probs     (N, K+1, 1000) float16   softmax outputs, view 0 = unfiltered
    cutoffs   (N, K)         float32    per-image low-pass cutoffs actually used
    labels    (N,)           int32      true ImageNet index
    paths     (N,)           str        source image path
    energies  (K,)           float32    the energy ladder targets
"""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.filters import radial_lowpass_mask
from src.models import load_resnet18, load_deit_tiny

#: where cached conditions are written / read
CACHE_ROOT = Path("cache")

#: The energy ladder: filtered view k keeps 'ENERGIES[k]' of the image's
#: spectral energy. K = len(ENERGIES) filtered views, plus the original.
ENERGIES = (0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93)

#: Alternative ladder: the same absolute cutoff for every image, instead of
#: solving per image for a target energy fraction. See the "fixed" ladder note
#: below `compute_condition` for why this exists.
FIXED_CUTOFFS = (0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.12)

#: Geometric alternative with the same endpoints and number of views as
#: FIXED_CUTOFFS.  The ratio is (0.12 / 0.9) ** (1 / 6) ~= 0.715.
#: Keeping endpoints and K fixed makes fixed-vs-geometric a clean ablation:
#: only the spacing of the spectral views changes.
GEOMETRIC_CUTOFFS = tuple(
    float(x) for x in np.geomspace(FIXED_CUTOFFS[0], FIXED_CUTOFFS[-1], len(FIXED_CUTOFFS))
)

LADDER_CUTOFFS = {
    "fixed": FIXED_CUTOFFS,
    "geometric": GEOMETRIC_CUTOFFS,
}

#: ImageNet normalisation constants - the same for ResNet-18 and DeiT-Tiny.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

#: model name (as used in cache paths and CLI flags) -> loader function
LOADERS = {"resnet18": load_resnet18, "deit": load_deit_tiny}


# --------------------------------------------------------------------------- #
# image listing
# --------------------------------------------------------------------------- #
EXTS = {".jpeg", ".jpg", ".png"}


def list_condition_images(corruption, severity, wnid_to_index,
                          corrupt_root="data/imagenet-c",
                          clean_root="data/imagenet/val",
                          n_per_class=None):
    """List every image of one condition, with its label.

    Deterministic: classes in sorted wnid order, images in sorted filename order
    within each class, so the ordering is identical for every model.

    Args:
        corruption: corruption name, or "clean" for the validation images.
        severity: 1..5 (ignored when corruption == "clean").
        wnid_to_index: the subset map (wnid -> ImageNet index).
        corrupt_root, clean_root: dataset roots.
        n_per_class: keep only the first N images of each class (None = all).

    Returns:
        (paths, labels): a list of path strings and an int32 array of the same
        length. Classes with no folder on disk are silently skipped.
    """
    if corruption == "clean":
        base = Path(clean_root)
        subdirs = {w: base / w for w in wnid_to_index}
    else:
        base = Path(corrupt_root) / corruption / str(severity)
        subdirs = {w: base / w for w in wnid_to_index}

    paths, labels = [], []
    for wnid in sorted(wnid_to_index):
        d = subdirs[wnid]
        if not d.is_dir():
            continue
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)
        if n_per_class:
            imgs = imgs[:n_per_class]
        paths.extend(str(p) for p in imgs)
        labels.extend([wnid_to_index[wnid]] * len(imgs))
    return paths, np.asarray(labels, dtype=np.int32)


# --------------------------------------------------------------------------- #
# GPU compute
# --------------------------------------------------------------------------- #
def load_image(path):
    """Load one image as a (3, 224, 224) float tensor in [0, 1].

    Resize the short side to 256, centre-crop 224, to tensor - the standard
    ImageNet eval geometry, applied before any filtering or normalisation.
    """
    pil = Image.open(path).convert("RGB")
    pil = TF.center_crop(TF.resize(pil, 256), [224, 224])
    return TF.to_tensor(pil)


def load_image_256(path):
    """Load one image as a (3, 256, 256) float tensor in [0, 1].

    Resize the short side to 256, centre-crop to a 256x256 square - the room
    the spatial-TTA baseline needs for its corner crops. Cropping this square
    to its own centre 224 reproduces 'load_image' exactly, so the spatial-TTA
    view 0 matches the spectral profile's view 0.
    """
    pil = Image.open(path).convert("RGB")
    pil = TF.center_crop(TF.resize(pil, 256), [256, 256])
    return TF.to_tensor(pil)


#: pixel offsets (top, left) of the four 224-crops inside a 256x256 image
_CORNER_OFFSETS = ((0, 0), (0, 32), (32, 0), (32, 32))


def spatial_views(images256, crop=224):
    """Build an 8-view spatial-TTA stack: crop position + horizontal flip.

    The 'equal view count' spatial-augmentation baseline the proposal asks
    for, sized to match the K+1=8 views of the spectral ladder: the centre
    crop (== the spectral profile's view 0) and its flip, the four corner
    crops, and two of their flips.

    Args:
        images256: (B, 3, 256, 256) tensor in [0, 1].
        crop: crop size (224, matching the spectral views).

    Returns:
        (B, 8, 3, crop, crop) tensor in [0, 1]; view 0 is the plain centre
        crop (the same geometry as every other view in this project).
    """
    h, w = images256.shape[-2:]
    cy, cx = (h - crop) // 2, (w - crop) // 2
    center = images256[:, :, cy:cy + crop, cx:cx + crop]
    corners = [images256[:, :, t:t + crop, l:l + crop] for t, l in _CORNER_OFFSETS]
    tl, tr, bl, br = corners

    views = [center, torch.flip(center, dims=[-1]),
             tl, tr, bl, br,
             torch.flip(tl, dims=[-1]), torch.flip(br, dims=[-1])]
    return torch.stack(views, dim=1)


@torch.no_grad()
def spectral_views(images, energies, bisect_iters=18):
    """Build the low-pass view stack and the per-image cutoffs, batched on GPU.

    View 0 is the original image; view k>0 is the original low-passed with a
    Gaussian whose cutoff is solved (by batched bisection on the retained-energy
    fraction) so the view keeps 'energies[k-1]' of that image's spectral energy.
    This is the vectorised equivalent of 'src.filters.make_spectral_views' + 
   'cutoffs_for_energy_ladder'.

    Args:
        images: (B, 3, H, W) tensor in [0, 1] on the target device.
        energies: iterable of K retained-energy targets.
        bisect_iters: bisection steps per target (18 -> ~1e-6 precision).

    Returns:
        (views, cutoffs): views (B, K+1, 3, H, W) in [0, 1],
        cutoffs (B, K) float.
    """
    b, c, h, w = images.shape
    spec = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
    power = spec.abs() ** 2
    total = power.sum(dim=(-3, -2, -1))                       # (B,)

    y = torch.linspace(-1.0, 1.0, h, device=images.device)
    x = torch.linspace(-1.0, 1.0, w, device=images.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r2 = xx ** 2 + yy ** 2                                    # (H,W)

    def mask(cutoff):                                         # cutoff (B,)
        return torch.exp(-r2[None] / (2.0 * cutoff[:, None, None] ** 2))

    def kept_fraction(cutoff):
        m = mask(cutoff)[:, None]                             # (B,1,H,W)
        # m multiplies the Fourier amplitude, so retained *energy* scales
        # with m**2.  Using power*m would overestimate retained energy.
        return (power * m.square()).sum(dim=(-3, -2, -1)) / total  # (B,)

    views = [images]
    cutoffs = []
    for target in energies:
        lo = torch.full((b,), 1e-3, device=images.device)
        hi = torch.full((b,), 1.0, device=images.device)
        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            need_larger = kept_fraction(mid) < target
            lo = torch.where(need_larger, mid, lo)
            hi = torch.where(need_larger, hi, mid)
        cut = 0.5 * (lo + hi)
        cutoffs.append(cut)
        filtered = torch.fft.ifft2(
            torch.fft.ifftshift(spec * mask(cut)[:, None], dim=(-2, -1)),
            dim=(-2, -1),
        ).real
        views.append(filtered.clamp(0.0, 1.0))

    return torch.stack(views, dim=1), torch.stack(cutoffs, dim=1)


@torch.no_grad()
def spectral_views_fixed(images, cutoffs=FIXED_CUTOFFS):
    """Low-pass view stack using the same absolute cutoffs for every image.

    Diagnostic finding this exists to fix: the energy-normalised ladder in
    `spectral_views` solves each cutoff so the view keeps a target fraction of
    *that image's* spectral energy. For an already-low-frequency image (e.g.
    one corrupted by blur or fog, whose spectrum is naturally concentrated near
    DC), "keep 99% of energy" is satisfied by a very narrow, aggressive mask -
    so the mildest rung of the energy ladder already devastates these images,
    while it stays mild for noise-corrupted images (whose spectra are spread
    out). Fixed cutoffs apply the same absolute filter regardless of the
    image's own spectral shape, so filter strength no longer depends on the
    corruption. See scripts/cache_profiles.py --ladder fixed.

    Args:
        images: (B, 3, H, W) tensor in [0, 1] on the target device.
        cutoffs: iterable of K absolute cutoffs (same for every image).

    Returns:
        (B, K+1, 3, H, W) tensor in [0, 1]; view 0 is the original.
    """
    spec = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
    h, w = images.shape[-2:]
    views = [images]
    for c in cutoffs:
        mask = radial_lowpass_mask(h, w, c, device=images.device, dtype=images.dtype)
        filtered = torch.fft.ifft2(
            torch.fft.ifftshift(spec * mask, dim=(-2, -1)), dim=(-2, -1),
        ).real
        views.append(filtered.clamp(0.0, 1.0))
    return torch.stack(views, dim=1)


@torch.no_grad()
def view_probs(model, views):
    """Normalise every view and run it through the model.

    Args:
        model: a frozen classifier.
        views: (B, K+1, 3, H, W) tensor in [0, 1].

    Returns:
        (B, K+1, num_classes) tensor of softmax probabilities.
    """
    b, k1 = views.shape[:2]
    flat = views.reshape(b * k1, *views.shape[2:])
    flat = TF.normalize(flat, MEAN, STD)
    logits = model(flat)
    return torch.softmax(logits, dim=-1).reshape(b, k1, -1)




def _prefetched_image_batches(image_paths, loader, batch_size, workers=8, prefetch_batches=2, pin_memory=False):
    """Yield CPU image batches while loading future batches in background threads.

    The old cache loop called ``pool.map`` only after the previous GPU batch had
    finished, so JPEG decode/resize and GPU inference alternated instead of
    overlapping.  Here up to ``prefetch_batches`` future batches are already in
    flight while the caller processes the current one.  Image order and the
    exact loader function are unchanged.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if prefetch_batches < 0:
        raise ValueError("prefetch_batches must be >= 0")
    n = len(image_paths)
    with ThreadPoolExecutor(workers) as pool:
        pending = deque()
        next_i = 0

        def submit_one(i):
            chunk = image_paths[i:i + batch_size]
            futures = [pool.submit(loader, p) for p in chunk]
            pending.append((i, len(chunk), futures))

        # Current batch + bounded look-ahead.
        for _ in range(prefetch_batches + 1):
            if next_i >= n:
                break
            submit_one(next_i)
            next_i += batch_size

        while pending:
            i, count, futures = pending.popleft()
            batch = torch.stack([f.result() for f in futures])
            if pin_memory:
                batch = batch.pin_memory()
            if next_i < n:
                submit_one(next_i)
                next_i += batch_size
            yield i, count, batch


@torch.no_grad()
def compute_condition(model, image_paths, device, energies=ENERGIES,
                      batch_size=64, workers=8, ladder="energy", prefetch_batches=2):
    """Compute the full profile array for a list of images.

    ~4-5 ms/image on an A10 (batched forward, GPU filtering, threaded loading).

    Args:
        model: a frozen classifier already on 'device'.
        image_paths: list of image paths (from 'list_condition_images').
        device: torch device.
        energies: the energy ladder targets, used when ladder == "energy".
            When ladder == "fixed", only `len(energies)` is used (to size the
            output) - the actual cutoffs are `FIXED_CUTOFFS`.
        batch_size: images per GPU batch (lower for DeiT if memory is tight).
        workers: threads for image loading.
        ladder: "energy" (per-image adaptive, the original design), "fixed"
            (same hand-spaced absolute cutoffs for every image), or "geometric"
            (same endpoints/K as fixed, constant-ratio spacing).

    Returns:
        (probs, cutoffs): probs (N, K+1, 1000) float16 softmax outputs,
        cutoffs (N, K) float32 - the low-pass cutoffs actually used per image
        (identical across images when ladder == "fixed").
    """
    k = len(energies)
    n = len(image_paths)
    probs = np.empty((n, k + 1, 1000), dtype=np.float16)
    cutoffs = np.empty((n, k), dtype=np.float32)
    if ladder in LADDER_CUTOFFS:
        ladder_cutoffs = tuple(LADDER_CUTOFFS[ladder][:k])
        fixed = np.asarray(ladder_cutoffs, dtype=np.float32)
    elif ladder != "energy":
        raise ValueError(
            f"unknown ladder {ladder!r}; expected 'energy', 'fixed', or 'geometric'"
        )

    pin = torch.device(device).type == "cuda"
    for i, count, batch_cpu in _prefetched_image_batches(
        image_paths, load_image, batch_size, workers=workers,
        prefetch_batches=prefetch_batches, pin_memory=pin,
    ):
        batch = batch_cpu.to(device, non_blocking=pin)
        if ladder in LADDER_CUTOFFS:
            views = spectral_views_fixed(batch, ladder_cutoffs)
            cutoffs[i:i + count] = fixed
        else:
            views, cut = spectral_views(batch, energies)
            cutoffs[i:i + count] = cut.cpu().numpy()
        p = view_probs(model, views)
        probs[i:i + count] = p.cpu().numpy().astype(np.float16)
    return probs, cutoffs


@torch.no_grad()
def compute_spatial_condition(model, image_paths, device, batch_size=64, workers=8,
                              prefetch_batches=2):
    """Compute the 8-view spatial-TTA profile for a list of images.

    Same shape and geometry budget as 'compute_condition' (K+1=8 views per
    image), but the views are crop/flip variants instead of low-pass filters -
    the "spatial TTA with equal view count" baseline from the proposal.

    Returns:
        probs: (N, 8, 1000) float16 softmax outputs; view 0 is the plain
        centre crop (matches view 0 of the spectral profile exactly).
    """
    n = len(image_paths)
    probs = np.empty((n, 8, 1000), dtype=np.float16)

    pin = torch.device(device).type == "cuda"
    for i, count, batch_cpu in _prefetched_image_batches(
        image_paths, load_image_256, batch_size, workers=workers,
        prefetch_batches=prefetch_batches, pin_memory=pin,
    ):
        batch = batch_cpu.to(device, non_blocking=pin)
        views = spatial_views(batch)
        p = view_probs(model, views)
        probs[i:i + count] = p.cpu().numpy().astype(np.float16)
    return probs


# --------------------------------------------------------------------------- #
# disk io
# --------------------------------------------------------------------------- #
def condition_path(model_name, corruption, severity):
    """'cache/<model>/<corruption>/<severity>.npz' (severity forced to 0 for clean)."""
    sev = "0" if corruption == "clean" else str(severity)
    return CACHE_ROOT / model_name / corruption / f"{sev}.npz"


def save_condition(model_name, corruption, severity, *, probs, cutoffs, labels,
                   paths, energies=ENERGIES, ladder=None):
    """Write one condition to a compressed .npz. Returns the path written.

    `energies` is meaningful only for the adaptive energy ladder.  Fixed and
    geometric caches can pass energies=None; their actual per-view cutoffs are
    already stored in `cutoffs`.  `ladder` is optional metadata and does not
    affect older readers.
    """
    out = condition_path(model_name, corruption, severity)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        probs=probs,
        cutoffs=cutoffs,
        labels=np.asarray(labels, dtype=np.int32),
        paths=np.asarray([str(p) for p in paths]),
    )
    if energies is not None:
        payload["energies"] = np.asarray(energies, dtype=np.float32)
    if ladder is not None:
        payload["ladder"] = np.asarray(str(ladder))
    np.savez_compressed(out, **payload)
    return out


def load_condition(model_name, corruption, severity):
    """Load one cached condition.

    Returns:
        dict with 'probs' (N, K+1, 1000) float16, 'labels' (N,) int32,
        'paths' (N,) str, and 'cutoffs' (N, K) float32 / 'energies' (K,)
        float32 -- both None for a cache saved without a filter ladder (e.g.
        the spatial-TTA or spectral+spatial combined caches, which reuse this
        loader's schema but only carry probs/labels/paths).
    """
    d = np.load(condition_path(model_name, corruption, severity), allow_pickle=False)
    return {
        "probs": d["probs"],
        "cutoffs": d["cutoffs"] if "cutoffs" in d.files else None,
        "labels": d["labels"],
        "paths": d["paths"],
        "energies": d["energies"] if "energies" in d.files else None,
        "ladder": str(d["ladder"]) if "ladder" in d.files else None,
        "ladder_len": int(d["ladder_len"]) if "ladder_len" in d.files else None,
        "view_names": d["view_names"] if "view_names" in d.files else None,
    }


def has_condition(model_name, corruption, severity):
    """True if this condition has already been cached."""
    return condition_path(model_name, corruption, severity).exists()


def save_spatial_condition(model_name, corruption, severity, *, probs, labels, paths,
                           cutoffs=None, ladder=None, ladder_len=None, view_names=None):
    """Write a generic multi-view cache condition.

    Spatial-only caches normally omit the optional metadata.  Combined caches
    should preserve the spectral ladder metadata so downstream reporting can
    label views from the cache rather than hard-coding fixed cutoffs.
    """
    out = condition_path(model_name, corruption, severity)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        probs=probs,
        labels=np.asarray(labels, dtype=np.int32),
        paths=np.asarray([str(p) for p in paths]),
    )
    if cutoffs is not None:
        payload["cutoffs"] = np.asarray(cutoffs)
    if ladder is not None:
        payload["ladder"] = np.asarray(str(ladder))
    if ladder_len is not None:
        payload["ladder_len"] = np.asarray(int(ladder_len), dtype=np.int32)
    if view_names is not None:
        payload["view_names"] = np.asarray([str(x) for x in view_names])
    np.savez_compressed(out, **payload)
    return out


def load_spatial_condition(model_name, corruption, severity):
    """Load one cached generic multi-view condition, including optional metadata."""
    d = np.load(condition_path(model_name, corruption, severity), allow_pickle=False)
    return {
        "probs": d["probs"],
        "labels": d["labels"],
        "paths": d["paths"],
        "cutoffs": d["cutoffs"] if "cutoffs" in d.files else None,
        "ladder": str(d["ladder"]) if "ladder" in d.files else None,
        "ladder_len": int(d["ladder_len"]) if "ladder_len" in d.files else None,
        "view_names": d["view_names"] if "view_names" in d.files else None,
    }


def profile_view_labels(record):
    """Return human-readable view labels from cached metadata when available.

    This avoids hard-coding fixed-ladder cutoffs in reporting scripts.  For an
    adaptive ladder, where cutoffs vary by image, generic spectral labels are
    returned instead of pretending one cutoff applies to every image.
    """
    if record.get("view_names") is not None:
        return [str(x) for x in record["view_names"]]

    n_views = int(record["probs"].shape[1])
    labels = ["original"]
    cutoffs = record.get("cutoffs")
    if cutoffs is not None:
        c = np.asarray(cutoffs)
        if c.ndim == 2 and len(c) and np.allclose(c, c[:1]):
            vals = c[0]
            labels += [f"lowpass_{float(x):.6g}" for x in vals]
        elif c.ndim == 1:
            labels += [f"lowpass_{float(x):.6g}" for x in c]
        else:
            labels += [f"spectral_{i}" for i in range(1, min(n_views, 1 + c.shape[-1]))]
    while len(labels) < n_views:
        labels.append(f"view_{len(labels)}")
    return labels[:n_views]
