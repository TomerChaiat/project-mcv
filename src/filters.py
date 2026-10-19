"""Frequency-domain low-pass filtering: the building block of a spectral
stability profile.

A profile is built by low-pass filtering an image at a ladder of strengths and
running each version through the classifier. This module produces those
filtered versions and the per-image cutoffs that define the ladder.

All images are expected in [0, 1] float space, before ImageNet normalisation.
"""
import torch


def radial_lowpass_mask(height, width, cutoff, device=None, dtype=torch.float32):
    """Build a Gaussian low-pass mask for the (fftshifted) Fourier domain.

    The mask is 1 at the centre (DC) and falls off smoothly with radius, so it
    attenuates high frequencies without the ringing a hard cutoff would cause.

    Args:
        height, width: spatial size of the image the mask multiplies.
        cutoff: filter strength in (0, 1]. It is the Gaussian's standard
            deviation in normalised frequency (the image spans [-1, 1] on each
            axis). Smaller = narrower Gaussian = stronger filtering.
        device, dtype: passed through to the created tensors.

    Returns:
        (height, width) real tensor with values in (0, 1].
    """
    if not (0.0 < cutoff <= 1.0):
        raise ValueError("cutoff must be in (0, 1]")

    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r2 = xx ** 2 + yy ** 2
    return torch.exp(-r2 / (2.0 * cutoff ** 2))


def apply_lowpass(image, cutoff):
    """Low-pass filter an image in the Fourier domain.

    Forward FFT, multiply by 'radial_lowpass_mask(cutoff)' (same mask on every
    channel), inverse FFT, take the real part and clamp back to [0, 1].

    Args:
        image: (C, H, W) or (B, C, H, W) tensor in [0, 1].
        cutoff: filter strength in (0, 1]; see 'radial_lowpass_mask'.

    Returns:
        Filtered tensor, same shape and range as 'image'.
    """
    if image.ndim not in (3, 4):
        raise ValueError("expected (C, H, W) or (B, C, H, W)")

    h, w = image.shape[-2:]
    mask = radial_lowpass_mask(h, w, cutoff, device=image.device, dtype=image.dtype)

    spectrum = torch.fft.fft2(image, dim=(-2, -1))
    spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
    spectrum = spectrum * mask
    spectrum = torch.fft.ifftshift(spectrum, dim=(-2, -1))
    filtered = torch.fft.ifft2(spectrum, dim=(-2, -1)).real

    # the inverse FFT can overshoot [0, 1] by a rounding error
    return filtered.clamp(0.0, 1.0)


def make_spectral_views(image, cutoffs):
    """Stack the original image with a series of low-passed copies.

    View 0 is the untouched image. Views 1..K are 'image' filtered at
    'cutoffs[0] .. cutoffs[K-1]' (each filter is applied to the original, not
    cascaded). Feeding this stack to a classifier gives the spectral stability
    profile S(x).

    Args:
        image: (C, H, W) or (B, C, H, W) tensor in [0, 1].
        cutoffs: iterable of K cutoff values, typically decreasing
            (progressively stronger filtering).

    Returns:
        (K+1, C, H, W) for a single image, or (B, K+1, C, H, W) for a batch.
    """
    if image.ndim not in (3, 4):
        raise ValueError("expected (C, H, W) or (B, C, H, W)")

    views = [image]
    for c in cutoffs:
        views.append(apply_lowpass(image, float(c)))

    return torch.stack(views, dim=0 if image.ndim == 3 else 1)


def energy_retained(image, cutoff):
    """Fraction of the image's spectral energy that survives the low-pass.

    Energy is the sum of squared FFT magnitudes. Monotonically increasing in
    'cutoff' (a wider mask keeps more), which is what makes 'cutoff_for_energy'
    solvable by bisection.

    Args:
        image: (C, H, W) or (B, C, H, W) tensor.
        cutoff: filter strength in (0, 1].

    Returns:
        Python float in (0, 1].
    """
    spectrum = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1))
    h, w = image.shape[-2:]
    mask = radial_lowpass_mask(h, w, cutoff, device=image.device, dtype=image.dtype)
    total = (spectrum.abs() ** 2).sum()
    kept = ((spectrum * mask).abs() ** 2).sum()
    return float((kept / total).item())


def cutoff_for_energy(image, target, lo=1e-3, hi=1.0, iters=40):
    """Solve for the cutoff whose low-pass retains 'target' of the image's energy.

    Bisection on 'energy_retained', which is monotonic in the cutoff. Solving
    per image (rather than using fixed cutoffs) makes the ladder comparable
    across images with different spectral content.

    Args:
        image: (C, H, W) or (B, C, H, W) tensor.
        target: desired retained-energy fraction in (0, 1].
        lo, hi: bracket for the search; the answer is assumed to lie in [lo, hi].
        iters: bisection steps (40 gives ~1e-12 precision).

    Returns:
        Python float cutoff in [lo, hi].
    """
    if not (0.0 < target <= 1.0):
        raise ValueError("target must be in (0, 1]")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if energy_retained(image, mid) < target:
            lo = mid   # kept too little -> need a wider mask
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cutoffs_for_energy_ladder(image, targets):
    """Solve 'cutoff_for_energy' for each target -> the ladder for this image.

    Args:
        image: (C, H, W) or (B, C, H, W) tensor.
        targets: iterable of retained-energy fractions, e.g.
            [0.99, 0.98, ... 0.93]. Decreasing targets give increasing filter
            strength.

    Returns:
        list of float cutoffs, one per target, in the same order.
    """
    return [cutoff_for_energy(image, float(t)) for t in targets]
