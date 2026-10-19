"""Method pipeline schematic for the report (Figure 1).

Builds the views with the same operations as the code base:
  * spectral views: Gaussian low-pass mask exp(-(u^2+v^2) / (2 q^2)) on the
    centred DFT, (u, v) in [-1, 1]^2 (src/filters.py::radial_lowpass_mask);
  * spatial views: original, h-flip, four 196-px corner crops resized back
    to 224, and flips of the top-left / bottom-right crops (src/cache.py).
The input is a stock sample image with synthetic Gaussian noise, used only
for illustration (not an ImageNet-C image).

Usage:  python scripts/make_pipeline_figure.py [out_dir]
Writes  pipeline.pdf (vector) and pipeline.png (300 dpi).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
from skimage import data

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "FreeSerif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "pdf.fonttype": 42,
})

INK = "#222222"
MUTED = "#666666"
SPEC = "#2a6fb0"     # spectral branch accent
SPAT = "#d4692a"     # spatial branch accent
FILL = "#f6f6f6"
QS = [0.90, 0.75, 0.60, 0.45, 0.30, 0.20, 0.12]
N = 224
CROP = 196


# ----------------------------------------------------------------- images ---
def load_input():
    img = data.astronaut()[30:330, 90:390]                 # face + suit
    img = np.asarray(Image.fromarray(img).resize((N, N), Image.BICUBIC))
    img = img.astype(np.float32) / 255.0
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0, 0.16, img.shape), 0, 1)  # noise corruption


def lowpass(img, q):
    u = np.linspace(-1, 1, N)
    uu, vv = np.meshgrid(u, u, indexing="ij")
    mask = np.exp(-(uu ** 2 + vv ** 2) / (2 * q ** 2))
    out = np.empty_like(img)
    for c in range(3):
        F = np.fft.fftshift(np.fft.fft2(img[..., c]))
        out[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))
    return np.clip(out, 0, 1)


def resize(a):
    im = Image.fromarray((a * 255).astype(np.uint8)).resize((N, N), Image.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0


def spatial_views(img):
    c, e = CROP, N - CROP
    tl, tr = resize(img[:c, :c]), resize(img[:c, e:])
    bl, br = resize(img[e:, :c]), resize(img[e:, e:])
    flip = lambda a: a[:, ::-1]
    return [img, flip(img), tl, tr, bl, br, flip(tl), flip(br)]


x = load_input()
spec = [x] + [lowpass(x, q) for q in QS]
spat = spatial_views(x)
spec_lab = ["orig."] + [f"{q:.2f}".lstrip("0") for q in QS]
spat_lab = ["orig.", "flip", "TL", "TR", "BL", "BR", "TL+f", "BR+f"]

# ----------------------------------------------------------------- canvas ---
W, H = 72.0, 17.2                          # units of 0.1 inch
fig = plt.figure(figsize=(W / 10, H / 10))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x0, y0, w, h, ec=INK, fc="white", lw=0.8, ls="-", r=0.6):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=ec, fc=fc, lw=lw, ls=ls))


def arrow(x0, y0, x1, y1, c=INK):
    ax.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=c,
                                shrinkA=0, shrinkB=0, mutation_scale=6))


def thumb(img, x0, y0, s, ec=None, ls="-"):
    ax.imshow(img, extent=(x0, x0 + s, y0, y0 + s), interpolation="lanczos", zorder=3)
    if ec:
        ax.add_patch(Rectangle((x0, y0), s, s, fill=False, ec=ec, lw=0.9, ls=ls, zorder=4))


# input image
S_IN = 6.6
yin = 6.0
thumb(x, 0.4, yin, S_IN, ec=INK)
ax.text(0.4 + S_IN / 2, yin - 0.7, "corrupted test\nimage $x$", ha="center", va="top")

# two branches
BX, BW, S, G = 9.0, 29.6, 3.3, 0.25
x_th0 = BX + (BW - (8 * S + 7 * G)) / 2


def branch(y0, h, color, title, views, labels, note):
    box(BX, y0, BW, h, ec=color, lw=0.9)
    ax.text(BX + 0.6, y0 + h - 0.45, title, ha="left", va="top", weight="bold", color=color)
    ty = y0 + 1.3
    for i, (im, lb) in enumerate(zip(views, labels)):
        xi = x_th0 + i * (S + G)
        shared = i == 0
        thumb(im, xi, ty, S, ec=MUTED if shared else None, ls=(0, (2, 1)) if shared else "-")
        ax.text(xi + S / 2, ty - 0.3, lb, ha="center", va="top", fontsize=6.6, color=INK)
    ax.text(BX + BW - 0.6, y0 + h - 0.45, note, ha="right", va="top", fontsize=6.6, color=MUTED)
    return ty


ty_spec = branch(10.6, 6.3, SPEC, "Spectral profile (8 views)", spec, spec_lab,
                 r"ordered Gaussian low-pass ladder, cutoff $q$")
branch(2.2, 6.3, SPAT, "Spatial TTA (8 views)", spat, spat_lab,
       "flip + resized 196-px corner crops")

# fan-out arrows from input
arrow(0.4 + S_IN, yin + S_IN * 0.75, BX, 13.75)
arrow(0.4 + S_IN, yin + S_IN * 0.25, BX, 5.35)

# shared original note between branches
ax.text(x_th0, 9.75, "original is shared by both branches  →  15 unique views in the combined set", ha="left", va="center",
        fontsize=6.3, color=MUTED, style="italic", transform=ax.transData)

# frozen classifier (+ predictions)
CX, CW = 40.6, 8.6
box(CX, 3.4, CW, 11.0, fc=FILL)
ax.text(CX + CW / 2, 13.6, "frozen\nclassifier $f$", ha="center", va="top", weight="bold",
        linespacing=1.1)
ax.text(CX + CW / 2, 10.9, "same weights,\nno updates", ha="center", va="top",
        fontsize=6.6, color=MUTED, linespacing=1.1)
ax.text(CX + CW / 2, 8.15, r"$p_k=\mathrm{softmax}(f(x_k))$", ha="center", va="center", fontsize=6.6)
# tiny probability bars
for j, (wid, col) in enumerate([(5.2, "#555555"), (3.0, "#8a8a8a"), (1.6, "#b5b5b5")]):
    ax.add_patch(Rectangle((CX + 1.6, 6.9 - j * 0.6), wid, 0.42, fc=col, ec="none"))
ax.text(CX + CW / 2, 3.85, "one $p_k$ per view", ha="center", va="bottom", fontsize=6.3,
        color=MUTED)
arrow(BX + BW, 13.75, CX, 11.2)
arrow(BX + BW, 5.35, CX, 6.6)

# reliability signals
RX, RW = 51.4, 20.2
box(RX, 9.9, RW, 7.0)
ax.text(RX + 0.6, 16.45, "per-image reliability signals (label-free)", ha="left", va="top",
        weight="bold")
rows = [
    (r"$C_k=\max_c\, p_k(c)$", "confidence"),
    (r"$A_k=-\frac{1}{V-1}\Sigma_{j\neq k}\mathrm{JS}(p_k,p_j)$", "agreement"),
    (r"$S_k=-\frac{1}{|N_m(k)|}\Sigma_{j\in N_m(k)}\mathrm{JS}(p_k,p_j)$", "local stability"),
]
for i, (eq, nm) in enumerate(rows):
    yy = 14.65 - i * 1.85
    ax.text(RX + 0.8, yy, eq, ha="left", va="center", fontsize=7.2)
    ax.text(RX + RW - 0.6, yy, nm, ha="right", va="center", fontsize=6.6, color=MUTED, style="italic")

# fusion
FX, FW = 51.4, 20.2
box(FX, 2.2, FW, 6.9, fc=FILL)
ax.text(FX + 0.6, 8.65, "fusion", ha="left", va="top", weight="bold")
ax.text(FX + FW - 0.6, 8.65, r"$b_k$: fixed learned view preference", ha="right", va="top",
        fontsize=6.6, color=MUTED)
ax.text(FX + FW / 2, 6.95, r"$r_k=\alpha\tilde C_k+\beta\tilde A_k+\gamma\tilde S_k+b_k$",
        ha="center", va="center", fontsize=7.4)
ax.text(FX + FW / 2, 5.25, r"$w_k=\mathrm{softmax}_k(r_k/\tau)$", ha="center", va="center",
        fontsize=7.4)
ax.text(FX + FW / 2, 3.45, r"$p^{\ast}=\Sigma_k w_k\,p_k$", ha="center", va="center",
        fontsize=8.2)
ax.add_patch(Rectangle((FX + FW / 2 - 3.4, 2.65), 6.8, 1.6, fill=False, ec=INK, lw=0.8))

arrow(CX + CW, 11.2, RX, 12.6)          # p_k -> signals
arrow(CX + CW, 6.6, FX, 3.45)            # p_k -> weighted sum
arrow(RX + RW / 2, 9.9, FX + FW / 2, 9.1)  # signals -> fusion

# protocol strip
ax.plot([BX, W - 0.4], [1.55, 1.55], color="#b03030", lw=0.6)
ax.text((BX + W - 0.4) / 2, 1.3,
        r"$\alpha,\beta,\gamma,b_k,\tau$ fitted on tuning identities (LOFO)"
        "   |   frozen choice evaluated on held-out identities   |   "
        "no labels, corruption type or weight updates at test time",
        ha="center", va="top", fontsize=6.4, color=INK)

for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    fig.savefig(OUT / f"pipeline.{ext}", **kw)
print("wrote", OUT / "pipeline.pdf")
