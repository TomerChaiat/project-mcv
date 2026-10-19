"""Spectral stability profile for a single image.

Builds the energy-ladder view stack, runs each view through a frozen
classifier, and reports the per-view predictions, the three raw signals
(C_k, A_k, S_k) and the fused prediction under a chosen config. Saves two
figures: the view stack and the signal curves.

    python scripts/profile_single_image.py --image <ImageNet-C image> [--model deit]
    python scripts/profile_single_image.py --image <img> --beta 0 --gamma 0 --tau 0.05
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src import analysis
from src.filters import make_spectral_views, cutoffs_for_energy_ladder
from src.models import load_resnet18, load_deit_tiny
from src.imagenet_labels import class_name, true_index_from_path

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main():
    """Profile one image: build views, run the model, print signals, save figures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", choices=["resnet18", "deit"], default="resnet18")
    ap.add_argument("--output", default="figures/profile.png")
    ap.add_argument("--energies", type=float, nargs="+",
                    default=list(np.arange(0.995, 0.92, -0.0025)),
                    help="retained-energy target per view (sets the ladder)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--m", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("device:", device)
    model = (load_resnet18 if args.model == "resnet18" else load_deit_tiny)(device)[0]

    # image -> [0,1] tensor, standard ImageNet eval geometry
    pil = Image.open(args.image).convert("RGB")
    pil = TF.center_crop(TF.resize(pil, 256), [224, 224])
    image = TF.to_tensor(pil)

    # energy-spaced low-pass ladder, then the view stack
    cutoffs = cutoffs_for_energy_ladder(image, args.energies)
    print("solved cutoffs:", [round(c, 4) for c in cutoffs])
    views = make_spectral_views(image, cutoffs=cutoffs).to(device)

    with torch.no_grad():
        logits = model(TF.normalize(views, MEAN, STD))
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[None]   # (1, K+1, C)

    # raw signals (for the printed table and the plot)
    C, A, S = analysis.profile_signals(probs, m=args.m, standardize=False)

    # fused prediction under the chosen config
    if args.no_standardize:
        Cs, As, Ss = C, A, S
    else:
        Cs, As, Ss = analysis.zscore(C), analysis.zscore(A), analysis.zscore(S)
    scores = analysis.reliability(Cs, As, Ss, args.alpha, args.beta, args.gamma)
    weights = analysis.fusion_weights(scores, tau=args.tau)
    fused = analysis.fuse(probs, weights)[0]                    # (C,)

    C, A, S, scores, weights = C[0], A[0], S[0], scores[0], weights[0]
    preds = probs[0].argmax(-1)
    fused_pred = int(fused.argmax())

    true_idx, wnid = true_index_from_path(args.image)
    if true_idx is not None:
        print(f"true class: {true_idx} = {class_name(true_idx)}  ({wnid})")
    else:
        print("true class: unknown (path is not ImageNet-C style)")

    ladder = ["original"] + [f"E={e:.4f}" for e in args.energies]

    # ---- per-view table ----
    print(f"\n{'view':>10} {'C_k':>8} {'A_k':>8} {'S_k':>8} "
          f"{'weight':>8} {'pred':>6}  class")
    for name, ck, ak, sk, wk, pk in zip(ladder, C, A, S, weights, preds):
        mark = "*" if (true_idx is not None and pk == true_idx) else " "
        print(f"{name:>10} {ck:>8.4f} {ak:>8.4f} {sk:>8.4f} {wk:>8.4f} "
              f"{pk:>6d}{mark} {class_name(int(pk))}")
    if true_idx is not None:
        print(f"\n{int((preds == true_idx).sum())}/{len(preds)} views correct; "
              f"{int((preds[1:] == preds[0]).sum())}/{len(preds) - 1} agree with the original")

    # ---- fusion result ----
    print(f"\nfused prediction: {fused_pred} = {class_name(fused_pred)}  "
          f"(p={fused.max():.4f})", end="")
    if true_idx is not None:
        print(f"  {'correct' if fused_pred == true_idx else 'wrong'}")
    else:
        print()

    # ---- figures ----
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    v = views.detach().cpu()
    n = len(ladder)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax, view, name in zip(np.ravel(axes), v, ladder):
        ax.imshow(view.permute(1, 2, 0).clamp(0, 1))
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    views_out = Path(args.output).with_name(Path(args.output).stem + "_views.png")
    fig.savefig(views_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", views_out)

    x = range(n)
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for ax, sig, name in zip(axes, [C, A, S], ["C_k", "A_k", "S_k"]):
        ax.plot(x, sig, "o-")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(ladder, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print("saved", args.output)


if __name__ == "__main__":
    main()
