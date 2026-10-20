"""
Smoke test: load the three classifiers and their transforms, print success.
Run once after install to check the pretrained weights download and load.
Unlike test_analysis.py, this one needs network access (first run downloads
the weights) and will use a GPU/MPS device if one is available, though CPU
also works.

run: PYTHONPATH=. python test/test_models.py
"""
import torch

from src.models import load_convnext_tiny, load_deit_tiny, load_resnet18


def main():
    """Smoke test: load ResNet-18, DeiT-Tiny and ConvNeXt-Tiny on the best available device."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    print("\nLoading ResNet-18...")
    resnet, resnet_transform = load_resnet18(device)
    print("ResNet-18 loaded successfully")

    print("\nLoading DeiT-Tiny...")
    deit, deit_transform = load_deit_tiny(device)
    print("DeiT-Tiny loaded successfully")

    print("\nLoading ConvNeXt-Tiny...")
    convnext, convnext_transform = load_convnext_tiny(device)
    print("ConvNeXt-Tiny loaded successfully")

    print("\nDone!")


# a script, not a pytest test: guarded so `pytest test` never downloads weights
if __name__ == "__main__":
    main()
