"""
Smoke test: load both classifiers and their transforms, print success.
Run once after install to check the pretrained weights download and load.
Unlike test_analysis.py, this one needs network access (first run downloads
the weights) and will use a GPU/MPS device if one is available, though CPU
also works.

run: PYTHONPATH=. python test/test_models.py
"""
import torch

from src.models import load_resnet18, load_deit_tiny


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

print("\nDone!")