"""Frozen pretrained classifiers.

All three are loaded in eval mode with gradients disabled: the method never
updates any weights, it only reads the classifier's response to filtered views.
The matching preprocessing transform is returned alongside each model, though
the experiment scripts build the [0,1] -> filter -> normalise pipeline by hand
so every architecture sees identical pixels. All three are ImageNet-1k-only
checkpoints (no extra pretraining data), so cross-architecture comparisons
aren't confounded by one model having seen more data than the others.
"""
import torch
import timm

from torchvision.models import resnet18, ResNet18_Weights


def load_resnet18(device):
    """Load a frozen ImageNet-pretrained ResNet-18.

    Args:
        device: torch device to move the model onto.

    Returns:
        (model, transform) where 'model' is in eval mode with all parameters
        frozen and 'transform' is the torchvision preprocessing for these
        weights (Resize 256 -> CenterCrop 224 -> ToTensor -> ImageNet
        normalise).
    """
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)
    model.eval()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    transform = weights.transforms()

    return model, transform


def load_deit_tiny(device):
    """Load a frozen ImageNet-pretrained DeiT-Tiny (timm 'deit_tiny_patch16_224').

    Downloads the weights from the timm hub on first call.

    Args:
        device: torch device to move the model onto.

    Returns:
        (model, transform) where 'model' is in eval mode with all parameters
        frozen and 'transform' is the timm eval preprocessing derived from the
        model's data config (same 256/224 geometry and ImageNet normalisation
        as ResNet-18).
    """
    model = timm.create_model(
        "deit_tiny_patch16_224.fb_in1k",
        pretrained=True
    )

    model.eval()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    config = timm.data.resolve_model_data_config(model)

    transform = timm.data.create_transform(
        **config,
        is_training=False
    )

    return model, transform


def load_convnext_tiny(device):
    """Load a frozen ImageNet-pretrained ConvNeXt-Tiny (timm 'convnext_tiny.fb_in1k').

    A modernised CNN (large-kernel depthwise convs, LayerNorm, GELU, a
    Transformer-influenced macro-design) trained on ImageNet-1k only, like
    ResNet-18 and DeiT-Tiny. A modern CNN without attention, used to tell
    whether a difference between ResNet-18 and DeiT-Tiny is about attention
    or about the architecture/training recipe more broadly.

    Args:
        device: torch device to move the model onto.

    Returns:
        (model, transform) where 'model' is in eval mode with all parameters
        frozen and 'transform' is the timm eval preprocessing derived from the
        model's data config (same 224 input size and ImageNet normalisation
        as the other two models).
    """
    model = timm.create_model(
        "convnext_tiny.fb_in1k",
        pretrained=True
    )

    model.eval()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    config = timm.data.resolve_model_data_config(model)

    transform = timm.data.create_transform(
        **config,
        is_training=False
    )

    return model, transform
