"""Model factory."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch


def build_unet(encoder: str = "resnet34", pretrained: bool = True) -> torch.nn.Module:
    """U-Net with a ResNet encoder, 3 input channels and 1 output channel.

    The output is a *logit* map (no sigmoid). Apply `torch.sigmoid` to get
    per-pixel building probabilities; the losses below expect logits.

    pretrained=True initializes the encoder with ImageNet weights (downloaded on
    first use). pretrained=False starts from random weights and works offline.
    """
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
    )
