"""Loss functions."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn


class BCEDiceLoss(nn.Module):
    """BCE (per-pixel) + Dice (overlap) on logits.

    Buildings cover only ~10% of pixels. Plain BCE can be minimized by predicting
    mostly background; Dice directly rewards overlap with the (rare) foreground,
    so the sum balances pixel-level calibration against region-level overlap.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, target) + self.dice(logits, target)
