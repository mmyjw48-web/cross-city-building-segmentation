"""Evaluation metrics.

All pixel counts are accumulated over the *whole* loader and the ratios are
computed once at the end ("dataset-level" IoU). This differs from averaging a
per-image IoU: images with few or no buildings would otherwise get a noisy or
undefined score and dominate the mean.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None, threshold: float = 0.5) -> dict:
    """Dataset-level IoU / precision / recall (+ optional mean loss).

    The model sees the inputs exactly as the loader produced them: no clamping or
    rescaling here (an earlier notebook version clamped inputs to [0, 1], which
    silently broke z-score-normalized data).
    """
    model.eval()
    tp = pred_sum = true_sum = pixels = 0.0
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if loss_fn is not None:
            total_loss += loss_fn(logits, y).item() * len(x)
        n += len(x)
        pred = (torch.sigmoid(logits) > threshold).float()
        tp += (pred * y).sum().item()
        pred_sum += pred.sum().item()
        true_sum += y.sum().item()
        pixels += y.numel()

    union = pred_sum + true_sum - tp
    out = {
        "iou": tp / max(union, 1.0),
        "precision": tp / max(pred_sum, 1.0),
        "recall": tp / max(true_sum, 1.0),
        "pred_frac": pred_sum / max(pixels, 1.0),   # fraction of pixels predicted as building
        "true_frac": true_sum / max(pixels, 1.0),   # fraction of pixels that truly are buildings
        "n_eval_images": n,
    }
    if loss_fn is not None:
        out["loss"] = total_loss / max(n, 1)
    return out


@torch.no_grad()
def threshold_sweep(model, loader, device, thresholds=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99)) -> dict:
    """IoU as a function of the probability threshold (one forward pass).

    Low thresholds matter for cross-city evaluation: a model whose probabilities
    are shifted down on the new city can be nearly silent at 0.5 yet still rank
    building pixels well.
    """
    model.eval()
    tp = {t: 0.0 for t in thresholds}
    ps = {t: 0.0 for t in thresholds}
    ts = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        prob = torch.sigmoid(model(x))
        ts += y.sum().item()
        for t in thresholds:
            pred = (prob > t).float()
            tp[t] += (pred * y).sum().item()
            ps[t] += pred.sum().item()
    return {t: tp[t] / max(ps[t] + ts - tp[t], 1.0) for t in thresholds}
