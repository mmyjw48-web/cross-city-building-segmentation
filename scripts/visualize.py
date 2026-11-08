#!/usr/bin/env python
"""Qualitative results: image / ground truth / probability map / error map for random tiles.

    python scripts/visualize.py --ckpt runs/khartoum_global_seed0/best.pt --city paris --n 6 \
        --out figures/qualitative_khartoum_seed0_on_paris.png

Tiles are sampled at random with a fixed --seed (not hand-picked), so the figure is
not a cherry-picked best case. Runs fine on CPU for a handful of tiles.

Error map colours:  green = true positive, red = false positive (predicted, not labelled),
                    blue = false negative (labelled, missed).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ccbs.data import BuildingsDataset, CITIES, NORMALIZATIONS  # noqa: E402
from ccbs.model import build_unet  # noqa: E402
from ccbs.utils import get_device  # noqa: E402


def stretch(rgb: np.ndarray) -> np.ndarray:
    """Contrast stretch (2-98 percentile) for display only."""
    lo, hi = np.percentile(rgb, [2, 98])
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def error_map(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    out = np.zeros(pred.shape + (3,), dtype="float32")
    out[(pred == 1) & (truth == 1)] = (0.2, 0.7, 0.3)  # TP green
    out[(pred == 1) & (truth == 0)] = (0.85, 0.2, 0.2)  # FP red
    out[(pred == 0) & (truth == 1)] = (0.2, 0.4, 0.85)  # FN blue
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--city", required=True, choices=list(CITIES), help="city to draw tiles from")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=0, help="seed for the random tile selection")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--norm", choices=NORMALIZATIONS, default="global")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    device = get_device(a.device)
    ds = BuildingsDataset(a.data_root, a.city, norm=a.norm)
    idx = sorted(np.random.default_rng(a.seed).choice(len(ds), size=min(a.n, len(ds)), replace=False))

    model = build_unet(pretrained=False).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()

    run = Path(a.ckpt).parent.name
    fig, axes = plt.subplots(len(idx), 4, figsize=(13, 3.2 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        x, y = ds[i]
        with torch.no_grad():
            prob = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
        truth = y[0].numpy().astype("uint8")
        pred = (prob > a.threshold).astype("uint8")
        inter = int(((pred == 1) & (truth == 1)).sum())
        union = int(((pred == 1) | (truth == 1)).sum())
        iou = f"IoU {inter / union:.2f}" if union else "no buildings, none predicted"

        axes[r, 0].imshow(stretch(x.permute(1, 2, 0).numpy()))
        axes[r, 0].set_title(f"{ds.ids[i]}", fontsize=9)
        axes[r, 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
        axes[r, 1].set_title(f"truth ({truth.mean():.1%} building)", fontsize=9)
        axes[r, 2].imshow(prob, cmap="viridis", vmin=0, vmax=1)
        axes[r, 2].set_title(f"probability (max {prob.max():.2f})", fontsize=9)
        axes[r, 3].imshow(error_map(pred, truth))
        axes[r, 3].set_title(f"@{a.threshold:g}: {pred.mean():.1%} predicted, {iou}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{run}  ->  {a.city} (random tiles, seed {a.seed}); error map: green TP, red FP, blue FN",
                 fontsize=10)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
