#!/usr/bin/env python
"""Evaluate a saved checkpoint on one city (optionally with a threshold sweep).

    python scripts/evaluate.py --ckpt runs/paris_global_seed0/best.pt --city khartoum --split all --sweep
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ccbs.data import BuildingsDataset, CITIES, NORMALIZATIONS, split_dataset  # noqa: E402
from ccbs.losses import BCEDiceLoss  # noqa: E402
from ccbs.metrics import evaluate, threshold_sweep  # noqa: E402
from ccbs.model import build_unet  # noqa: E402
from ccbs.utils import get_device  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--city", required=True, choices=list(CITIES))
    p.add_argument("--split", choices=["train", "val", "all"], default="val",
                   help="'val'/'train' use the fixed 80/20 split of that city; 'all' uses every image")
    p.add_argument("--data-root", default="data")
    p.add_argument("--norm", choices=NORMALIZATIONS, default="global",
                   help="must match the normalization the checkpoint was trained with")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--sweep", action="store_true", help="also report IoU at several thresholds")
    p.add_argument("--out", default=None, help="also write the result JSON here")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    device = get_device(a.device)
    ds = BuildingsDataset(a.data_root, a.city, norm=a.norm, max_images=a.max_images)
    if a.split != "all":
        train, val = split_dataset(ds, seed=a.split_seed)
        ds = train if a.split == "train" else val
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)

    model = build_unet(pretrained=False).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))

    out = {"ckpt": a.ckpt, "run_name": Path(a.ckpt).parent.name, "city": a.city,
           "split": a.split, "norm": a.norm,
           "metrics": evaluate(model, loader, device, BCEDiceLoss())}
    if a.sweep:
        sweep = threshold_sweep(model, loader, device)
        out["threshold_sweep"] = {str(k): v for k, v in sweep.items()}
        # ORACLE: picks the threshold using this split's labels. Use it only to diagnose
        # calibration shift (ranking quality vs. threshold), never as a deployable result.
        best_t = max(sweep, key=sweep.get)
        out["oracle_best_threshold"] = {"threshold": best_t, "iou": sweep[best_t]}
    print(json.dumps(out, indent=2))
    if a.out:
        from ccbs.utils import save_json
        save_json(out, a.out)


if __name__ == "__main__":
    main()
