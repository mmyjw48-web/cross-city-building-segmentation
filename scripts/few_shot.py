#!/usr/bin/env python
"""Few-shot adaptation: fine-tune on N images of the target city, evaluate on its val split.

Compare a source-trained model against the ImageNet-only control, with the SAME
learning rate and epochs for both:

    python scripts/few_shot.py --target-city khartoum --init-ckpt runs/paris_global_seed0/best.pt --seeds 0 1 2
    python scripts/few_shot.py --target-city khartoum --init-ckpt imagenet                      --seeds 0 1 2

The seeds given here control which N images are drawn and the fine-tuning
randomness. They are independent of the seed the source checkpoint was trained with.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ccbs.data import BuildingsDataset, CITIES, NORMALIZATIONS, split_dataset  # noqa: E402
from ccbs.losses import BCEDiceLoss  # noqa: E402
from ccbs.metrics import evaluate  # noqa: E402
from ccbs.model import build_unet  # noqa: E402
from ccbs.train import fine_tune_and_eval  # noqa: E402
from ccbs.utils import environment_info, get_device, save_json, set_seed  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-city", required=True, choices=list(CITIES))
    p.add_argument("--init-ckpt", required=True,
                   help="path to a source-trained best.pt, or the word 'imagenet' for the control")
    p.add_argument("--n-images", type=int, nargs="+", default=[0, 10, 25, 50, 100, 200],
                   help="0 = no fine-tuning (zero-shot evaluation of the init)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--data-root", default="data")
    p.add_argument("--norm", choices=NORMALIZATIONS, default="global")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true", help="random encoder init (smoke tests)")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out", default=None, help="output JSON (default results/runs/fewshot_<target>_<init>.json)")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    device = get_device(a.device)
    loss_fn = BCEDiceLoss()
    ds = BuildingsDataset(a.data_root, a.target_city, norm=a.norm, max_images=a.max_images)
    train, val = split_dataset(ds, seed=a.split_seed)
    val_loader = DataLoader(val, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)

    if a.init_ckpt == "imagenet":
        init_state, init_name = None, "imagenet"
    else:
        init_state = torch.load(a.init_ckpt, map_location="cpu")
        init_name = "source"

    rows = []
    for n in a.n_images:
        for seed in a.seeds:
            if n == 0:
                # zero-shot: evaluate the init as is (an ImageNet-only U-Net has a random decoder)
                set_seed(seed)
                m = build_unet(pretrained=init_state is None and not a.no_pretrained)
                if init_state is not None:
                    m.load_state_dict(init_state)
                m.to(device)
                metrics = evaluate(m, val_loader, device)
            else:
                metrics = fine_tune_and_eval(
                    init_state, train, val_loader, n_images=n, seed=seed, epochs=a.epochs,
                    lr=a.lr, batch_size=a.batch_size, device=device, loss_fn=loss_fn,
                    num_workers=a.num_workers, pretrained_encoder=not a.no_pretrained)
            rows.append({**metrics, "n_images": n, "seed": seed})  # n_images = # TRAINING images
            print(f"{init_name:>8} | N={n:>3} | seed {seed} | val IoU {metrics['iou']:.3f}", flush=True)

    out = a.out or f"results/runs/fewshot_{a.target_city}_{init_name}.json"
    save_json({"config": {**vars(a), "init": init_name}, "rows": rows,
               "environment": environment_info()}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
