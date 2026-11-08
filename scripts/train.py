#!/usr/bin/env python
"""Train a U-Net on one city, then evaluate in-city and on the other city.

Example (the baseline, three seeds):

    for s in 0 1 2; do
      python scripts/train.py --train-city paris --seed $s
      python scripts/train.py --train-city khartoum --seed $s
    done

Outputs go to  <out-dir>/<run-name>/  (best.pt, last.pt, result.json).
Re-running the same command resumes an interrupted run from last.pt.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader  # noqa: E402

from ccbs.data import BuildingsDataset, CITIES, NORMALIZATIONS, other_city, split_dataset  # noqa: E402
from ccbs.losses import BCEDiceLoss  # noqa: E402
from ccbs.metrics import evaluate  # noqa: E402
from ccbs.model import build_unet  # noqa: E402
from ccbs.train import fit  # noqa: E402
from ccbs.utils import environment_info, get_device, save_json, set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--train-city", required=True, choices=list(CITIES))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0, help="training seed (the data split seed is fixed)")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--norm", choices=NORMALIZATIONS, default="global")
    p.add_argument("--aug-brightness", action="store_true", help="random brightness/colour gain (global norm only)")
    p.add_argument("--no-pretrained", action="store_true", help="random encoder init (offline / smoke tests)")
    p.add_argument("--no-monitor", action="store_true", help="do not evaluate the other city each epoch")
    p.add_argument("--max-images", type=int, default=None, help="use only the first N images per city (smoke test)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--mask-cache", default=None, help="directory for cached rasterized masks")
    p.add_argument("--out-dir", default="runs", help="checkpoints + result.json (git-ignored)")
    p.add_argument("--results-dir", default="results/runs", help="small result JSON copy (safe to commit)")
    p.add_argument("--device", default=None)
    return p.parse_args()


def run_name(a) -> str:
    name = f"{a.train_city}_{a.norm}"
    if a.aug_brightness:
        name += "_bright"
    if a.no_pretrained:
        name += "_scratch"
    return f"{name}_seed{a.seed}"


def main():
    a = parse_args()
    if a.aug_brightness and a.norm != "global":
        raise SystemExit("--aug-brightness only makes sense with --norm global")

    device = get_device(a.device)
    src, tgt = a.train_city, other_city(a.train_city)
    name = run_name(a)
    out = Path(a.out_dir) / name
    print(f"run: {name} | device: {device}")

    def make(city):
        return BuildingsDataset(a.data_root, city, norm=a.norm,
                                mask_cache_dir=a.mask_cache, max_images=a.max_images)

    src_train, src_val = split_dataset(make(src), seed=a.split_seed)
    tgt_ds = make(tgt)
    _, tgt_val = split_dataset(tgt_ds, seed=a.split_seed)

    def loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=a.num_workers)

    set_seed(a.seed)
    model = build_unet(pretrained=not a.no_pretrained).to(device)
    loss_fn = BCEDiceLoss()

    model, history, best_epoch = fit(
        model, loader(src_train, shuffle=True), loader(src_val),
        epochs=a.epochs, lr=a.lr, device=device, loss_fn=loss_fn,
        brightness=a.aug_brightness,
        monitor_loader=None if a.no_monitor else loader(tgt_ds),
        ckpt_dir=out,
    )

    metrics = {
        "source_val": evaluate(model, loader(src_val), device, loss_fn),
        "target_all": evaluate(model, loader(tgt_ds), device, loss_fn),
        "target_val": evaluate(model, loader(tgt_val), device, loss_fn),
    }
    result = {"config": {**vars(a), "run_name": name, "source": src, "target": tgt},
              "n_train": len(src_train), "n_val": len(src_val), "n_target": len(tgt_ds),
              "best_epoch": best_epoch, "history": history, "metrics": metrics,
              "environment": environment_info()}
    save_json(result, out / "result.json")
    save_json(result, Path(a.results_dir) / f"{name}.json")

    print(f"\n{src} val IoU: {metrics['source_val']['iou']:.3f} | "
          f"{tgt} (all {len(tgt_ds)} images) IoU: {metrics['target_all']['iou']:.3f}")
    print(f"saved {out / 'result.json'}")


if __name__ == "__main__":
    main()
