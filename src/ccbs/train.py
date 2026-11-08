"""Training and fine-tuning loops."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import few_shot_subset
from .metrics import evaluate
from .model import build_unet
from .utils import set_seed


def augment_batch(x: torch.Tensor, y: torch.Tensor, brightness: bool = False):
    """Batch-level augmentation (the same flip is applied to the whole batch).

    - horizontal / vertical flip, each with probability 0.5 (image and mask together)
    - optional brightness: per-image gain in U(0.5, 1.6) and per-channel gain in
      U(0.9, 1.1), then clamp to [0, 1]. Only meaningful for `--norm global`.
    """
    if torch.rand(1).item() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if torch.rand(1).item() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    if brightness:
        g = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.5, 1.6)
        c = torch.empty(x.size(0), 3, 1, 1, device=x.device).uniform_(0.9, 1.1)
        x = torch.clamp(x * g * c, 0, 1)
    return x, y


def _train_one_epoch(model, loader, optimizer, loss_fn, device, brightness):
    model.train()
    running, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x, y = augment_batch(x, y, brightness)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(x)
        n += len(x)
    return running / max(n, 1)


def fit(model, train_loader, val_loader, *, epochs: int, lr: float, device, loss_fn,
        brightness: bool = False, monitor_loader=None, ckpt_dir=None, log=print):
    """Train with Adam + cosine LR decay; keep the epoch with the best val IoU.

    Parameters
    ----------
    val_loader : same-city validation set. Used for model selection.
    monitor_loader : optional other-city set evaluated every epoch for *reporting
        only*; it never influences which checkpoint is kept.
    ckpt_dir : if given, `best.pt` (weights) and `last.pt` (full training state)
        are written there. If `last.pt` exists the run resumes from it, which
        makes training robust to Colab disconnects.

    Returns
    -------
    model (best weights loaded), history (list of per-epoch dicts), best_epoch (1-based)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history, best_iou, best_epoch, start = [], -1.0, 0, 0
    best_state = None  # in-memory copy of the best weights when no ckpt_dir is given

    best_path = last_path = None
    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path, last_path = ckpt_dir / "best.pt", ckpt_dir / "last.pt"
        if last_path.exists():
            ck = torch.load(last_path, map_location=device)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["opt"])
            scheduler.load_state_dict(ck["sched"])
            history, best_iou = ck["history"], ck["best_iou"]
            best_epoch, start = ck["best_epoch"], ck["epoch"] + 1
            log(f"resuming from epoch {start + 1}/{epochs}")

    for epoch in range(start, epochs):
        t0 = time.time()
        train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, device, brightness)
        scheduler.step()
        val = evaluate(model, val_loader, device, loss_fn)
        row = {"epoch": epoch + 1, "train_loss": train_loss,
               "val_loss": val["loss"], "val_iou": val["iou"]}
        if monitor_loader is not None:
            row["other_city_iou"] = evaluate(model, monitor_loader, device)["iou"]
        row["seconds"] = time.time() - t0
        history.append(row)

        msg = (f"epoch {epoch + 1}/{epochs} | train loss {train_loss:.3f} | "
               f"val loss {val['loss']:.3f} | val IoU {val['iou']:.3f}")
        if "other_city_iou" in row:
            msg += f" | other-city IoU {row['other_city_iou']:.3f}"
        log(msg + f" | {row['seconds']:.0f}s")

        if val["iou"] > best_iou:
            best_iou, best_epoch = val["iou"], epoch + 1
            if best_path is not None:
                torch.save(model.state_dict(), best_path)
            else:
                best_state = copy.deepcopy(model.state_dict())
        if last_path is not None:
            torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                        "sched": scheduler.state_dict(), "epoch": epoch, "history": history,
                        "best_iou": best_iou, "best_epoch": best_epoch}, last_path)

    if best_path is not None:
        model.load_state_dict(torch.load(best_path, map_location=device))
    elif best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def fine_tune_and_eval(init_state, train_subset, val_loader, *, n_images: int, seed: int,
                       epochs: int, lr: float, batch_size: int, device, loss_fn,
                       num_workers: int = 2, pretrained_encoder: bool = True) -> dict:
    """Few-shot adaptation: fine-tune on `n_images` target-city images, then evaluate.

    init_state : state_dict of a source-trained model, or None to start from the
        ImageNet-pretrained encoder (the control: no source-city training).
    The images are drawn from the target city's *train* split (seed-dependent);
    evaluation is always on the target city's fixed *val* split.
    """
    set_seed(seed)
    # ImageNet weights are only needed when there is no source checkpoint to load
    model = build_unet(pretrained=pretrained_encoder and init_state is None)
    if init_state is not None:
        model.load_state_dict(init_state)
    model.to(device)

    subset = few_shot_subset(train_subset, n_images, seed)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        _train_one_epoch(model, loader, optimizer, loss_fn, device, brightness=False)
    return evaluate(model, val_loader, device)
