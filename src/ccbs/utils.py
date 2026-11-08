"""Small helpers: seeding, device selection, JSON I/O, environment info."""

from __future__ import annotations

import json
import platform
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    Note: this does not make cuDNN fully deterministic. Two runs with the same
    seed can still differ slightly on GPU. We accept that and instead report
    mean +/- std over several seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    """Return `name` if given, otherwise CUDA when available, else CPU."""
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def environment_info() -> dict:
    """Versions and hardware, stored next to every result for provenance."""
    import segmentation_models_pytorch as smp

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "smp": smp.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
