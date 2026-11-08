"""Data loading for SpaceNet-2 buildings (PS-RGB imagery + GeoJSON footprints).

Expected layout (what `scripts/download_data.sh` produces):

    <root>/AOI_3_Paris/PS-RGB/SN2_buildings_train_AOI_3_Paris_PS-RGB_img10.tif
    <root>/AOI_3_Paris/geojson_buildings/SN2_buildings_train_AOI_3_Paris_geojson_buildings_img10.geojson
    <root>/AOI_5_Khartoum/...

Each image is a 650x650, 3-band uint16 GeoTIFF. Each GeoJSON holds the building
footprint polygons (lon/lat). We rasterize the polygons onto the image grid to
get a binary mask (1 = building, 0 = background).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from rasterio import open as rio_open
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from torch.utils.data import Dataset, Subset, random_split

CITIES = {"paris": "AOI_3_Paris", "khartoum": "AOI_5_Khartoum"}

# Raw digital numbers go up to ~1800 in these two cities; dividing by 2000 and
# clipping maps them to [0, 1] with a *fixed* constant shared by every image.
MAX_VALUE = 2000.0

NORMALIZATIONS = ("global", "zscore")


def other_city(city: str) -> str:
    """The other city of the pair ('paris' <-> 'khartoum')."""
    return "khartoum" if city == "paris" else "paris"


# --------------------------------------------------------------------------- #
# Low-level loading
# --------------------------------------------------------------------------- #
def rasterize_buildings(geojson_path, shape, transform, crs) -> np.ndarray:
    """Burn the building polygons of one GeoJSON file into a uint8 mask.

    An empty GeoJSON (no buildings) is valid and gives an all-zero mask; we must
    skip `rasterize` in that case because it raises on an empty shapes list.
    """
    with open(geojson_path) as f:
        features = json.load(f)["features"]

    shapes = []
    for feat in features:
        geom = feat.get("geometry")
        if geom is None:
            continue
        # GeoJSON is lon/lat (EPSG:4326); reproject to the image CRS.
        shapes.append((transform_geom("EPSG:4326", crs, geom), 1))

    if not shapes:
        return np.zeros(shape, dtype="uint8")
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="uint8")


def normalize(img: np.ndarray, mode: str = "global") -> np.ndarray:
    """Turn a raw (C, H, W) float array into network input.

    global : clip(x / 2000, 0, 1). The same constant for every image, so
             absolute brightness differences between cities are preserved.
    zscore : after the global scaling, standardize each channel of each image to
             zero mean / unit variance, ignoring all-zero (no-data) pixels. This
             removes per-image brightness and contrast.
    """
    img = np.clip(img / MAX_VALUE, 0, 1).astype("float32")
    if mode == "global":
        return img
    if mode != "zscore":
        raise ValueError(f"unknown normalization {mode!r}; choose from {NORMALIZATIONS}")

    valid = img.sum(axis=0) > 0  # no-data pixels are exactly zero in all bands
    out = np.zeros_like(img)
    if valid.sum() > 100:
        for c in range(img.shape[0]):
            v = img[c][valid]
            out[c][valid] = (v - v.mean()) / (v.std(ddof=1) + 1e-6)
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class BuildingsDataset(Dataset):
    """All images of one city. Returns (image[3,H,W] float32, mask[1,H,W] float32).

    Parameters
    ----------
    root : directory containing the AOI_* folders.
    city : 'paris' or 'khartoum'.
    norm : 'global' or 'zscore' (see `normalize`).
    mask_cache_dir : optional directory where rasterized masks are cached as
        .npy files, so the (slow) rasterization happens only once per image.
    max_images : optional cap, used for quick smoke tests.
    """

    def __init__(self, root, city: str, norm: str = "global",
                 mask_cache_dir=None, max_images: int | None = None):
        if city not in CITIES:
            raise ValueError(f"city must be one of {list(CITIES)}, got {city!r}")
        if norm not in NORMALIZATIONS:
            raise ValueError(f"norm must be one of {NORMALIZATIONS}, got {norm!r}")
        self.city = city
        self.aoi = CITIES[city]
        self.norm = norm
        self.img_dir = Path(root) / self.aoi / "PS-RGB"
        self.gj_dir = Path(root) / self.aoi / "geojson_buildings"
        self.prefix = f"SN2_buildings_train_{self.aoi}"
        self.mask_cache_dir = Path(mask_cache_dir) / self.aoi if mask_cache_dir else None
        if self.mask_cache_dir:
            self.mask_cache_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(self.img_dir.glob("*.tif"))
        if not files:
            raise FileNotFoundError(f"no .tif files in {self.img_dir}")
        self.ids = [p.stem.split("_")[-1] for p in files]  # e.g. 'img10'
        if max_images is not None:
            self.ids = self.ids[:max_images]

    def __len__(self) -> int:
        return len(self.ids)

    def _paths(self, img_id: str):
        return (self.img_dir / f"{self.prefix}_PS-RGB_{img_id}.tif",
                self.gj_dir / f"{self.prefix}_geojson_buildings_{img_id}.geojson")

    def __getitem__(self, i: int):
        img_id = self.ids[i]
        img_path, gj_path = self._paths(img_id)

        with rio_open(img_path) as src:
            raw = src.read().astype("float32")  # (3, H, W)
            transform, crs = src.transform, src.crs
        shape = raw.shape[1:]

        cache_file = self.mask_cache_dir / f"{img_id}.npy" if self.mask_cache_dir else None
        if cache_file is not None and cache_file.exists():
            mask = np.load(cache_file)
        else:
            mask = rasterize_buildings(gj_path, shape, transform, crs)
            if cache_file is not None:
                np.save(cache_file, mask)

        img = normalize(raw, self.norm)
        return torch.from_numpy(img), torch.from_numpy(mask[None].astype("float32"))


def split_dataset(ds: Dataset, val_frac: float = 0.2, seed: int = 42):
    """Random train/val split with a *fixed* seed, independent of the training seed.

    Using the same split for every training seed means seed-to-seed variation
    reflects training randomness only, not a different validation set.

    Caveat: neighbouring tiles of one city overlap spatially, so a random split
    leaks a little spatial information from train to val. In-city val IoU is
    therefore optimistic. Cross-city numbers are not affected.
    """
    n_val = int(val_frac * len(ds))
    gen = torch.Generator().manual_seed(seed)
    train, val = random_split(ds, [len(ds) - n_val, n_val], generator=gen)
    return train, val


def few_shot_subset(train_subset: Subset, n_images: int, seed: int) -> Subset:
    """Pick `n_images` random images from a training subset (seed-dependent)."""
    import random

    idx = list(range(len(train_subset)))
    random.Random(seed).shuffle(idx)
    return Subset(train_subset, idx[:n_images])
