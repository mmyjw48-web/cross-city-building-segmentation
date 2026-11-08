"""Synthetic SpaceNet-like data for smoke tests (no download needed).

Writes the same folder layout / file names as the real dataset: 3-band uint16
GeoTIFFs in EPSG:4326 plus GeoJSON footprints in lon/lat. Buildings are bright
rectangles on a dark noisy background; the two "cities" use different base
brightness so there is a (toy) domain shift.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

AOI = {"paris": "AOI_3_Paris", "khartoum": "AOI_5_Khartoum"}
BASE_BRIGHTNESS = {"paris": 300.0, "khartoum": 700.0}


def make_city(root, city: str, n_images: int = 16, size: int = 128, seed: int = 0,
              empty_every: int = 7):
    """Create `n_images` tiles. Every `empty_every`-th tile has no buildings."""
    rng = np.random.default_rng(seed + (0 if city == "paris" else 1000))
    aoi = AOI[city]
    img_dir = Path(root) / aoi / "PS-RGB"
    gj_dir = Path(root) / aoi / "geojson_buildings"
    img_dir.mkdir(parents=True, exist_ok=True)
    gj_dir.mkdir(parents=True, exist_ok=True)
    px = 1e-5  # degrees per pixel
    for k in range(n_images):
        lon0, lat0 = 2.0 + 0.01 * k, 48.0
        transform = from_origin(lon0, lat0, px, px)
        img = rng.normal(BASE_BRIGHTNESS[city], 40, size=(3, size, size))
        feats = []
        if k % empty_every != empty_every - 1:
            for _ in range(rng.integers(3, 8)):
                w, h = rng.integers(10, 25, size=2)
                x0, y0 = rng.integers(0, size - w), rng.integers(0, size - h)
                img[:, y0:y0 + h, x0:x0 + w] += 900.0
                ring = [[lon0 + x * px, lat0 - y * px]
                        for x, y in [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h), (x0, y0)]]
                feats.append({"type": "Feature", "properties": {},
                              "geometry": {"type": "Polygon", "coordinates": [ring]}})
        img = np.clip(img, 0, 1800).astype("uint16")
        stem = f"SN2_buildings_train_{aoi}"
        with rasterio.open(img_dir / f"{stem}_PS-RGB_img{k}.tif", "w", driver="GTiff",
                           height=size, width=size, count=3, dtype="uint16",
                           crs="EPSG:4326", transform=transform) as dst:
            dst.write(img)
        with open(gj_dir / f"{stem}_geojson_buildings_img{k}.geojson", "w") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f)
