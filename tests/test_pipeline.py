"""Fast tests. Run with:  python -m pytest -q tests   (or: python tests/test_pipeline.py)"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ccbs.data import BuildingsDataset, normalize, split_dataset  # noqa: E402
from ccbs.losses import BCEDiceLoss  # noqa: E402
from ccbs.metrics import evaluate  # noqa: E402
from ccbs.model import build_unet  # noqa: E402
from ccbs.train import fit  # noqa: E402
from synthetic import make_city  # noqa: E402


def _data(tmp, n=14):
    make_city(tmp, "paris", n_images=n)
    make_city(tmp, "khartoum", n_images=n)


def test_dataset_shapes_and_masks():
    with tempfile.TemporaryDirectory() as tmp:
        _data(tmp)
        ds = BuildingsDataset(tmp, "paris")
        x, y = ds[0]
        assert x.shape == (3, 128, 128) and y.shape == (1, 128, 128)
        assert x.dtype == torch.float32 and 0.0 <= x.min() and x.max() <= 1.0
        assert set(y.unique().tolist()) <= {0.0, 1.0}
        assert y.sum() > 0
        # tile index 6 (k % 7 == 6) was generated without buildings -> empty GeoJSON -> zero mask
        empty = [i for i, name in enumerate(ds.ids) if name == "img6"][0]
        assert ds[empty][1].sum() == 0


def test_mask_matches_bright_pixels():
    """The rasterized footprint should coincide with the bright rectangles we painted."""
    with tempfile.TemporaryDirectory() as tmp:
        _data(tmp)
        ds = BuildingsDataset(tmp, "paris")
        x, y = ds[0]
        bright = x.mean(0) > 0.4  # background ~0.15, buildings ~0.6
        inter = (bright & (y[0] > 0)).sum().item()
        union = (bright | (y[0] > 0)).sum().item()
        assert inter / union > 0.99  # the masks should align exactly with the painted buildings


def test_zscore_normalization():
    raw = np.random.default_rng(0).uniform(100, 1500, size=(3, 64, 64)).astype("float32")
    raw[:, :10, :] = 0  # no-data strip
    z = normalize(raw, "zscore")
    valid = raw.sum(0) > 0
    for c in range(3):
        assert abs(z[c][valid].mean()) < 1e-3
        assert abs(z[c][valid].std(ddof=1) - 1) < 1e-2
    assert (z[:, ~valid] == 0).all()


def test_split_is_reproducible_and_disjoint():
    with tempfile.TemporaryDirectory() as tmp:
        _data(tmp)
        ds = BuildingsDataset(tmp, "paris")
        a_tr, a_va = split_dataset(ds, 0.2, seed=42)
        b_tr, b_va = split_dataset(ds, 0.2, seed=42)
        assert list(a_va.indices) == list(b_va.indices)
        assert not set(a_tr.indices) & set(a_va.indices)
        assert len(a_tr) + len(a_va) == len(ds)


def test_metrics_perfect_and_empty():
    class Perfect(torch.nn.Module):
        def forward(self, x):  # logit +10 where the input is bright, -10 elsewhere
            return (x.mean(1, keepdim=True) > 0.4).float() * 20 - 10

    with tempfile.TemporaryDirectory() as tmp:
        _data(tmp)
        loader = DataLoader(BuildingsDataset(tmp, "paris"), batch_size=4)
        m = evaluate(Perfect(), loader, "cpu")
        assert "n_images" not in m  # would collide with the few-shot rows' training-set size
        assert m["iou"] > 0.9 and m["precision"] > 0.9 and m["recall"] > 0.9

        class Zero(torch.nn.Module):
            def forward(self, x):
                return torch.full_like(x[:, :1], -10.0)

        m0 = evaluate(Zero(), loader, "cpu")
        assert m0["iou"] == 0.0 and m0["pred_frac"] == 0.0 and m0["true_frac"] > 0


def test_end_to_end_smoke_and_resume():
    """Train 2 epochs on toy data (random init, CPU), then resume to 3 epochs."""
    with tempfile.TemporaryDirectory() as tmp:
        data, ck = Path(tmp) / "data", Path(tmp) / "ck"
        _data(data, n=12)
        ds = BuildingsDataset(data, "paris")
        tr, va = split_dataset(ds, 0.25)
        other = BuildingsDataset(data, "khartoum")
        mk = lambda d, s=False: DataLoader(d, batch_size=4, shuffle=s)  # noqa: E731
        torch.manual_seed(0)
        model = build_unet(pretrained=False)
        loss = BCEDiceLoss()
        model, hist, best = fit(model, mk(tr, True), mk(va), epochs=2, lr=1e-3, device="cpu",
                                loss_fn=loss, monitor_loader=mk(other), ckpt_dir=ck, log=lambda *_: None)
        assert len(hist) == 2 and 1 <= best <= 2
        assert all(0 <= h["val_iou"] <= 1 and "other_city_iou" in h for h in hist)
        assert (ck / "best.pt").exists() and (ck / "last.pt").exists()
        # re-running with a larger epoch budget resumes after epoch 2
        model2 = build_unet(pretrained=False)
        _, hist2, _ = fit(model2, mk(tr, True), mk(va), epochs=3, lr=1e-3, device="cpu",
                          loss_fn=loss, ckpt_dir=ck, log=lambda *_: None)
        assert [h["epoch"] for h in hist2] == [1, 2, 3]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
