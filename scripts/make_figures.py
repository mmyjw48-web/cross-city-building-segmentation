#!/usr/bin/env python
"""Aggregate results/runs/*.json into figures and a markdown summary.

    python scripts/make_figures.py                 # reads results/runs, writes figures/ and results/summary.md

Training runs (written by scripts/train.py) and few-shot runs (scripts/few_shot.py)
are told apart by their content. Everything is aggregated as mean +/- std over seeds;
the number of seeds is always printed next to the number.
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE = "#2B6CB0", "#C05621"  # colour-blind-safe pair
CITY = {"paris": "Paris", "khartoum": "Khartoum"}


def load_runs(results_dir):
    train, fewshot = [], []
    for path in sorted(glob.glob(str(Path(results_dir) / "*.json"))):
        with open(path) as f:
            d = json.load(f)
        (fewshot if "rows" in d else train).append(d)
    return train, fewshot


def variant(cfg) -> str:
    """Name of a training configuration, without the seed."""
    v = cfg["norm"]
    if cfg.get("aug_brightness"):
        v += "+bright"
    if cfg.get("no_pretrained"):
        v += "+scratch"
    return v


def ms(values):
    """mean, std (0 if a single value), n."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def fmt(values):
    m, s, n = ms(values)
    return f"{m:.3f} ± {s:.3f} (n={n})" if n > 1 else f"{m:.3f} (n=1)"


# --------------------------------------------------------------------------- #
def transfer_matrix(train_runs, out_png, which="global"):
    cells = defaultdict(list)  # (train_city, eval_city) -> [iou per seed]
    for r in train_runs:
        c = r["config"]
        if variant(c) != which:
            continue
        s, t = c["source"], c["target"]
        cells[(s, s)].append(r["metrics"]["source_val"]["iou"])
        cells[(s, t)].append(r["metrics"]["target_all"]["iou"])
    if not cells:
        return None

    cities = ["paris", "khartoum"]
    grid = np.full((2, 2), np.nan)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for i, tr in enumerate(cities):
        for j, ev in enumerate(cities):
            if cells.get((tr, ev)):
                grid[i, j] = ms(cells[(tr, ev)])[0]
    vmax = max(0.7, np.nanmax(grid))
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=vmax)
    for i, tr in enumerate(cities):
        for j, ev in enumerate(cities):
            if cells.get((tr, ev)):
                m, s, n = ms(cells[(tr, ev)])
                txt = f"{m:.3f}\n± {s:.3f}\n(n={n})" if n > 1 else f"{m:.3f}\n(n=1)"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if m / vmax > 0.6 else "black", fontsize=11)
    ax.set_xticks([0, 1], [CITY[c] for c in cities])
    ax.set_yticks([0, 1], [CITY[c] for c in cities])
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("trained on")
    ax.set_title(f"IoU, train city × evaluation city  [{which}]\n"
                 "diagonal: held-out val split; off-diagonal: all images of the other city",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def epoch_curves(train_runs, out_png, which="global"):
    by_city = defaultdict(list)
    for r in train_runs:
        if variant(r["config"]) == which:
            by_city[r["config"]["source"]].append(r)
    if not by_city:
        return None

    fig, axes = plt.subplots(1, len(by_city), figsize=(5.2 * len(by_city), 3.8), squeeze=False, sharey=True)
    for ax, city in zip(axes[0], [c for c in CITY if c in by_city]):
        runs = by_city[city]
        n_ep = min(len(r["history"]) for r in runs)
        epochs = np.arange(1, n_ep + 1)
        for key, label, colour in [("val_iou", f"{CITY[city]} val (same city)", BLUE),
                                   ("other_city_iou", "other city (all images)", ORANGE)]:
            if not all(key in r["history"][0] for r in runs):
                continue
            arr = np.array([[h[key] for h in r["history"][:n_ep]] for r in runs])
            mean = arr.mean(0)
            ax.plot(epochs, mean, color=colour, marker="o", ms=3, label=label)
            if len(runs) > 1:
                sd = arr.std(0, ddof=1)
                ax.fill_between(epochs, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                                color=colour, alpha=0.15)
        ax.set_title(f"trained on {CITY[city]} (n={len(runs)} seeds)", fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_xticks(epochs)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("IoU")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def few_shot_plot(fewshot_runs, out_png):
    if not fewshot_runs:
        return None
    targets = [c for c in CITY if any(d["config"]["target_city"] == c for d in fewshot_runs)]
    fig, axes = plt.subplots(1, len(targets), figsize=(5.4 * len(targets), 4.0), squeeze=False, sharey=True)
    for ax, target in zip(axes[0], targets):
        for d in fewshot_runs:
            cfg = d["config"]
            if cfg["target_city"] != target:
                continue
            by_n = defaultdict(list)
            for row in d["rows"]:
                by_n[row["n_images"]].append(row["iou"])
            ns = sorted(by_n)
            mean = np.array([np.mean(by_n[n]) for n in ns])
            sd = np.array([np.std(by_n[n], ddof=1) if len(by_n[n]) > 1 else 0 for n in ns])
            colour = BLUE if cfg["init"] == "source" else ORANGE
            label = "source-trained init" if cfg["init"] == "source" else "ImageNet-only init"
            x = np.arange(len(ns))
            ax.plot(x, mean, marker="o", color=colour, label=f"{label} (lr={cfg['lr']:g})")
            ax.fill_between(x, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=colour, alpha=0.15)
            ax.set_xticks(x, [str(n) for n in ns])
        ax.set_title(f"adapting to {CITY[target]}", fontsize=10)
        ax.set_xlabel("# target-city training images (0 = zero-shot)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("IoU on the target city's val split")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def load_sweeps(sweep_dir):
    out = []
    for path in sorted(glob.glob(str(Path(sweep_dir) / "*.json"))):
        with open(path) as f:
            out.append(json.load(f))
    return out


def sweep_plot(sweeps, out_png):
    """IoU vs. probability threshold on the other city, one line per run."""
    if not sweeps:
        return None
    # target city = evaluation city; source city = prefix of the run name
    groups = defaultdict(list)
    for d in sweeps:
        groups[(d["run_name"].split("_")[0], d["city"])].append(d)
    keys = [k for k in [("paris", "khartoum"), ("khartoum", "paris")] if k in groups]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(5.4 * len(keys), 4.0), squeeze=False, sharey=True)
    for ax, (src, tgt) in zip(axes[0], keys):
        for d in sorted(groups[(src, tgt)], key=lambda d: d["run_name"]):
            ts = sorted(float(t) for t in d["threshold_sweep"])
            ious = [d["threshold_sweep"][str(t)] for t in ts]
            ax.plot(ts, ious, marker="o", ms=3, label=d["run_name"].split("_")[-1])
        ax.axvline(0.5, color="grey", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(f"trained on {CITY[src]}, evaluated on {CITY[tgt]}", fontsize=10)
        ax.set_xlabel("probability threshold (dashed: 0.5)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, title="seed")
    axes[0][0].set_ylabel("IoU")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def summary_md(train_runs, fewshot_runs, out_md, sweeps=()):
    lines = ["# Results summary (auto-generated by scripts/make_figures.py)", ""]
    groups = defaultdict(list)
    for r in train_runs:
        groups[(r["config"]["source"], variant(r["config"]))].append(r)
    if groups:
        lines += ["## Training runs", "",
                  "| trained on | variant | epochs | seeds | source val IoU | other-city IoU (all images) | other-city precision | other-city recall |",
                  "|---|---|---|---|---|---|---|---|"]
        for (city, var), runs in sorted(groups.items()):
            g = lambda split, key: [r["metrics"][split][key] for r in runs]  # noqa: E731
            lines.append(f"| {CITY[city]} | {var} | {runs[0]['config']['epochs']} | {len(runs)} | "
                         f"{fmt(g('source_val', 'iou'))} | {fmt(g('target_all', 'iou'))} | "
                         f"{fmt(g('target_all', 'precision'))} | {fmt(g('target_all', 'recall'))} |")
        lines.append("")

        # Per-run table: the averages above hide that the *predicted building fraction*
        # on the other city can swing a lot between seeds (a calibration effect at
        # threshold 0.5), while in-city it tracks the true fraction closely.
        lines += ["## Per-run details", "",
                  "`pred/true` = fraction of the other city's pixels predicted / labelled as building at threshold 0.5.", "",
                  "| run | best epoch | source val IoU | other-city IoU | other-city precision | other-city recall | other-city pred / true |",
                  "|---|---|---|---|---|---|---|"]
        for r in sorted(train_runs, key=lambda r: (r["config"]["source"], variant(r["config"]), r["config"]["seed"])):
            sv, ta = r["metrics"]["source_val"], r["metrics"]["target_all"]
            lines.append(f"| {r['config']['run_name']} | {r['best_epoch']} | {sv['iou']:.3f} | {ta['iou']:.3f} | "
                         f"{ta['precision']:.3f} | {ta['recall']:.3f} | {ta['pred_frac']:.3f} / {ta['true_frac']:.3f} |")
        lines.append("")
    if sweeps:
        lines += ["## Threshold sweep on the other city", "",
                  "IoU at the default threshold 0.5 vs. at the best threshold. The best threshold is an **oracle**"
                  " (chosen with the evaluation labels): it diagnoses calibration shift, it is not a deployable result.", "",
                  "| run | evaluated on | IoU @0.5 | best threshold | IoU @best |", "|---|---|---|---|---|"]
        for d in sorted(sweeps, key=lambda d: d["run_name"]):
            o = d["oracle_best_threshold"]
            lines.append(f"| {d['run_name']} | {CITY[d['city']]} ({d['split']}) | {d['threshold_sweep']['0.5']:.3f} | "
                         f"{o['threshold']:g} | {o['iou']:.3f} |")
        lines.append("")
    if fewshot_runs:
        lines += ["## Few-shot adaptation (IoU on the target city's val split)", ""]
        for d in fewshot_runs:
            cfg = d["config"]
            lines += [f"**{CITY[cfg['target_city']]}, init = {cfg['init']}**, "
                      f"{cfg['epochs']} epochs, lr={cfg['lr']:g}", "",
                      "| N images | IoU |", "|---|---|"]
            by_n = defaultdict(list)
            for row in d["rows"]:
                by_n[row["n_images"]].append(row["iou"])
            for n in sorted(by_n):
                lines.append(f"| {n} | {fmt(by_n[n])} |")
            lines.append("")
    Path(out_md).write_text("\n".join(lines) + "\n")
    return out_md


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results/runs")
    p.add_argument("--fig-dir", default="figures")
    p.add_argument("--summary", default="results/summary.md")
    p.add_argument("--sweep-dir", default="results/sweeps", help="JSONs written by scripts/evaluate.py --sweep --out")
    p.add_argument("--variant", default="global", help="which training variant to plot (e.g. global, zscore, global+bright)")
    a = p.parse_args()

    train, fewshot = load_runs(a.results_dir)
    sweeps = load_sweeps(a.sweep_dir)
    Path(a.fig_dir).mkdir(parents=True, exist_ok=True)
    print(f"{len(train)} training runs, {len(fewshot)} few-shot files")
    available = sorted({variant(r["config"]) for r in train})
    if train and a.variant not in available:
        print(f"note: no runs for variant {a.variant!r}; available: {available} (use --variant)")
    outputs = [transfer_matrix(train, Path(a.fig_dir) / "transfer_matrix.png", a.variant),
               epoch_curves(train, Path(a.fig_dir) / "epoch_curves.png", a.variant),
               few_shot_plot(fewshot, Path(a.fig_dir) / "few_shot.png"),
               sweep_plot(sweeps, Path(a.fig_dir) / "threshold_sweep.png"),
               summary_md(train, fewshot, a.summary, sweeps)]
    for out in outputs:
        if out:
            print("wrote", out)


if __name__ == "__main__":
    main()
