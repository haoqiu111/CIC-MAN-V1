#!/usr/bin/env python3
"""t-SNE of CIC-MAN v6ic health (z_h) and domain (z_d) representations.

For each cross-dataset target, loads the last-epoch v6ic checkpoint (paper
protocol), embeds a subsample of source-val + target-test windows, and plots
  row 1: router-fused z_h coloured by fault class (marker: source vs target)
  row 2: view-mean  z_d coloured by dataset
Output: outputs/figures/fig_tsne.{png,pdf}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def add_src_to_path() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


add_src_to_path()

from cicman.v2.data import ALL_VIEWS, CachedWindowDataset, ViewCache  # noqa: E402
from cicman.v2.model import CICMANv2  # noqa: E402

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})

TARGETS = ["hust", "ottawa", "paderborn"]
CLASSES = ["Normal", "Inner", "Outer"]
CLASS_COLORS = ["#4878d0", "#e49444", "#d1494e"]
DATASET_COLORS = {"hust": "#4878d0", "ottawa": "#e49444", "paderborn": "#d1494e"}


def subsample(ds, limit, rng):
    idx = np.arange(len(ds))
    if len(idx) > limit:
        idx = rng.choice(idx, size=limit, replace=False)
    return np.sort(idx)


@torch.no_grad()
def embed(model, ds, indices, device, batch_size=64):
    zh, zd, ys, dsets = [], [], [], []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        items = [ds[int(i)] for i in chunk]
        views = {v: torch.stack([it["views"][v] for it in items]).to(device) for v in ALL_VIEWS}
        feats = torch.stack([it["feats"] for it in items]).to(device)
        out = model(views, feats)
        w = out["router_weights"].unsqueeze(-1)  # (B, K, 1)
        zh.append((out["z_h"] * w).sum(dim=1).cpu().numpy())
        zd.append(out["z_d"].mean(dim=1).cpu().numpy())
        ys.extend(int(it["y"]) for it in items)
        dsets.extend(ds.rows[int(i)]["dataset_id"] for i in chunk)
    return np.concatenate(zh), np.concatenate(zd), np.array(ys), np.array(dsets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-side", type=int, default=1200, help="max windows from source val and from target test")
    parser.add_argument("--checkpoint", default="last.pt", help="last.pt (paper protocol) or best.pt")
    args = parser.parse_args()

    from sklearn.manifold import TSNE

    root = args.project_root
    windows_root = root / "data/paper1_cicman/cache/windows/cross_dataset_task3_source_mixed"
    cache_root = root / "data/paper1_cicman/cache/views_v2"
    ckpt_root = root / "outputs/checkpoints"
    out_dir = root / "outputs/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    cache = ViewCache(cache_root, views=ALL_VIEWS)

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.0))
    for col, target in enumerate(TARGETS):
        task_dir = windows_root / f"target_dataset_{target}"
        ckpt_path = ckpt_root / f"v2_cicman_v6ic_target_{target}_seed{args.seed}" / args.checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")

        model = CICMANv2(
            num_classes=3, views=ALL_VIEWS, num_domains=2,
            router_mode="causal", router_prior=[1.0 / len(ALL_VIEWS)] * len(ALL_VIEWS),
        )
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()

        src_ds = CachedWindowDataset(task_dir / "val_windows.csv", cache)
        tgt_ds = CachedWindowDataset(task_dir / "test_windows.csv", cache)
        src_idx = subsample(src_ds, args.per_side, rng)
        tgt_idx = subsample(tgt_ds, args.per_side, rng)
        zh_s, zd_s, y_s, d_s = embed(model, src_ds, src_idx, device)
        zh_t, zd_t, y_t, d_t = embed(model, tgt_ds, tgt_idx, device)

        zh = np.concatenate([zh_s, zh_t])
        zd = np.concatenate([zd_s, zd_t])
        y = np.concatenate([y_s, y_t])
        dset = np.concatenate([d_s, d_t])
        is_target = np.concatenate([np.zeros(len(y_s), bool), np.ones(len(y_t), bool)])

        e_h = TSNE(n_components=2, perplexity=30, init="pca", random_state=args.seed).fit_transform(zh)
        e_d = TSNE(n_components=2, perplexity=30, init="pca", random_state=args.seed).fit_transform(zd)

        ax = axes[0, col]
        for c in range(3):
            m = (~is_target) & (y == c)
            ax.scatter(e_h[m, 0], e_h[m, 1], s=4, c=CLASS_COLORS[c], marker="o", alpha=0.45, linewidths=0)
            m = is_target & (y == c)
            ax.scatter(e_h[m, 0], e_h[m, 1], s=7, c=CLASS_COLORS[c], marker="^", alpha=0.7, linewidths=0)
        ax.set_title(f"target {target.capitalize()}: $z_h$")
        ax.set_xticks([]), ax.set_yticks([])

        ax = axes[1, col]
        for name, color in DATASET_COLORS.items():
            m = dset == name
            if m.any():
                ax.scatter(e_d[m, 0], e_d[m, 1], s=4, c=color, alpha=0.45, linewidths=0, label=name.capitalize())
        ax.set_title(f"target {target.capitalize()}: $z_d$")
        ax.set_xticks([]), ax.set_yticks([])
        print(f"{target}: src={len(y_s)} tgt={len(y_t)} done", flush=True)

    class_handles = [plt.Line2D([], [], color=CLASS_COLORS[c], marker="s", ls="", label=CLASSES[c]) for c in range(3)]
    class_handles += [
        plt.Line2D([], [], color="gray", marker="o", ls="", label="source"),
        plt.Line2D([], [], color="gray", marker="^", ls="", label="target"),
    ]
    axes[0, 0].legend(handles=class_handles, fontsize=6, loc="best")
    axes[1, 0].legend(fontsize=6, loc="best")
    fig.suptitle("CIC-MAN v6ic representations (t-SNE, last-epoch checkpoints)", y=0.99)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_tsne.{ext}")
    print(f"saved -> {out_dir / 'fig_tsne.png'}")


if __name__ == "__main__":
    main()
