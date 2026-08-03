#!/usr/bin/env python3
"""R1-revision figures for paper 1 (unified journal style).

  fig_views_demo    six soft-intervention views of one faulty window   (red #4)
  fig_probe_matrix  6 views x 3 targets cross-source transfer heatmap  (red #5)
  fig_hero          the at-a-glance success figure                     (txt #7)
  fig_tsne_compare  feature space before vs after CIC-MAN              (red #8)

Style: shared palette, no top/right spines, y-grid only, consistent fonts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P1 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P1 / "src"))

FIGS = P1 / "outputs" / "figures"
CKPT = P1 / "outputs" / "checkpoints"
TABLES = P1 / "outputs" / "tables"
CACHE = Path(__file__).resolve().parents[2] / "data/paper1_cicman/cache/views_v2"

BLUE, ORANGE, RED, GREEN, PURPLE, GRAY = "#4878d0", "#e49444", "#b0413e", "#5f9e6e", "#8172b3", "#9aa0a6"
INK, MUTED = "#1a1a1a", "#555555"
VIEWS = ["raw", "denoise", "env_spec", "env_order", "stft", "cwt"]
TARGETS = ["hust", "ottawa", "paderborn"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "figure.dpi": 200,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.titlesize": 9.5, "axes.labelsize": 8.5, "text.color": INK,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
})


def save(fig, name, *, svg=False, png_dpi=None):
    fig.savefig(FIGS / f"{name}.png", dpi=png_dpi)
    fig.savefig(FIGS / f"{name}.pdf")
    if svg:
        fig.savefig(FIGS / f"{name}.svg")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- views demo
def fig_views_demo():
    """Six views of one outer-race window: the soft interventions made visible."""
    import csv
    from cicman.v2.data import ViewCache

    cache = ViewCache(CACHE, views=VIEWS)
    # pick a HUST outer-race window with a strong order peak
    task = (
        Path(__file__).resolve().parents[2]
        / "data/paper1_cicman/cache/windows/cross_dataset_task3_source_mixed"
        / "target_dataset_paderborn/train_windows.csv"
    )
    rows = list(csv.DictReader(task.open(newline="", encoding="utf-8")))
    cand = [r for r in rows if r["dataset_id"] == "hust" and r["label"] == "outer"][:400]
    best, best_p = None, -1
    for r in cand:
        ci = cache.key_to_row.get((r["dataset_id"], r["recording_id"], int(r["window_index"])))
        if ci is None:
            continue
        spec = np.abs(cache.arrays["env_order"][ci])
        p = spec[20:].max() / (np.median(spec[20:]) + 1e-9)
        if p > best_p:
            best_p, best = p, ci
    i = best
    fs = 25600
    t = np.arange(4096) / fs * 1000  # ms
    orders = np.linspace(0.25, 32, 256)
    hz = np.linspace(2, 640, 256)

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.0))
    ax = axes[0, 0]
    ax.plot(t, cache.arrays["raw"][i], lw=0.5, color=BLUE)
    ax.set_title("(a) raw waveform — untouched measurement")
    ax.set_xlabel("time [ms]"); ax.set_ylabel("normalized amp.")

    ax = axes[0, 1]
    ax.plot(t, cache.arrays["denoise"][i], lw=0.5, color=GREEN)
    ax.set_title("(b) spectral denoise — noise-floor intervention")
    ax.set_xlabel("time [ms]")

    ax = axes[0, 2]
    ax.plot(hz, cache.arrays["env_spec"][i], lw=0.8, color=ORANGE)
    ax.set_title("(c) envelope spectrum — carrier removed")
    ax.set_xlabel("frequency [Hz]")

    ax = axes[1, 0]
    ax.plot(orders, cache.arrays["env_order"][i], lw=0.9, color=RED)
    for band, name in (((2.7, 4.5), "BPFO"), ((4.5, 6.8), "BPFI")):
        ax.axvspan(*band, color=GRAY, alpha=0.15)
        ax.text(np.mean(band), ax.get_ylim()[1] * 0.97, name, ha="center",
                va="top", fontsize=6.5, color=MUTED)
    ax.set_title("(d) shaft-order envelope — speed removed")
    ax.set_xlabel("shaft order"); ax.set_ylabel("log-unit magnitude")

    for ax, view, title in ((axes[1, 1], "stft", "(e) STFT — time-frequency image"),
                            (axes[1, 2], "cwt", "(f) CWT — multi-scale image")):
        ax.imshow(cache.arrays[view][i], aspect="auto", origin="lower", cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("time frame"); ax.set_ylabel("freq. bin")
        ax.grid(False)

    fig.suptitle("Soft interventions on the measurement mechanism: one outer-race window, six views",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig_views_demo")


# ------------------------------------------------------------- probe matrix
def fig_probe_matrix():
    """Cross-source LOSO transfer of ALL six views on the three tasks,
    identity vs hp800-counterfactual, from the cached reliability probes."""
    ident = np.zeros((len(VIEWS), len(TARGETS)))
    icmin = np.zeros_like(ident)
    for j, tgt in enumerate(TARGETS):
        payload = json.loads((CKPT / f"v2_reliability_ic_target_dataset_{tgt}_seed42"
                              / "view_reliability.json").read_text(encoding="utf-8"))
        for i, v in enumerate(VIEWS):
            ident[i, j] = payload["per_variant"]["views_v2"][v]
            icmin[i, j] = payload["reliability"][v]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, mat, title in ((axes[0], ident, "(a) statistical LOSO transfer (identity)"),
                           (axes[1], icmin, "(b) intervention-consistent (min over {id, HP800})")):
        im = ax.imshow(mat, cmap="Blues", vmin=0.2, vmax=0.75, aspect="auto")
        ax.set_xticks(range(len(TARGETS)), [t.capitalize() for t in TARGETS], fontsize=8)
        ax.set_yticks(range(len(VIEWS)), VIEWS, fontsize=8)
        for i in range(len(VIEWS)):
            for j in range(len(TARGETS)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if mat[i, j] > 0.5 else INK)
        ax.set_title(title, fontsize=9)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.85, label="cross-source recording macro-F1")
    fig.suptitle(
        "Which views carry transferable fault evidence? "
        "(chance-level ≈ 0.33; balanced classes)",
        fontsize=10,
    )
    save(fig, "fig_probe_matrix")


# --------------------------------------------------------------------- hero
def fig_hero():
    """The at-a-glance figure: accuracy x shortcut-sensitivity quadrants (a)
    and the full method field against the published cross-machine band (b)."""
    import pandas as pd
    audit_path = TABLES / "recording_aggregation_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError(f"Official recording audit required for hero figure: {audit_path}")
    df = pd.read_csv(audit_path)
    df = df[df["epoch_rule"] == "last"].copy()
    df["last"] = pd.to_numeric(df["majority_recording_macro_f1"], errors="coerce")
    mean5 = df.groupby(["model", "seed"])["last"].mean().groupby("model").mean()
    std5 = df.groupby(["model", "seed"])["last"].mean().groupby("model").std()
    worst5 = df.groupby(["model", "seed"])["last"].min().groupby("model").mean()

    order = ["single_raw", "dann", "dg_irm", "dg_coral", "dg_mmd", "dg_groupdro",
             "moe", "ensemble", "cicman_v4", "single_env_order", "cicman_v6ic"]
    gap = {}
    for model_name in order:
        payload = json.loads((CKPT / f"v2sc_{model_name}_seed42" / "shortcut_metrics.json").read_text(encoding="utf-8"))
        if payload.get("recording_aggregation") != "majority_probability_tiebreak":
            raise RuntimeError(f"shortcut metrics for {model_name} are not under the official recording protocol")
        if payload.get("checkpoint_epoch") != 40:
            raise RuntimeError(f"shortcut metrics for {model_name} are not from the frozen last-epoch checkpoint")
        gap[model_name] = float(payload["reversal_gap_rec"])
    label = {"single_raw": "Raw", "dann": "DANN", "dg_irm": "IRM",
             "dg_coral": "CORAL", "dg_mmd": "MMD", "dg_groupdro": "GroupDRO",
             "moe": "MoE", "ensemble": "Ensemble", "cicman_v4": "CIC-MAN (stat)",
             "single_env_order": "Order agent", "cicman_v6ic": "CIC-MAN (IC)"}
    color = {m: GRAY for m in order}
    color.update({"cicman_v4": ORANGE, "single_env_order": BLUE, "cicman_v6ic": RED})
    # Point offsets are fixed, deterministic, and deliberately separated for the
    # dense 0.49--0.54 shortcut-gap cluster.
    offsets = {
        "single_raw": (-10, -14), "dann": (-12, 12), "dg_irm": (-8, 17),
        "dg_coral": (-24, -17), "dg_mmd": (2, -17), "dg_groupdro": (21, -4),
        "moe": (-4, 13), "ensemble": (22, 13), "cicman_v4": (0, -18),
        "single_env_order": (0, 13), "cicman_v6ic": (13, -16),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0),
                             gridspec_kw={"width_ratios": [1.25, 1.10]})

    ax = axes[0]
    ax.axvspan(0.5, 0.62, color=GREEN, alpha=0.06)
    ax.axhspan(-0.1, 0.45, color=GREEN, alpha=0.06)
    for m in order:
        g = gap[m]
        x = float(mean5[m])
        ax.scatter(x, g, s=90 + 600 * float(worst5[m]), color=color[m],
                   edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(label[m], (x, g), xytext=offsets[m], textcoords="offset points",
                    ha="center", va="center", fontsize=7.1,
                    fontweight="bold" if m in {"cicman_v4", "single_env_order", "cicman_v6ic"} else "normal",
                    arrowprops={"arrowstyle": "-", "color": color[m], "lw": 0.55,
                                "shrinkA": 1.5, "shrinkB": 4.5}, zorder=4)
    ax.annotate("high accuracy &\nlow shortcut gap", (0.585, 0.075), fontsize=7.5, color=GREEN,
                ha="center", style="italic")
    ax.set_xlabel("cross-dataset mean recording macro-F1 (5 seeds)")
    ax.set_ylabel("shortcut reversal gap  (lower = less dependent)")
    ax.set_xlim(0.19, 0.61)
    ax.set_ylim(-0.08, 0.72)
    ax.set_title("(a) all methods: accuracy vs. shortcut sensitivity\n(bubble area = strict worst-target F1)")

    ax = axes[1]
    labels2 = ["Raw", "DANN", "IRM", "CORAL", "MMD", "GroupDRO", "MoE", "Ens.",
               "CIC-MAN\n(stat)", "Order\nagent", "CIC-MAN\n(IC)"]
    xs = np.arange(len(order))
    vals = [float(mean5[m]) for m in order]
    errs = [float(std5[m]) for m in order]
    cols = [GRAY] * 8 + [ORANGE, BLUE, RED]
    ax.axhspan(0.34, 0.52, color=PURPLE, alpha=0.10)
    ax.text(0.15, 0.525, "published cross-machine DG band (34-52%)",
            fontsize=7.2, color=PURPLE, va="bottom")
    ax.axhline(1 / 3, color=MUTED, ls="--", lw=0.8)
    ax.text(len(order) - 0.4, 1 / 3 - 0.017, "chance", fontsize=7, color=MUTED, ha="right")
    ax.bar(xs, vals, yerr=errs, capsize=2.2, color=cols, width=0.62)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels2, fontsize=6.5, rotation=32, ha="right")
    ax.set_ylabel("mean recording macro-F1")
    ax.set_ylim(0, 0.62)
    ax.set_title("(b) strict target-free cross-dataset field")
    save(fig, "fig_hero", svg=True, png_dpi=600)


# ------------------------------------------------------------- tsne compare
def fig_tsne_compare(target="hust", seed=42):
    """Cross-model diagnostic with original-space metrics.

    The two independently trained models do not share a representation basis,
    so their t-SNE maps are intentionally embedded independently and are not
    treated as before/after coordinates. Quantitative claims come only from
    each original feature space.
    """
    import json
    import torch
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from cicman.v2.data import CachedWindowDataset, ViewCache
    from cicman.v2.model import CICMANv2

    task = Path(__file__).resolve().parents[2] / "data/paper1_cicman/cache/windows/cross_dataset_task3_source_mixed" \
        / f"target_dataset_{target}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract(views, run, router_prior, split):
        cache = ViewCache(CACHE, views=views)
        ds = CachedWindowDataset(task / f"{split}_windows.csv", cache)
        idx = np.linspace(0, len(ds) - 1, min(1500, len(ds))).astype(int)
        ckpt = torch.load(CKPT / run / "last.pt", map_location="cpu", weights_only=True)
        m = CICMANv2(num_classes=3, views=views, num_domains=2,
                     router_mode="causal" if router_prior else "uniform",
                     router_prior=router_prior)
        m.load_state_dict(ckpt["model"])
        m.to(device).eval()
        zs, ys = [], []
        with torch.no_grad():
            for s in range(0, len(idx), 128):
                items = [ds[int(i)] for i in idx[s:s + 128]]
                v = {vw: torch.stack([it["views"][vw] for it in items]).to(device) for vw in views}
                f = torch.stack([it["feats"] for it in items]).to(device)
                out = m(v, f)
                w = out["router_weights"].unsqueeze(-1)
                zs.append((out["z_h"] * w).sum(1).cpu().numpy())
                ys.extend(int(it["y"]) for it in items)
        return np.concatenate(zs), np.array(ys)

    def original_space_metrics(source_z, source_y, target_z, target_y):
        classes = np.unique(target_y)
        target_centers = np.stack([target_z[target_y == c].mean(0) for c in classes])
        between = np.mean([
            np.linalg.norm(target_centers[i] - target_centers[j])
            for i in range(len(classes)) for j in range(i + 1, len(classes))
        ])
        within = np.mean(np.concatenate([
            np.linalg.norm(target_z[target_y == c] - target_centers[i], axis=1)
            for i, c in enumerate(classes)
        ]))
        class_domain_gaps = []
        for c in classes:
            source_c = source_z[source_y == c]
            target_c = target_z[target_y == c]
            source_center = source_c.mean(0)
            target_center = target_c.mean(0)
            pooled_within = np.mean(np.concatenate([
                np.linalg.norm(source_c - source_center, axis=1),
                np.linalg.norm(target_c - target_center, axis=1),
            ]))
            class_domain_gaps.append(float(np.linalg.norm(source_center - target_center) / pooled_within))
        return {
            "target_silhouette": float(silhouette_score(target_z, target_y)),
            "target_between_within": float(between / within),
            "mean_class_conditional_source_target_gap": float(np.mean(class_domain_gaps)),
            "class_conditional_source_target_gaps": class_domain_gaps,
        }

    raw_run = f"v2_single_raw_target_{target}_seed{seed}"
    cic_run = f"v2_cicman_v6ic_target_{target}_seed{seed}"
    z_raw, y = extract(["raw"], raw_run, None, "test")
    z_ic, y2 = extract(VIEWS, cic_run, [1 / 6.0] * 6, "test")
    zs_raw, ys_raw = extract(["raw"], raw_run, None, "train")
    zs_ic, ys_ic = extract(VIEWS, cic_run, [1 / 6.0] * 6, "train")
    metrics = {
        "protocol": "independent_tsne_original_space_metrics_v1",
        "target": target, "seed": seed, "n_target": int(len(y)),
        "n_source": int(len(ys_raw)),
        "raw": original_space_metrics(zs_raw, ys_raw, z_raw, y),
        "cicman": original_space_metrics(zs_ic, ys_ic, z_ic, y2),
    }
    (TABLES / "r1_tsne_compare_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    cls_colors = [BLUE, ORANGE, RED]
    names = ["Normal", "Inner", "Outer"]
    raw_m, cic_m = metrics["raw"], metrics["cicman"]
    for ax, z, yy, title in (
        (axes[0], z_raw, y,
         f"(a) Raw CNN  F1=.25 | sil={raw_m['target_silhouette']:.3f} | gap={raw_m['mean_class_conditional_source_target_gap']:.2f}"),
        (axes[1], z_ic, y2,
         f"(b) CIC-MAN $z_h$  F1=.51 | sil={cic_m['target_silhouette']:.3f} | gap={cic_m['mean_class_conditional_source_target_gap']:.2f}"),
    ):
        e = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(z)
        for c in range(3):
            m_ = yy == c
            ax.scatter(e[m_, 0], e[m_, 1], s=5, color=cls_colors[c], alpha=0.55,
                       linewidths=0, label=names[c])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7, markerscale=2.2, loc="best")
    fig.suptitle(f"Cross-model target geometry diagnostic ({target}; independent embeddings)",
                 fontsize=10)
    save(fig, "fig_tsne_compare")


if __name__ == "__main__":
    fig_views_demo()
    fig_probe_matrix()
    fig_hero()
    fig_tsne_compare()
