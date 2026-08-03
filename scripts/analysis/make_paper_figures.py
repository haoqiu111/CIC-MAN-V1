#!/usr/bin/env python3
"""Generate the paper figure set from saved result artifacts (PNG + PDF).

Figures (outputs/figures/):
  fig_probe_views          pivotal linear-probe comparison per view/target
  fig_main_matrix          5-seed cross-dataset main matrix (rec macro-F1)
  fig_ablation             v4/v6 component ablations
  fig_shortcut_reversal    correlated/reversed/neutral per model
  fig_perturbation_<t>     robustness profiles per target
  fig_router_priors        clean vs intervention-consistent reliability priors
  fig_confusion_<t>        v6ic recording-level confusion matrices
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# R1 revision: unified journal style shared with make_r1_figures.py
INK, MUTED = "#1a1a1a", "#555555"
PAL = {"blue": "#4878d0", "orange": "#e49444", "red": "#b0413e",
       "green": "#5f9e6e", "purple": "#8172b3", "gray": "#9aa0a6"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "figure.dpi": 200,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "text.color": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
})

TARGETS = ["hust", "ottawa", "paderborn"]
CLASSES = ["Normal", "Inner", "Outer"]


def save(fig, out_dir: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved {name}")


def fig_probe(tables: Path, out: Path):
    data = json.loads((tables / "pivotal_env_order_probe.json").read_text(encoding="utf-8"))
    views = ["raw_spec", "env_spec", "env_order"]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    x = np.arange(len(TARGETS))
    w = 0.25
    for i, v in enumerate(views):
        vals = [data[f"{t}/{v}"]["recording_macro_f1"] for t in TARGETS]
        ax.bar(x + (i - 1) * w, vals, w, label=v.replace("_", " "))
    ax.axhline(1 / 3, color="gray", ls="--", lw=0.8, label="random")
    ax.set_xticks(x, [t.capitalize() for t in TARGETS])
    ax.set_ylabel("Recording macro-F1")
    ax.set_title("Linear probe transfer of signal views (LODO)")
    ax.legend(fontsize=7)
    save(fig, out, "fig_probe_views")


def fig_main(tables: Path, out: Path):
    frames = [pd.read_csv(tables / f"v2_pilot_summary_seed{s}.csv") for s in [42, 2025, 2026, 7, 123]]
    df = pd.concat(frames)
    df["rec"] = pd.to_numeric(df["recording_macro_f1_last"], errors="coerce")
    models = ["single_raw", "dann", "ensemble", "moe", "dg_irm", "dg_coral", "dg_mmd", "dg_groupdro",
              "single_env_order", "cicman_v4", "cicman_v6ic"]
    labels = ["Raw CNN", "DANN", "Ensemble", "MoE", "IRM", "CORAL", "MMD", "GroupDRO",
              "Order agent", "CIC-MAN (stat)", "CIC-MAN (IC)"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    means, stds = [], []
    for m in models:
        per_seed = df[df.model == m].groupby("seed")["rec"].mean()
        means.append(per_seed.mean())
        stds.append(per_seed.std() if len(per_seed) > 1 else 0)
    colors = [PAL["gray"]] * 8 + [PAL["orange"], PAL["blue"], PAL["red"]]
    ax.bar(range(len(models)), means, yerr=stds, capsize=2.5, color=colors)
    ax.axhline(1 / 3, color="gray", ls="--", lw=0.8)
    ax.set_xticks(range(len(models)), labels, rotation=35, ha="right")
    ax.set_ylabel("Mean recording macro-F1")
    ax.set_title("Cross-dataset leave-one-dataset-out (target-free)")
    save(fig, out, "fig_main_matrix")


def fig_ablation(tables: Path, out: Path):
    text = (tables / "v2_ablation_matrix.md").read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if line.startswith("| A"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            try:
                mean, std = parts[4].split("±")
                rows.append((parts[0], float(mean), float(std)))
            except ValueError:
                continue
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    names = [r[0] for r in rows]
    ax.barh(names, [r[1] for r in rows], xerr=[r[2] for r in rows], capsize=2.5,
            color=[PAL["red"] if "Full" in n else PAL["blue"] for n in names])
    ax.set_xlabel("Mean recording macro-F1 (3 seeds)")
    ax.set_title("Component ablations")
    save(fig, out, "fig_ablation")


def fig_shortcut(tables: Path, out: Path):
    text = (tables / "v2_shortcut_reversal_seed42.md").read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) == 5 and parts[0] not in ("Model", "---") and not parts[0].startswith("-"):
            try:
                rows.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    x = np.arange(len(rows))
    w = 0.26
    for i, (label, key) in enumerate([("correlated", 1), ("reversed", 2), ("neutral", 3)]):
        ax.bar(x + (i - 1) * w, [r[key] for r in rows], w, label=label)
    ax.set_xticks(x, [r[0].replace("cicman_", "CIC-MAN ").replace("single_", "") for r in rows], rotation=25, ha="right")
    ax.set_ylabel("Recording macro-F1")
    ax.set_title("Shortcut reversal (class-conditioned tone)")
    ax.legend(fontsize=7)
    save(fig, out, "fig_shortcut_reversal")


def fig_perturbations(tables: Path, out: Path):
    pert_dir = tables / "v2_perturbations"
    perts = ["clean", "gauss_snr10", "gauss_snr0", "impulse", "harmonic", "scale_0.5", "scale_2.0", "speed_jitter_3pct"]
    models = ["cicman_v6ic", "cicman_v4", "single_env_order", "single_raw"]
    for target in TARGETS:
        fig, ax = plt.subplots(figsize=(5.8, 2.6))
        for m in models:
            p = pert_dir / f"{m}_{target}_seed42.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            ax.plot(range(len(perts)), [d[k]["recording_macro_f1"] for k in perts], marker="o", ms=3, label=m)
        ax.set_xticks(range(len(perts)), [p.replace("_", "\n") for p in perts], fontsize=6.5)
        ax.set_ylabel("Recording macro-F1")
        ax.set_title(f"Measurement-intervention robustness (target {target})")
        ax.legend(fontsize=6.5)
        save(fig, out, f"fig_perturbation_{target}")


def fig_priors(ckpt_root: Path, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.4), sharey=True)
    views = ["raw", "denoise", "env_spec", "env_order", "stft", "cwt"]
    for ax, target in zip(axes, TARGETS):
        clean = json.loads((ckpt_root / f"v2_reliability_target_dataset_{target}_seed42" / "view_reliability.json").read_text(encoding="utf-8"))
        ic = json.loads((ckpt_root / f"v2_reliability_ic_target_dataset_{target}_seed42" / "view_reliability.json").read_text(encoding="utf-8"))
        x = np.arange(len(views))
        ax.bar(x - 0.2, [clean["reliability"][v] for v in views], 0.4, label="statistical")
        ax.bar(x + 0.2, [ic["reliability"][v] for v in views], 0.4, label="intervention-consistent")
        ax.set_xticks(x, views, rotation=45, ha="right", fontsize=6.5)
        ax.set_title(target, fontsize=8)
    axes[0].set_ylabel("Cross-source reliability")
    axes[0].legend(fontsize=6)
    save(fig, out, "fig_router_priors")


def fig_confusions(ckpt_root: Path, out: Path):
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap

    audit = pd.read_csv(ckpt_root.parent / "tables" / "recording_aggregation_audit.csv")
    fig = plt.figure(figsize=(7.2, 2.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.055],
                          left=0.105, right=0.965, bottom=0.24, top=0.87, wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    cmap = LinearSegmentedColormap.from_list("paper_blue", ["#F7FBFF", "#6BAED6", "#08306B"])
    for index, (ax, target) in enumerate(zip(axes, TARGETS)):
        row = audit[
            audit["model"].eq("cicman_v6ic")
            & audit["target"].eq(target)
            & audit["seed"].eq(42)
            & audit["epoch_rule"].eq("last")
        ]
        if len(row) != 1:
            raise RuntimeError(f"missing official confusion matrix for target={target}")
        if row.iloc[0]["official_aggregation"] != "majority_probability_tiebreak":
            raise RuntimeError(f"unexpected recording aggregation for target={target}")
        cm = np.array(json.loads(row.iloc[0]["majority_confusion_matrix"]), dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7.2,
                        fontweight="bold" if i == j else "normal",
                        color="white" if cmn[i, j] >= 0.55 else "#222222")
        ax.set_xticks(range(3), CLASSES, fontsize=7)
        ax.set_yticks(range(3), CLASSES if index == 0 else [], fontsize=7)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="y", length=0 if index else 2.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"({chr(97 + index)}) {target.capitalize()}", fontsize=8, fontweight="bold", pad=7)
    axes[0].set_ylabel("True class", labelpad=8)
    fig.supxlabel("Predicted class", y=0.075, fontsize=8)
    cb = fig.colorbar(im, cax=cax, ticks=np.linspace(0, 1, 6))
    cb.set_label("Row-normalized proportion", rotation=270, labelpad=11)
    cb.outline.set_linewidth(0.5)
    fig.savefig(out / "fig_confusion_v6ic.pdf", dpi=600)
    fig.savefig(out / "fig_confusion_v6ic.svg", dpi=600)
    fig.savefig(out / "fig_confusion_v6ic.png", dpi=600)
    plt.close(fig)
    print("saved fig_confusion_v6ic")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    tables = args.project_root / "outputs/tables"
    ckpt_root = args.project_root / "outputs/checkpoints"
    out = args.project_root / "outputs/figures"
    out.mkdir(parents=True, exist_ok=True)

    fig_probe(tables, out)
    fig_main(tables, out)
    fig_ablation(tables, out)
    fig_shortcut(tables, out)
    fig_perturbations(tables, out)
    fig_priors(ckpt_root, out)
    fig_confusions(ckpt_root, out)
    print("all figures saved")


if __name__ == "__main__":
    main()
