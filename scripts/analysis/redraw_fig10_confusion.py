#!/usr/bin/env python3
"""Redraw Paper1 Fig. 10 from the frozen recording-level audit artifact."""

from __future__ import annotations

# ================================================================
# ASSET CONFIRMATION
# ================================================================
# Asset inspected: assets/figures/ConfusionMatrix/plot_SectorConfusionMatrix.py
# Companion PNG inspected: plot_SectorConfusionMatrix.png
# Semantic decision: cross-type inheritance only. The asset is a binary sector
# matrix, whereas Fig. 10 requires three row-normalized 3-class heatmaps.
# Parameters inherited: direct cell labels, shared percentage scale, restrained
# borders, sans-serif typography.
# ================================================================

import argparse
import json
from pathlib import Path

import matplotlib as mpl

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette — COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


TARGETS = ["hust", "ottawa", "paderborn"]
PANEL_TITLES = ["(a) HUST", "(b) Ottawa", "(c) Paderborn"]
CLASSES = ["Normal", "Inner", "Outer"]
OFFICIAL_AGGREGATION = "majority_probability_tiebreak"


def load_official_matrices(audit_csv: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    audit = pd.read_csv(audit_csv)
    counts: list[np.ndarray] = []
    normalized: list[np.ndarray] = []
    for target in TARGETS:
        row = audit[
            audit["model"].eq("cicman_v6ic")
            & audit["target"].eq(target)
            & audit["seed"].eq(42)
            & audit["epoch_rule"].eq("last")
        ]
        if len(row) != 1:
            raise RuntimeError(f"expected one official row for {target}, found {len(row)}")
        protocol = str(row.iloc[0]["official_aggregation"])
        if protocol != OFFICIAL_AGGREGATION:
            raise RuntimeError(f"unexpected aggregation for {target}: {protocol}")
        cm = np.asarray(json.loads(row.iloc[0]["majority_confusion_matrix"]), dtype=np.int64)
        if cm.shape != (3, 3) or np.any(cm < 0):
            raise RuntimeError(f"invalid confusion matrix for {target}: shape={cm.shape}")
        row_totals = cm.sum(axis=1, keepdims=True)
        cmn = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals > 0)
        if not np.allclose(cmn.sum(axis=1), 1.0):
            raise RuntimeError(f"row normalization failed for {target}")
        counts.append(cm)
        normalized.append(cmn)
    return counts, normalized


def render(audit_csv: Path, output_stem: Path) -> None:
    counts, matrices = load_official_matrices(audit_csv)
    cmap = LinearSegmentedColormap.from_list("paper_blue", SEQUENTIAL, N=256)

    # Double-column master: 183 mm wide. A dedicated colorbar column prevents
    # the third matrix and its labels from being squeezed.
    fig = plt.figure(figsize=(183 / 25.4, 61 / 25.4))
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1, 1, 1, 0.055],
        left=0.105, right=0.965, bottom=0.24, top=0.87,
        wspace=0.28,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    for index, (ax, title, cm, cmn) in enumerate(zip(axes, PANEL_TITLES, counts, matrices)):
        image = ax.imshow(cmn, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
        ax.set_title(title, pad=7, fontweight="bold")
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES if index == 0 else [])
        ax.tick_params(axis="x", pad=3, length=2.5)
        ax.tick_params(axis="y", pad=3, length=0 if index else 2.5)

        # Thin white separators make the three classes readable without a boxy grid.
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(3):
            for j in range(3):
                value = cmn[i, j]
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center",
                    fontsize=7.2,
                    color="white" if value >= 0.55 else BLACK,
                    fontweight="bold" if i == j else "normal",
                )

    axes[0].set_ylabel("True class", labelpad=8)
    fig.supxlabel("Predicted class", y=0.075, fontsize=8)
    colorbar = fig.colorbar(image, cax=cax, ticks=np.linspace(0, 1, 6))
    colorbar.set_label("Row-normalized proportion", rotation=270, labelpad=11)
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(length=2.5, width=0.6, pad=2)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "source": str(audit_csv),
        "model": "cicman_v6ic",
        "seed": 42,
        "epoch_rule": "last",
        "recording_aggregation": OFFICIAL_AGGREGATION,
        "targets": TARGETS,
        "classes": CLASSES,
        "counts": {target: matrix.tolist() for target, matrix in zip(TARGETS, counts)},
    }
    output_stem.with_name(output_stem.name + "_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    audit = args.project_root / "outputs/tables/recording_aggregation_audit.csv"
    output = args.project_root / "outputs/figures/fig_confusion_v6ic"
    render(audit, output)
    print(f"saved {output}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
