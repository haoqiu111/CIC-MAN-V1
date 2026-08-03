#!/usr/bin/env python3
"""Schematic figures for the paper: SCM causal diagram + CIC-MAN framework.

Outputs fig_scm and fig_framework (PNG+PDF) to outputs/figures/ and paper/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})

OUT_DIRS = [
    Path(__file__).resolve().parents[2] / "outputs/figures",
    Path(__file__).resolve().parents[2] / "paper/figures",
]

BLUE, ORANGE, RED, GRAY, GREEN = "#4878d0", "#e49444", "#d1494e", "#777777", "#5f9e6e"


def save(fig, name):
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(d / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved {name}")


def box(ax, xy, w, h, text, fc="#eef2fb", ec=BLUE, fontsize=8, lw=1.2, style="round,pad=0.02"):
    b = FancyBboxPatch(xy, w, h, boxstyle=style, fc=fc, ec=ec, lw=lw)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return b


def arrow(ax, p, q, color=GRAY, lw=1.2, style="-|>", shrink=2, ls="-"):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=lw, color=color,
                        shrinkA=shrink, shrinkB=shrink, linestyle=ls)
    ax.add_patch(a)


def fig_scm():
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def node(x, y, label, ec):
        c = plt.Circle((x, y), 0.62, fc="white", ec=ec, lw=1.5)
        ax.add_patch(c)
        ax.text(x, y, label, ha="center", va="center", fontsize=10)

    node(1.2, 4.7, "$Y$", RED)
    node(1.2, 1.3, "$M$", BLUE)
    node(4.2, 4.7, "$Z_h$", RED)
    node(4.2, 1.3, "$Z_d$", BLUE)
    node(7.2, 3.0, "$X$", GRAY)
    arrow(ax, (1.85, 4.7), (3.55, 4.7), color=RED, lw=1.6, shrink=6)
    arrow(ax, (1.85, 1.3), (3.55, 1.3), color=BLUE, lw=1.6, shrink=6)
    arrow(ax, (4.75, 4.45), (6.7, 3.35), color=RED, lw=1.6, shrink=6)
    arrow(ax, (4.75, 1.55), (6.7, 2.65), color=BLUE, lw=1.6, shrink=6)
    ax.text(2.7, 5.05, "fault mechanism", fontsize=7.5, ha="center", color=RED)
    ax.text(2.7, 0.85, "rig / sensor chain", fontsize=7.5, ha="center", color=BLUE)
    ax.text(8.9, 3.0, "observed\nvibration", fontsize=7.5, ha="center", va="center")
    ax.text(5.0, 0.15, "intervention $T_v$, $T_{\\mathrm{hp}}$ act on the $Z_d\\!\\to\\!X$ path",
            fontsize=7.5, ha="center", style="italic")
    save(fig, "fig_scm")


def fig_framework():
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 13)
    ax.axis("off")

    views = ["raw", "denoise", "env_spec", "env_order", "stft", "cwt"]
    # input
    box(ax, (0.3, 5.6), 2.4, 1.6, "vibration\nwindow $x$", fc="#f5f5f5", ec=GRAY)
    box(ax, (0.3, 3.4), 2.4, 1.4, "HP-800Hz\n$T_{\\mathrm{hp}}(x)$", fc="#fdf3e6", ec=ORANGE)
    # view bank
    ys = [i * 1.9 + 1.0 for i in range(6)][::-1]
    for v, y in zip(views, ys):
        box(ax, (4.0, y), 2.4, 1.3, f"$T_v$: {v}", fc="#eef2fb", ec=BLUE)
        arrow(ax, (2.7, 6.4), (4.0, y + 0.65))
        box(ax, (7.2, y), 2.0, 1.3, f"$E_v$", fc="#eef2fb", ec=BLUE)
        arrow(ax, (6.4, y + 0.65), (7.2, y + 0.65))
        box(ax, (9.9, y + 0.72), 1.7, 0.62, "$z^h_v$", fc="#fbecec", ec=RED, fontsize=7.5)
        box(ax, (9.9, y), 1.7, 0.62, "$z^d_v$", fc="#e9f0fa", ec=BLUE, fontsize=7.5)
        arrow(ax, (9.2, y + 0.65), (9.9, y + 1.0))
        arrow(ax, (9.2, y + 0.65), (9.9, y + 0.3))
    arrow(ax, (2.7, 4.1), (4.0, 2.4), color=ORANGE, ls="--")
    ax.text(3.1, 2.6, "augment", fontsize=7, color=ORANGE, rotation=-38)

    # shared classifier + per-view logits
    box(ax, (12.6, 8.6), 2.6, 1.5, "shared\nclassifier $C$", fc="#fbecec", ec=RED)
    for y in ys:
        arrow(ax, (11.6, y + 1.0), (12.6, 9.3), color=RED, lw=0.8)
    box(ax, (16.2, 8.6), 2.6, 1.5, "per-view\nlogits $o_v$", fc="#fbecec", ec=RED)
    arrow(ax, (15.2, 9.35), (16.2, 9.35), color=RED)

    # router
    box(ax, (12.6, 4.6), 2.6, 2.6,
        "router $R$\n$[\\phi; c; e]$\n$+\\log\\pi$", fc="#eaf4ec", ec=GREEN)
    box(ax, (12.6, 1.6), 2.6, 2.0,
        "IC reliability\nprior $\\pi$\n(LOSO, min over\n$\\{\\mathrm{id},T_{\\mathrm{hp}}\\}$)",
        fc="#eaf4ec", ec=GREEN, fontsize=7)
    arrow(ax, (13.9, 3.6), (13.9, 4.6), color=GREEN)
    box(ax, (16.2, 5.0), 2.6, 1.5, "weights $w$", fc="#eaf4ec", ec=GREEN)
    arrow(ax, (15.2, 5.9), (16.2, 5.8), color=GREEN)

    # fusion
    box(ax, (19.8, 6.8), 3.4, 1.8, "fused logits\n$o=\\sum_v w_v o_v$\n$\\Rightarrow \\hat{y}$",
        fc="#fbecec", ec=RED)
    arrow(ax, (18.8, 9.3), (20.4, 8.6), color=RED)
    arrow(ax, (18.8, 5.75), (20.4, 6.9), color=GREEN)

    # disentanglement heads
    box(ax, (19.8, 3.6), 3.4, 1.6, "GRL adversary $A$\non fused $z_h$", fc="#f5f5f5", ec=GRAY, fontsize=7.5)
    box(ax, (19.8, 1.4), 3.4, 1.6, "domain head $D$ on $z_d$\n$\\perp$ orthogonality", fc="#f5f5f5", ec=GRAY, fontsize=7.5)

    ax.text(5.2, 12.5, "six heterogeneous intervention agents", fontsize=8.5, ha="center", style="italic")
    ax.text(17.5, 12.5, "causal routing and consensus fusion", fontsize=8.5, ha="center", style="italic")
    save(fig, "fig_framework")


if __name__ == "__main__":
    fig_scm()
    fig_framework()
