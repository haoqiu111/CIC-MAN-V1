#!/usr/bin/env python3
"""Nature-style, code-faithful CIC-MAN framework figure.

Designed at 183 mm two-column width.  The SVG keeps text editable and the PDF
uses TrueType fonts.  No target-domain data enters prior estimation, training,
model selection, or routing; the target is used only for final inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUTS = [ROOT / "paper/figures", ROOT / "outputs/figures"]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "text.color": "#252A30",
    }
)

# Restrained method-family palette: neutral scaffold, blue mechanism, warm
# counterfactual, green reliability, violet domain branch, red supervision.
INK = "#252A30"
MUTED = "#65717D"
LINE = "#78838E"
PANEL = "#F8FAFC"
BLUE = "#376FB6"
BLUE_F = "#EDF3FB"
ORANGE = "#C96D1B"
ORANGE_F = "#FFF3E7"
GREEN = "#347A4B"
GREEN_F = "#EDF6EF"
TEAL = "#267F7A"
TEAL_F = "#EAF6F5"
RED = "#B84242"
RED_F = "#FBEFEF"
VIOLET = "#7357A8"
VIOLET_F = "#F2EFF9"
GRAY_F = "#F0F2F4"

fig, ax = plt.subplots(figsize=(7.20, 5.02))  # 183 x 127.5 mm
ax.set_xlim(0, 14.4)
ax.set_ylim(0, 10.0)
ax.axis("off")


def rounded(x, y, w, h, fc="white", ec=LINE, lw=0.75, radius=0.07, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def box(x, y, w, h, title, body="", *, fc="white", ec=LINE,
        tc=INK, title_fs=6.6, body_fs=5.7, lw=0.75):
    rounded(x, y, w, h, fc, ec, lw)
    if body:
        ax.text(x + w / 2, y + h - 0.15, title, ha="center", va="top",
                fontsize=title_fs, fontweight="bold", color=tc, zorder=5)
        ax.text(x + w / 2, y + 0.10 + (h - 0.38) / 2, body,
                ha="center", va="center", fontsize=body_fs,
                color=INK, linespacing=1.12, zorder=5)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=tc, zorder=5)


def arrow(x1, y1, x2, y2, *, color=INK, lw=0.85, ls="-", rad=0.0,
          head=6.5, z=1):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=head,
        linewidth=lw, linestyle=ls, color=color, shrinkA=1.2, shrinkB=1.2,
        connectionstyle=f"arc3,rad={rad}", zorder=z,
    ))


def pill(x, y, w, text, *, color=LINE, fc="white", fs=5.25, bold=False):
    rounded(x, y, w, 0.33, fc, color, 0.6, radius=0.12, z=5)
    ax.text(x + w / 2, y + 0.165, text, ha="center", va="center",
            fontsize=fs, color=color, fontweight="bold" if bold else "normal", zorder=6)


def panel_label(letter, x, y, title):
    ax.text(x, y, letter, fontsize=8.0, fontweight="bold", va="center", ha="left")
    ax.text(x + 0.34, y, title, fontsize=7.7, fontweight="bold", va="center", ha="left")


# ---------------------------------------------------------------------------
# a | Source-only prior (supporting evidence strip)
# ---------------------------------------------------------------------------
rounded(0.10, 7.48, 14.20, 2.38, PANEL, "#C2C8CE", 0.7, radius=0.06, z=0)
panel_label("a", 0.24, 9.58, "Offline source-only intervention-consistent prior")
ax.text(13.98, 9.58, "target unseen", ha="right", va="center",
        fontsize=5.6, color=GREEN, fontweight="bold")

box(0.28, 8.03, 1.45, 1.13, "Sources", "$\\mathcal{S}_1,\\ldots,\\mathcal{S}_K$\nlabeled recordings",
    fc=GRAY_F, ec=LINE, title_fs=6.5, body_fs=5.5)
box(2.10, 8.70, 2.20, 0.72, "Identity views", "$T_v(x)$",
    fc=BLUE_F, ec=BLUE, tc=BLUE, title_fs=6.4, body_fs=5.8)
box(2.10, 7.82, 2.20, 0.72, "HP-800 counterfactual", "$T_v(T_{hp}(x))$",
    fc=ORANGE_F, ec=ORANGE, tc=ORANGE, title_fs=6.2, body_fs=5.7)
box(4.75, 8.03, 2.35, 1.13, "Per-view LOSO probes",
    "train $\\mathcal{S}\\setminus\\mathcal{S}_k$; test $\\mathcal{S}_k$\n$\\bar F_v^{id}$ and $\\bar F_v^{hp}$",
    fc=BLUE_F, ec=BLUE, tc=BLUE, title_fs=6.4, body_fs=5.35)
box(7.55, 8.03, 2.15, 1.13, "Worst-variant reliability",
    "$\\rho_v=\\min(\\bar F_v^{id},\\bar F_v^{hp})$\npenalizes shortcut views",
    fc=ORANGE_F, ec=ORANGE, tc=ORANGE, title_fs=6.25, body_fs=5.35)
box(10.15, 8.03, 2.05, 1.13, "Frozen routing prior",
    "$\\pi=\\mathrm{softmax}(\\rho/T_\\pi)$\nsource-only anchor",
    fc=GREEN_F, ec=GREEN, tc=GREEN, title_fs=6.35, body_fs=5.45)
box(12.65, 8.13, 1.32, 0.93, "$\\log\\boldsymbol{\\pi}$", "fed to router",
    fc=GREEN_F, ec=GREEN, tc=GREEN, title_fs=7.0, body_fs=5.2)

arrow(1.73, 8.60, 2.10, 9.06, color=BLUE, rad=-0.12)
arrow(1.73, 8.55, 2.10, 8.18, color=ORANGE, rad=0.12)
arrow(4.30, 9.06, 4.75, 8.74, color=BLUE)
arrow(4.30, 8.18, 4.75, 8.45, color=ORANGE)
arrow(7.10, 8.60, 7.55, 8.60, color=BLUE)
arrow(9.70, 8.60, 10.15, 8.60, color=ORANGE)
arrow(12.20, 8.60, 12.65, 8.60, color=GREEN)
ax.text(4.76, 7.64, "internal probe aggregation: probability-sum recording macro-F1",
        fontsize=4.9, color=MUTED, ha="left", va="center")

# ---------------------------------------------------------------------------
# b | Main CIC-MAN mechanism (hero panel)
# ---------------------------------------------------------------------------
rounded(0.10, 0.12, 14.20, 7.12, PANEL, "#C2C8CE", 0.7, radius=0.06, z=0)
panel_label("b", 0.24, 6.97, "CIC-MAN training and target-free deployment")
ax.text(14.00, 6.97, "solid: forward  ·  dashed: prior / update",
        ha="right", va="center", fontsize=4.9, color=MUTED)

# Inputs and intervention bank.
box(0.28, 3.72, 1.35, 1.42, "Source window",
    "$x\\in\\mathbb{R}^{4096}$\nlabel $y$; rig $m$\nmetadata $\\phi\\in\\mathbb{R}^{16}$",
    fc=GRAY_F, ec=LINE, title_fs=6.4, body_fs=5.25)
box(0.28, 2.28, 1.35, 0.91, "HP-800 twin", "$T_{hp}(x)$; same $y$",
    fc=ORANGE_F, ec=ORANGE, tc=ORANGE, title_fs=6.3, body_fs=5.25)
pill(0.23, 1.78, 1.45, "union only; no paired loss", color=ORANGE, fs=4.55)

rounded(1.96, 2.10, 2.20, 3.65, BLUE_F, BLUE, 0.75)
ax.text(3.06, 5.55, "Six intervention views", ha="center", va="center",
        fontsize=6.8, color=BLUE, fontweight="bold")
views = [("$T_{raw}$", "wave"), ("$T_{denoise}$", "wave"),
         ("$T_{env\\_spec}$", "spectrum"), ("$T_{env\\_order}$", "order"),
         ("$T_{stft}$", "map"), ("$T_{cwt}$", "map")]
for i, (name, kind) in enumerate(views):
    col, row = i % 2, i // 2
    x, y = 2.12 + col * 1.00, 4.70 - row * 0.82
    rounded(x, y, 0.88, 0.60, "white", "#A9C2E5", 0.48, radius=0.05, z=3)
    ax.text(x + 0.44, y + 0.36, name, ha="center", va="center", fontsize=5.35, zorder=5)
    ax.text(x + 0.44, y + 0.13, kind, ha="center", va="center", fontsize=4.4,
            color=MUTED, zorder=5)
ax.text(3.06, 2.35, "1-D: wave/spectrum   ·   2-D: maps", ha="center", va="center",
        fontsize=4.55, color=MUTED)

box(4.52, 3.22, 1.45, 1.84, "Six encoders $E_v$",
    "data-type matched\n$\\boldsymbol{f}_v\\in\\mathbb{R}^{128}$\nindependent agents",
    fc=BLUE_F, ec=BLUE, tc=BLUE, title_fs=6.5, body_fs=5.35)

# Health/domain split.
box(6.34, 4.34, 1.55, 1.05, "Health evidence",
    "$\\boldsymbol{z}_v^h\\in\\mathbb{R}^{64}$\nfault-relevant",
    fc=RED_F, ec=RED, tc=RED, title_fs=6.25, body_fs=5.2)
box(6.34, 2.63, 1.55, 1.05, "Domain evidence",
    "$\\boldsymbol{z}_v^d\\in\\mathbb{R}^{32}$\nrig-specific",
    fc=VIOLET_F, ec=VIOLET, tc=VIOLET, title_fs=6.25, body_fs=5.2)
pill(6.22, 3.88, 1.78, "$\\mathcal{L}_{orth}:\\ z_v^h\\perp z_v^d$", color=VIOLET, fs=4.95)

# Heads and router.
box(8.26, 4.35, 1.55, 1.04, "Shared classifier $C$",
    "$\\boldsymbol{o}_v=C(\\boldsymbol{z}_v^h)$\nper-view logits",
    fc=RED_F, ec=RED, tc=RED, title_fs=6.05, body_fs=5.2)
pill(8.24, 5.57, 1.60, "$\\mathcal{L}_{view}$: CE per view", color=RED, fs=4.8)
box(8.26, 2.35, 1.55, 1.36, "Domain constraints",
    "$D(z_v^d)\\to m$  [$\\mathcal{L}_{dom}$]\n$A(\\mathrm{GRL}(\\bar z_w^h))\\to m$\n[$\\mathcal{L}_{adv}$]",
    fc=VIOLET_F, ec=VIOLET, tc=VIOLET, title_fs=6.0, body_fs=4.9)

box(10.18, 2.58, 2.12, 2.96, "Causal evidence router $R$",
    "$c_v=\\|z_v^h-\\bar z^h\\|_2$\n$e_v=H[\\mathrm{softmax}(o_v)]$\n\n$\\boldsymbol{r}=R([\\phi;\\boldsymbol{c};\\boldsymbol{e}])+\\log\\pi$\n$\\boldsymbol{w}=\\mathrm{softmax}(\\boldsymbol{r})$\n\nview dropout $p=0.15$\n$\\mathcal{L}_{prior}=KL(w\\|\\pi)$",
    fc=TEAL_F, ec=TEAL, tc=TEAL, title_fs=6.35, body_fs=5.05)
pill(10.16, 5.72, 2.16, "$\\mathcal{L}_{cons}$: trusted-view agreement", color=TEAL, fs=4.7)

# Outputs.
box(12.66, 4.67, 1.36, 0.86, "Weighted fusion",
    "$\\boldsymbol{o}=\\sum_v w_v\\boldsymbol{o}_v$",
    fc=RED_F, ec=RED, tc=RED, title_fs=6.2, body_fs=5.2)
box(12.66, 3.48, 1.36, 0.86, "Window prediction",
    "normal / inner / outer",
    fc=RED_F, ec=RED, tc=RED, title_fs=6.0, body_fs=4.95)
box(12.55, 2.08, 1.58, 1.04, "Recording aggregation",
    "majority over windows\ntie: $\\sum p$ (tied classes) $\\to$ min id",
    fc=GRAY_F, ec=LINE, tc=INK, title_fs=5.9, body_fs=4.65)
ax.text(13.34, 5.82, "TARGET: inference only", fontsize=5.15, color=GREEN,
        fontweight="bold", ha="center", va="center")

# Forward path.
arrow(1.63, 4.42, 1.96, 4.42, color=BLUE)
arrow(0.96, 3.72, 0.96, 3.19, color=ORANGE)
arrow(1.63, 2.73, 1.96, 3.06, color=ORANGE)
arrow(4.16, 4.08, 4.52, 4.08, color=BLUE)
arrow(5.97, 4.45, 6.34, 4.87, color=RED)
arrow(5.97, 3.72, 6.34, 3.16, color=VIOLET)
arrow(7.89, 4.87, 8.26, 4.87, color=RED)
arrow(7.89, 3.16, 8.26, 3.16, color=VIOLET)
arrow(7.89, 4.58, 10.18, 4.18, color=TEAL, lw=0.72, rad=0.13)
arrow(9.81, 4.84, 10.18, 4.60, color=TEAL)
arrow(9.81, 5.02, 12.66, 5.10, color=RED, lw=0.72, rad=-0.22)
arrow(12.30, 4.35, 12.66, 5.00, color=TEAL, lw=0.95)
arrow(13.34, 4.67, 13.34, 4.34, color=RED)
arrow(13.34, 3.48, 13.34, 3.12, color=INK)

# Frozen prior and metadata: direct labels prevent ambiguous causal paths.
arrow(13.31, 8.13, 11.65, 5.54, color=GREEN, lw=0.9, ls="--", rad=0.20)
ax.text(12.20, 6.24, "frozen $\\log\\pi$", fontsize=5.0, color=GREEN,
        fontweight="bold", ha="center")
ax.plot([1.63, 1.78, 9.98], [4.78, 6.18, 6.18],
        color=LINE, lw=0.58, zorder=1)
arrow(9.98, 6.18, 10.18, 5.20, color=LINE, lw=0.58, head=5.5)
ax.text(8.90, 6.27, "$\\phi$: router metadata only", fontsize=4.7, color=MUTED, ha="center")

# Objective and feedback rail.
rounded(2.00, 0.50, 10.35, 0.88, "white", "#D9A0A0", 0.7, radius=0.05)
ax.text(2.20, 1.12, "Training objective", fontsize=5.9, color=RED,
        fontweight="bold", ha="left", va="center")
ax.text(7.55, 0.91,
        "$\\mathcal{L}=\\mathcal{L}_{cls}+0.3\\mathcal{L}_{view}+0.1\\mathcal{L}_{cons}"
        "+0.1\\mathcal{L}_{adv}+0.1\\mathcal{L}_{dom}+0.05\\mathcal{L}_{orth}+1.0\\mathcal{L}_{prior}$",
        fontsize=5.3, ha="center", va="center")
ax.text(12.12, 0.62, "AdamW · end-to-end", fontsize=4.8, color=RED,
        fontweight="bold", ha="right", va="center")
pill(12.55, 0.67, 1.58, "$\\mathcal{L}_{cls}$ (+ optional hard weighting)", color=RED, fs=4.5)

# One update backbone with explicit trainable destinations; avoids a web of
# loss arrows while preserving the feedback semantics.
ax.plot([3.30, 11.20], [1.55, 1.55], color=RED, lw=0.7, ls="--", zorder=1)
for x, y in [(5.24, 3.22), (7.12, 2.63), (9.03, 2.35), (11.24, 2.58)]:
    arrow(x, 1.55, x, y, color=RED, lw=0.7, ls="--", head=5.5, z=1)
ax.text(7.25, 1.66, "back-propagation updates encoders, projections, heads and router",
        fontsize=4.55, color=RED, ha="center", va="bottom")

# Final-size integrity note (figure content, not a claim): source-only boundary.
ax.text(0.25, 0.31, "Last-epoch frozen deployment; no target adaptation or target-based selection",
        fontsize=4.7, color=MUTED, ha="left", va="center")

fig.subplots_adjust(left=0.006, right=0.994, bottom=0.008, top=0.994)

for out in OUTS:
    out.mkdir(parents=True, exist_ok=True)
    base = out / "fig_framework_p1_nature"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.025,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    # Manuscript aliases.
    fig.savefig(out / "fig_framework_p1.svg", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out / "fig_framework_p1.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out / "fig_framework_p1.png", dpi=600, bbox_inches="tight", pad_inches=0.025)

qa = {
    "core_conclusion": "A strictly source-only worst-variant reliability prior anchors six-view evidence routing without target adaptation or selection.",
    "archetype": "schematic-led composite",
    "backend": "python/matplotlib",
    "final_size_mm": [183.0, 127.5],
    "primary_format": "SVG with editable text",
    "minimum_nominal_font_pt": 4.5,
    "review_risks_addressed": [
        "target is inference-only",
        "union augmentation is not a paired-consistency loss",
        "metadata enters only the router",
        "internal probe aggregation differs from official target aggregation",
        "last-epoch majority-vote recording protocol is explicit",
    ],
}
(ROOT / "outputs/figures/fig_framework_p1_nature_qa.json").write_text(
    json.dumps(qa, indent=2), encoding="utf-8"
)
plt.close(fig)
print("saved Nature-style CIC-MAN framework (SVG/PDF/TIFF/PNG)")
