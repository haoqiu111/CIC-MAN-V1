#!/usr/bin/env python3
"""Publication framework figure for Paper 1 (CIC-MAN).

The diagram is deliberately code-faithful:
  * Stage 1 uses source-only LOSO probes on identity and HP-800 caches.
  * Stage 2 trains on the union of original and HP-800 windows; there is no
    explicit paired-consistency loss.
  * The router consumes the 16-D metadata descriptor, detached consensus
    distances and detached prediction entropies, plus the frozen log prior.
  * Recording predictions use the frozen official majority/tie-break protocol.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [
    ROOT / "outputs/figures",
    ROOT / "paper/figures",
]

# Color-blind-safe, restrained editorial palette.
INK = "#17212B"
MUTED = "#52606D"
LINE = "#7C8793"
PANEL = "#F7F9FC"
BLUE = "#2F66B3"
BLUE_F = "#EAF1FB"
TEAL = "#177C78"
TEAL_F = "#E8F5F3"
ORANGE = "#C66A1B"
ORANGE_F = "#FFF2E6"
RED = "#B33A3A"
RED_F = "#FCECEC"
PURPLE = "#6651A3"
PURPLE_F = "#F1EEFA"
GREEN = "#327448"
GREEN_F = "#EAF4ED"
GRAY_F = "#EEF1F4"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 8.2,
        "text.color": INK,
        "axes.unicode_minus": True,
    }
)

fig, ax = plt.subplots(figsize=(16.0, 9.25))
ax.set_xlim(0, 32)
ax.set_ylim(0, 18.5)
ax.axis("off")


def rounded(x, y, w, h, fc, ec=LINE, lw=1.1, radius=0.10, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.035,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(p)
    return p


def text_box(x, y, w, h, title, body="", *, fc=GRAY_F, ec=LINE,
             title_color=INK, title_size=8.3, body_size=7.1, lw=1.1):
    rounded(x, y, w, h, fc, ec, lw)
    if body:
        ax.text(x + w / 2, y + h - 0.27, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=title_color, zorder=4)
        ax.text(x + w / 2, y + 0.16 + (h - 0.58) / 2, body, ha="center", va="center",
                fontsize=body_size, color=MUTED, linespacing=1.22, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color, zorder=4)


def arrow(x1, y1, x2, y2, *, color=LINE, lw=1.25, style="-", rad=0.0,
          head=8, z=3):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=head,
        linewidth=lw, linestyle=style, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5, zorder=z,
    ))


def line(x1, y1, x2, y2, *, color=LINE, lw=1.0, style="-", z=2):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=style, zorder=z)


def pill(x, y, w, text, *, fc="white", ec=LINE, color=INK, fs=6.7):
    rounded(x, y, w, 0.42, fc, ec, 0.85, radius=0.16, z=5)
    ax.text(x + w / 2, y + 0.21, text, ha="center", va="center",
            fontsize=fs, color=color, zorder=6)


def panel(y, h, number, title, subtitle, color):
    rounded(0.25, y, 31.5, h, PANEL, "#A7B0BA", 1.0, radius=0.10, z=0)
    rounded(0.55, y + h - 0.70, 1.22, 0.46, color, color, 0, radius=0.17, z=5)
    ax.text(1.16, y + h - 0.47, f"STAGE {number}", ha="center", va="center",
            fontsize=8.0, color="white", fontweight="bold", zorder=6)
    ax.text(2.05, y + h - 0.45, title, ha="left", va="center",
            fontsize=10.6, fontweight="bold", color=INK, zorder=6)
    ax.text(2.05, y + h - 0.80, subtitle, ha="left", va="center",
            fontsize=6.9, color=MUTED, zorder=6)


def waveform_icon(x, y, w, h, kind, color):
    """Small deterministic signal glyph; purely explanatory."""
    xx = np.linspace(0, 1, 120)
    if kind == "raw":
        yy = 0.18*np.sin(2*np.pi*7*xx) + 0.11*np.sin(2*np.pi*19*xx)
        yy += 0.32*np.exp(-((xx-0.53)/0.055)**2)*np.sin(2*np.pi*28*xx)
        ax.plot(x + xx*w, y + h*(0.5 + yy), color=color, lw=0.75, zorder=5)
    elif kind == "denoise":
        yy = 0.20*np.sin(2*np.pi*6*xx) + 0.25*np.exp(-((xx-0.54)/0.07)**2)*np.sin(2*np.pi*20*xx)
        ax.plot(x + xx*w, y + h*(0.5 + yy), color=color, lw=0.8, zorder=5)
    elif kind in {"env_spec", "env_order"}:
        base = y + 0.18*h
        peaks = [0.16, 0.32, 0.49, 0.66, 0.82] if kind == "env_order" else [0.12, 0.28, 0.57, 0.74]
        heights = [0.55, 0.85, 0.42, 0.68, 0.35] if kind == "env_order" else [0.42, 0.80, 0.55, 0.31]
        line(x, base, x+w, base, color=LINE, lw=0.45, z=4)
        for p, q in zip(peaks, heights):
            line(x+p*w, base, x+p*w, base+q*h*0.72, color=color, lw=0.9, z=5)
    else:
        # Compact time-frequency texture for STFT/CWT.
        nrow, ncol = 5, 8
        for r in range(nrow):
            for c in range(ncol):
                val = (np.sin((r+1)*(c+2)*0.7) + 1) / 2
                alpha = 0.10 + 0.55*val
                ax.add_patch(Rectangle((x+c*w/ncol, y+r*h/nrow), w/ncol, h/nrow,
                                       facecolor=color, edgecolor="none", alpha=alpha, zorder=4))


# ---------------------------------------------------------------------------
# STAGE 1: source-only reliability prior
# ---------------------------------------------------------------------------
panel(13.15, 5.05, 1, "Source-only intervention-consistent reliability prior",
      "Offline LOSO probing; the held-out target is never used for training, selection, routing or prior estimation.", BLUE)

text_box(0.85, 14.15, 3.05, 2.15, "Labeled source recordings",
         "$\\mathcal{S}=\\{\\mathcal{S}_1,\\ldots,\\mathcal{S}_K\\}$\nrecording-level splits\ntarget dataset unseen",
         fc=GRAY_F, ec=LINE, title_size=8.1)

# Two intervention lanes.
text_box(4.75, 15.27, 4.10, 1.17, "Identity cache", "$T_v(x)$ for each of six views",
         fc=BLUE_F, ec=BLUE, title_color=BLUE)
text_box(4.75, 13.79, 4.10, 1.17, "Measurement counterfactual",
         "$T_v(T_{hp}(x))$; HP-800 twin cache", fc=ORANGE_F, ec=ORANGE, title_color=ORANGE)
arrow(3.90, 15.55, 4.75, 15.86, color=BLUE)
arrow(3.90, 14.83, 4.75, 14.38, color=ORANGE)

text_box(9.70, 14.17, 4.65, 2.14, "Per-view LOSO probes",
         "for each $v,k$: train on $\\mathcal{S}\\setminus\\mathcal{S}_k$\nscore on $\\mathcal{S}_k$\nrecording macro-F1 (internal probability-sum)",
         fc="white", ec=BLUE, title_color=BLUE, body_size=6.8)
arrow(8.85, 15.86, 9.70, 15.55, color=BLUE)
arrow(8.85, 14.38, 9.70, 14.83, color=ORANGE)

text_box(15.15, 14.17, 4.20, 2.14, "Cross-source reliability",
         "$\\bar F^{id}_v=K^{-1}\\sum_k F^{(k)}_{v,id}$\n$\\bar F^{hp}_v=K^{-1}\\sum_k F^{(k)}_{v,hp}$",
         fc=BLUE_F, ec=BLUE, title_color=BLUE, body_size=7.2)
arrow(14.35, 15.24, 15.15, 15.24, color=BLUE)

text_box(20.15, 14.17, 4.15, 2.14, "Worst-variant operator",
         "$\\rho_v=\\min(\\bar F^{id}_v,\\bar F^{hp}_v)$\npenalizes low-frequency\nshortcut dependence",
         fc=ORANGE_F, ec=ORANGE, title_color=ORANGE, body_size=7.0)
arrow(19.35, 15.24, 20.15, 15.24, color=ORANGE)

text_box(25.10, 14.17, 3.55, 2.14, "Frozen routing prior",
         "$\\pi_v=\\mathrm{softmax}(\\rho_v/T_\\pi)$\nsource-only reliability anchor",
         fc=GREEN_F, ec=GREEN, title_color=GREEN, body_size=7.2)
arrow(24.30, 15.24, 25.10, 15.24, color=GREEN)

text_box(29.20, 14.17, 1.85, 2.14, "Artifact",
         "$\\log\\boldsymbol{\\pi}$\nfixed in\nStage 2", fc=GREEN_F, ec=GREEN,
         title_color=GREEN, title_size=7.5, body_size=6.7)
arrow(28.65, 15.24, 29.20, 15.24, color=GREEN)

# ---------------------------------------------------------------------------
# STAGE 2: intervention-augmented end-to-end training and deployment
# ---------------------------------------------------------------------------
panel(0.35, 12.30, 2, "CIC-MAN training and target-free deployment",
      "Solid arrows: forward data flow   ·   dashed red: supervised loss/update path   ·   dashed green: frozen Stage-1 prior.", TEAL)

# Input / union construction
text_box(0.70, 7.15, 2.70, 2.12, "Source window",
         "$x\\in\\mathbb{R}^{4096}$\nlabel $y$, source rig $m$\n+ 16-D router metadata $\\boldsymbol{\\phi}$",
         fc=GRAY_F, ec=LINE, body_size=6.8)
text_box(0.70, 4.48, 2.70, 1.76, "HP-800 twin", "$T_{hp}(x)$\nsame class label $y$",
         fc=ORANGE_F, ec=ORANGE, title_color=ORANGE, body_size=6.8)
arrow(2.05, 7.15, 2.05, 6.24, color=ORANGE)
pill(0.52, 3.83, 3.08, "union sampling; no paired-consistency loss", fc="white", ec=ORANGE, color=ORANGE, fs=5.85)

# Six view bank with signal glyphs
rounded(4.05, 3.82, 4.35, 6.28, BLUE_F, BLUE, 1.2)
ax.text(6.23, 9.80, "Six soft measurement interventions", ha="center", va="center",
        fontsize=8.4, fontweight="bold", color=BLUE, zorder=5)
view_names = [
    ("raw", "$T_{raw}$"), ("denoise", "$T_{denoise}$"),
    ("env_spec", "$T_{env\\_spec}$"), ("env_order", "$T_{env\\_order}$"),
    ("stft", "$T_{stft}$"), ("cwt", "$T_{cwt}$"),
]
for i, (kind, label) in enumerate(view_names):
    yy = 8.95 - i*0.87
    rounded(4.35, yy, 3.75, 0.67, "white", "#B4C6E1", 0.65, radius=0.06, z=3)
    waveform_icon(4.55, yy+0.11, 1.52, 0.43, kind, BLUE)
    ax.text(6.37, yy+0.34, label, ha="left", va="center", fontsize=7.0, zorder=5)
arrow(3.40, 8.25, 4.05, 7.75, color=BLUE)
arrow(3.40, 5.34, 4.05, 5.95, color=ORANGE)

# Agent stack and disentanglement
text_box(9.20, 5.08, 2.55, 3.75, "Six independent agents",
         "data-type-matched $E_v$\n1-D CNN: wave/spectrum\n2-D CNN: STFT/CWT\n\nfeature $\\boldsymbol{f}_v\\in\\mathbb{R}^{128}$",
         fc=BLUE_F, ec=BLUE, title_color=BLUE, body_size=6.8)
arrow(8.40, 6.95, 9.20, 6.95, color=BLUE, lw=1.6)

text_box(12.45, 7.28, 2.55, 1.70, "Health projection",
         "$\\boldsymbol{z}^h_v\\in\\mathbb{R}^{64}$\nfault-relevant evidence",
         fc=RED_F, ec=RED, title_color=RED, body_size=6.9)
text_box(12.45, 4.94, 2.55, 1.70, "Domain projection",
         "$\\boldsymbol{z}^d_v\\in\\mathbb{R}^{32}$\nrig-specific evidence",
         fc=PURPLE_F, ec=PURPLE, title_color=PURPLE, body_size=6.9)
arrow(11.75, 7.28, 12.45, 8.05, color=RED)
arrow(11.75, 6.15, 12.45, 5.78, color=PURPLE)

# Classifier and evidence extraction
text_box(15.80, 7.35, 3.08, 1.58, "Shared classifier $C$",
         "$\\boldsymbol{o}_v=C(\\boldsymbol{z}^h_v)$\nper-view logits / probabilities",
         fc=RED_F, ec=RED, title_color=RED, body_size=6.9)
arrow(15.00, 8.13, 15.80, 8.13, color=RED)

text_box(15.80, 4.83, 3.08, 1.65, "Domain head $D$",
         "$D(\\boldsymbol{z}^d_v)\\rightarrow m$\n$\\mathcal{L}_{dom}$ captures rig evidence",
         fc=PURPLE_F, ec=PURPLE, title_color=PURPLE, body_size=6.8)
arrow(15.00, 5.79, 15.80, 5.79, color=PURPLE)

# Router input bundle
rounded(19.70, 4.83, 4.20, 4.10, TEAL_F, TEAL, 1.3)
ax.text(21.80, 8.60, "Causal evidence router $R$", ha="center", va="center",
        fontsize=8.5, fontweight="bold", color=TEAL, zorder=5)
ax.text(21.80, 7.91, "$c_v=\\|\\boldsymbol{z}^h_v-\\bar{\\boldsymbol{z}}^h\\|_2$",
        ha="center", va="center", fontsize=7.0, zorder=5)
ax.text(21.80, 7.36, "$e_v=H[\\mathrm{softmax}(\\boldsymbol{o}_v)]$",
        ha="center", va="center", fontsize=7.0, zorder=5)
ax.text(21.80, 6.80, "$\\boldsymbol{r}=R([\\boldsymbol{\\phi};\\boldsymbol{c};\\boldsymbol{e}])+\\log\\boldsymbol{\\pi}$",
        ha="center", va="center", fontsize=7.0, zorder=5)
ax.text(21.80, 6.24, "$\\boldsymbol{w}=\\mathrm{softmax}(\\boldsymbol{r})$",
        ha="center", va="center", fontsize=7.0, fontweight="bold", zorder=5)
pill(20.13, 5.30, 3.33, "view dropout $p=0.15$ (train only)", fc="white", ec=TEAL, color=TEAL, fs=6.4)
arrow(15.00, 8.28, 19.70, 8.08, color=TEAL, rad=-0.10)  # z_h -> c
arrow(18.88, 8.05, 19.70, 7.48, color=TEAL)             # logits -> entropy
# Metadata bypass: route above the representation stack so it is visibly a
# router-only input rather than a causal health feature.
line(3.40, 8.76, 3.70, 10.30, color=LINE, lw=0.85)
line(3.70, 10.30, 19.15, 10.30, color=LINE, lw=0.85)
arrow(19.15, 10.30, 20.15, 8.93, color=LINE, lw=0.9)
ax.text(16.80, 10.48, "$\\boldsymbol{\\phi}\\in\\mathbb{R}^{16}$ (router metadata only)",
        fontsize=6.4, color=MUTED, ha="center", va="center", zorder=7)

# Frozen prior feed
arrow(30.10, 14.17, 22.85, 8.93, color=GREEN, lw=1.6, style="--", rad=0.12)
pill(26.30, 10.33, 2.55, "frozen $\\log\\boldsymbol{\\pi}$", fc="white", ec=GREEN, color=GREEN, fs=6.6)

# Fusion, window prediction, recording aggregation
text_box(24.70, 7.22, 2.63, 1.78, "Weighted fusion",
         "$\\boldsymbol{o}=\\sum_v w_v\\boldsymbol{o}_v$\n$\\bar{\\boldsymbol{z}}^{h}_w=\\sum_v \\mathrm{sg}(w_v)\\boldsymbol{z}^h_v$",
         fc=RED_F, ec=RED, title_color=RED, title_size=7.7, body_size=6.9)
arrow(23.90, 7.02, 24.70, 8.10, color=TEAL, lw=1.5)
arrow(18.88, 8.41, 24.70, 8.52, color=RED, lw=1.15, rad=-0.04)

text_box(28.10, 7.22, 2.85, 1.78, "Window prediction",
         "$p(y\\mid x)=\\mathrm{softmax}(\\boldsymbol{o})$\nnormal / inner / outer",
         fc=RED_F, ec=RED, title_color=RED, body_size=6.9)
arrow(27.33, 8.11, 28.10, 8.11, color=RED)

text_box(26.58, 4.35, 4.37, 1.76, "Official recording aggregation",
         "majority vote over windows\ntie: cumulative probability $\\rightarrow$ smallest class id",
         fc=GRAY_F, ec=LINE, title_size=7.7, body_size=6.6)
arrow(29.53, 7.22, 29.10, 6.11, color=LINE)
ax.text(28.77, 4.08, "Frozen last-epoch model · no target adaptation or selection",
        ha="center", va="top", fontsize=6.55, color=MUTED, zorder=5)

# Disentanglement adversary and orthogonality
text_box(15.80, 2.48, 3.08, 1.55, "GRL domain adversary $A$",
         "$A(\\mathrm{GRL}(\\bar{\\boldsymbol{z}}^h_w))\\rightarrow m$\n$\\mathcal{L}_{adv}$ removes rig evidence",
         fc=PURPLE_F, ec=PURPLE, title_color=PURPLE, body_size=6.55)
arrow(25.10, 7.22, 18.88, 3.28, color=PURPLE, lw=1.0, style="--", rad=-0.13)
pill(11.60, 3.58, 4.20, "$\\mathcal{L}_{orth}$: cosine decorrelation $\\boldsymbol{z}^h_v\\perp\\boldsymbol{z}^d_v$",
     fc="white", ec=PURPLE, color=PURPLE, fs=6.5)
line(13.73, 4.94, 13.73, 4.00, color=PURPLE, lw=0.9, style="--")

# Loss rail, labels and optimizer feedback
rounded(4.05, 0.78, 26.90, 1.14, "white", "#D2A2A2", 1.0, radius=0.08, z=2)
ax.text(4.35, 1.56, "Training objective", ha="left", va="center", fontsize=7.4,
        fontweight="bold", color=RED, zorder=5)
ax.text(17.47, 1.20,
        "$\\mathcal{L}=\\mathcal{L}_{cls}+0.3\\mathcal{L}_{view}+0.1\\mathcal{L}_{cons}"
        "+0.1\\mathcal{L}_{adv}+0.1\\mathcal{L}_{dom}+0.05\\mathcal{L}_{orth}+1.0\\mathcal{L}_{prior}$",
        ha="center", va="center", fontsize=7.35, zorder=5)
ax.text(30.55, 1.56, "AdamW", ha="right", va="center", fontsize=7.4,
        fontweight="bold", color=RED, zorder=5)

pill(24.18, 3.47, 2.60, "$\\mathcal{L}_{cls}$ (+ optional hard weighting)", fc="white", ec=RED, color=RED, fs=6.2)
pill(15.55, 9.25, 3.58, "$\\mathcal{L}_{view}$: CE on every $\\boldsymbol{o}_v$", fc="white", ec=RED, color=RED, fs=6.35)
pill(19.78, 9.25, 4.00, "$\\mathcal{L}_{cons}$: trusted-view health consensus", fc="white", ec=TEAL, color=TEAL, fs=6.2)
pill(19.78, 3.47, 3.62, "$\\mathcal{L}_{prior}=KL(\\boldsymbol{w}\\|\\boldsymbol{\\pi})$", fc="white", ec=GREEN, color=GREEN, fs=6.35)

# Loss-to-objective collection paths.
for sx, sy in [(25.48, 3.47), (17.34, 9.25), (21.78, 9.25),
               (21.59, 3.47), (17.34, 2.48), (17.34, 4.83), (13.70, 3.58)]:
    arrow(sx, sy, 17.5, 1.92, color=RED, lw=0.72, style="--", head=6, z=1)

# Global optimizer feedback, visually separated from forward paths.
arrow(30.15, 0.78, 10.45, 5.08, color=RED, lw=1.15, style="--", rad=-0.22, head=7, z=1)
arrow(29.55, 0.78, 21.70, 4.83, color=RED, lw=1.05, style="--", rad=-0.13, head=7, z=1)
ax.text(24.80, 2.26, "back-propagation updates $E_v$, projections, heads and router",
        ha="center", va="center", fontsize=6.45, color=RED, zorder=5)

# Target-free badge in the deployment corner.
pill(26.96, 11.05, 3.95, "TARGET DATA: inference only", fc=GREEN_F, ec=GREEN, color=GREEN, fs=6.8)

fig.subplots_adjust(left=0.008, right=0.992, bottom=0.012, top=0.992)
# The figure is placed at full manuscript width; a modest final type scale keeps
# all labels above the usual 6-pt print threshold while preserving whitespace.
for artist in ax.texts:
    artist.set_fontsize(artist.get_fontsize() * 1.10)
for out in OUT_DIRS:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_framework_p1.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out / "fig_framework_p1.svg", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out / "fig_framework_p1.png", dpi=600, bbox_inches="tight", pad_inches=0.03)

print("saved fig_framework_p1.{pdf,svg,png} to:")
for out in OUT_DIRS:
    print(f"  {out}")
