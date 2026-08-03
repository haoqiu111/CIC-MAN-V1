# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) source-only prior schematic -> assets/figures/multipanel/ -> param inherit
# (b) six-agent routing schematic -> assets/figures/multipanel/ -> param inherit
# (c) deployment protocol schematic -> assets/figures/multipanel/ -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

# Academic Figure Skill Typography Baseline - COPY VERBATIM, place at TOP of script
import matplotlib as mpl
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

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline - COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [
    ROOT / "paper" / "figures",
    ROOT  / "outputs" / "figures",
]

# Semantic roles from the mandatory palette: blue=mechanism/router,
# orange=counterfactual, green=source boundary/deployment,
# purple=domain branch, red=loss/fault evidence.
INK = BLACK
MID = CATEGORICAL[5]
LINE = GREY
PALE = "#F7F9FB"
BLUE = CATEGORICAL[0]
BLUE_L = "#EAF2FB"
AMBER = CATEGORICAL[3]
AMBER_L = "#FFF2E4"
GREEN = CATEGORICAL[2]
GREEN_L = "#EAF5ED"
TEAL = BLUE
TEAL_L = BLUE_L
VIOLET = CATEGORICAL[4]
VIOLET_L = "#F1EDF8"
RED = ACCENT_RED
RED_L = "#FBEDED"
GRAY_L = "#EEF1F4"

fig, ax = plt.subplots(figsize=(7.20, 4.46))  # 183 x 113 mm
ax.set_xlim(0, 14.6)
ax.set_ylim(0, 9.05)
ax.axis("off")


def rounded(x, y, w, h, *, fc="white", ec=LINE, lw=0.75, r=0.08, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(p)
    return p


def arrow(x1, y1, x2, y2, *, color=INK, lw=0.9, ls="-", rad=0.0,
          head=7.0, z=3):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=head,
        linewidth=lw, linestyle=ls, color=color,
        shrinkA=1.2, shrinkB=1.2, connectionstyle=f"arc3,rad={rad}", zorder=z,
    ))


def label_box(x, y, w, h, title, body="", *, fc="white", ec=LINE,
              tc=INK, tfs=6.7, bfs=5.65, align="center", lw=0.75):
    rounded(x, y, w, h, fc=fc, ec=ec, lw=lw)
    ha = "center" if align == "center" else "left"
    tx = x + w / 2 if align == "center" else x + 0.12
    if body:
        ax.text(tx, y + h - 0.14, title, ha=ha, va="top", fontsize=tfs,
                fontweight="bold", color=tc, zorder=5)
        ax.text(tx, y + (h - 0.26) * 0.43, body, ha=ha, va="center",
                fontsize=bfs, linespacing=1.12, color=INK, zorder=5)
    else:
        ax.text(tx, y + h / 2, title, ha=ha, va="center", fontsize=tfs,
                fontweight="bold", color=tc, zorder=5)


def chip(x, y, w, text, *, fc="white", ec=LINE, color=INK, fs=5.25,
         bold=False):
    rounded(x, y, w, 0.34, fc=fc, ec=ec, lw=0.58, r=0.13, z=5)
    ax.text(x + w / 2, y + 0.17, text, ha="center", va="center", fontsize=fs,
            color=color, fontweight="bold" if bold else "normal", zorder=6)


def panel(x, y, w, h, letter, title, subtitle, accent):
    rounded(x, y, w, h, fc=PALE, ec="#C7CDD3", lw=0.72, r=0.07, z=0)
    ax.add_patch(FancyBboxPatch(
        (x + 0.12, y + h - 0.58), 0.42, 0.36,
        boxstyle="round,pad=0.02,rounding_size=.07",
        facecolor=accent, edgecolor=accent, linewidth=0, zorder=5,
    ))
    ax.text(x + 0.33, y + h - 0.40, letter, ha="center", va="center",
            color="white", fontsize=7.7, fontweight="bold", zorder=6)
    ax.text(x + 0.66, y + h - 0.31, title, ha="left", va="center",
            fontsize=7.25, fontweight="bold", color=INK, zorder=6)
    ax.text(x + 0.66, y + h - 0.54, subtitle, ha="left", va="center",
            fontsize=5.2, color=MID, zorder=6)


def waveform(x, y, w, h, color=BLUE, phase=0.0, z=6):
    xs = np.linspace(x, x + w, 60)
    ys = y + h / 2 + 0.23 * h * np.sin(np.linspace(phase, phase + 7*np.pi, 60))
    ys += 0.12 * h * np.sin(np.linspace(phase, phase + 19*np.pi, 60))
    ax.plot(xs, ys, color=color, lw=0.75, clip_on=False, zorder=z)


def spectrum(x, y, w, h, color=BLUE, z=6):
    pts = [(x, y), (x + 0.11*w, y + 0.10*h), (x + 0.20*w, y + 0.78*h),
           (x + 0.27*w, y + 0.12*h), (x + 0.43*w, y + 0.35*h),
           (x + 0.51*w, y + 0.08*h), (x + 0.66*w, y + 0.62*h),
           (x + 0.73*w, y + 0.10*h), (x + w, y + 0.18*h)]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=0.75, zorder=z)


def heatmap_icon(x, y, w, h, color=BLUE, z=6):
    vals = np.array([[0.2, .65, .3, .8], [.55, .25, .8, .35], [.25, .75, .4, .65]])
    for i in range(3):
        for j in range(4):
            c = mpl.colors.to_rgba(color, 0.18 + 0.55*vals[i, j])
            ax.add_patch(Polygon([
                (x+j*w/4, y+i*h/3), (x+(j+1)*w/4, y+i*h/3),
                (x+(j+1)*w/4, y+(i+1)*h/3), (x+j*w/4, y+(i+1)*h/3)],
                closed=True, facecolor=c, edgecolor="white", linewidth=0.2, zorder=z))


# Top protocol boundary: the single most important review guard.
rounded(0.18, 8.48, 14.24, 0.43, fc="#F2F6F3", ec="#B9CDBF", lw=0.65, r=0.10)
ax.text(0.38, 8.695, "STRICT SOURCE-ONLY BOUNDARY", fontsize=6.25,
        color=GREEN, fontweight="bold", va="center")
ax.text(3.72, 8.695,
        "prior estimation  |  training  |  model selection", fontsize=5.65,
        color=INK, va="center")
ax.text(11.02, 8.695, "TARGET", fontsize=6.25, color=GREEN,
        fontweight="bold", va="center")
ax.text(12.00, 8.695, "inference only", fontsize=5.65, color=INK, va="center")

# Three independent panels.
panel(0.18, 0.66, 3.78, 7.58, "a", "Intervention-consistent prior",
      "offline LOSO probes; target never observed", GREEN)
panel(4.16, 0.66, 6.85, 7.58, "b", "Six-agent causal evidence routing",
      "union training on source windows and HP-800 twins", BLUE)
panel(11.21, 0.66, 3.21, 7.58, "c", "Frozen deployment protocol",
      "last epoch; no target adaptation or selection", GREEN)

# -------------------------------------------------------------------------
# a | Offline prior
# -------------------------------------------------------------------------
label_box(0.48, 6.38, 1.30, 0.86, "Labeled sources",
          r"$\mathcal{S}_1,\ldots,\mathcal{S}_K$" + "\nrecording split",
          fc=GRAY_L, ec=LINE, tfs=6.2, bfs=5.15)

# Fork into identity/counterfactual paths.
label_box(2.08, 6.70, 1.50, 0.72, "Identity cache", "$T_v(x)$",
          fc=BLUE_L, ec=BLUE, tc=BLUE, tfs=6.15, bfs=5.35)
label_box(2.08, 5.78, 1.50, 0.72, "HP-800 cache", "$T_v(T_{hp}(x))$",
          fc=AMBER_L, ec=AMBER, tc=AMBER, tfs=6.15, bfs=5.35)
arrow(1.78, 6.83, 2.08, 7.06, color=BLUE, rad=-0.12)
arrow(1.78, 6.76, 2.08, 6.14, color=AMBER, rad=0.12)

label_box(0.48, 4.38, 3.10, 1.04, "Per-view leave-one-source-out probes",
          r"train on $\mathcal{S}\setminus\mathcal{S}_k$  ->  score $\mathcal{S}_k$" + "\n"
          "internal probability-sum recording macro-F1",
          fc="white", ec="#AAB5BE", tfs=6.25, bfs=5.18)
arrow(2.82, 5.78, 2.48, 5.42, color=AMBER)
arrow(2.82, 6.70, 2.64, 5.42, color=BLUE)

label_box(0.48, 2.92, 3.10, 1.09, "Worst-variant reliability",
          r"$\rho_v=\min_{T\in\{id,hp\}}\;K^{-1}\sum_k\rho_{v,T}^{(k)}$" + "\n"
          "shortcut-sensitive views lose trust",
          fc=AMBER_L, ec=AMBER, tc=AMBER, tfs=6.25, bfs=5.25)
arrow(2.03, 4.38, 2.03, 4.01, color=AMBER)

label_box(0.48, 1.43, 3.10, 1.03, "Frozen source prior",
          r"$\boldsymbol{\pi}=\mathrm{softmax}(\boldsymbol{\rho}/T_\pi)$,  $T_\pi=0.08$" + "\n"
          r"$\log\boldsymbol{\pi}$ anchors Stage b",
          fc=GREEN_L, ec=GREEN, tc=GREEN, tfs=6.35, bfs=5.30)
arrow(2.03, 2.92, 2.03, 2.46, color=GREEN)
chip(0.70, 0.88, 2.64, "source data only  |  one frozen seed-42 prior",
     fc="white", ec=GREEN, color=GREEN, fs=5.0)

# -------------------------------------------------------------------------
# b | Main method (hero)
# -------------------------------------------------------------------------
# Input union (explicitly not paired consistency).
label_box(4.45, 6.34, 1.18, 0.92, "Source batch",
          r"$x,y,m,\phi_{16}$" + "\n sampled from union",
          fc=GRAY_L, ec=LINE, tfs=6.05, bfs=5.1)
chip(4.47, 5.82, 1.14, "union; no paired loss", fc=AMBER_L, ec=AMBER,
     color=AMBER, fs=5.0)

# Six views as an integrated bank, not six repeated box chains.
rounded(5.92, 5.10, 1.74, 2.29, fc=BLUE_L, ec=BLUE, lw=0.72)
ax.text(6.79, 7.18, "Six soft interventions", ha="center", va="center",
        fontsize=6.45, color=BLUE, fontweight="bold")
view_rows = [
    ("raw", "wave"), ("denoise", "wave"), ("env_spec", "spec"),
    ("env_order", "spec"), ("STFT", "map"), ("CWT", "map"),
]
for i, (name, kind) in enumerate(view_rows):
    yy = 6.85 - i * 0.28
    ax.text(6.04, yy, name, fontsize=5.0, ha="left", va="center", color=INK)
    if kind == "wave":
        waveform(6.68, yy-0.08, 0.72, 0.16, BLUE, phase=i)
    elif kind == "spec":
        spectrum(6.68, yy-0.07, 0.72, 0.15, BLUE)
    else:
        heatmap_icon(6.75, yy-0.08, 0.60, 0.16, BLUE)

label_box(7.97, 5.72, 1.17, 1.34, "Agent encoders",
          "$E_v$ matched to data\n" + r"$f_v\in\mathbb{R}^{128}$",
          fc=BLUE_L, ec=BLUE, tc=BLUE, tfs=6.15, bfs=5.15)

# Representation split with a strong visual bifurcation.
label_box(9.46, 6.48, 1.18, 0.82, "Health $z_v^h$",
          "64-D fault evidence", fc=RED_L, ec=RED, tc=RED,
          tfs=6.0, bfs=5.0)
label_box(9.46, 5.30, 1.18, 0.82, "Domain $z_v^d$",
          "32-D rig evidence", fc=VIOLET_L, ec=VIOLET, tc=VIOLET,
          tfs=6.0, bfs=5.0)

arrow(5.63, 6.78, 5.92, 6.35, color=BLUE)
arrow(7.66, 6.25, 7.97, 6.39, color=BLUE)
arrow(9.14, 6.50, 9.46, 6.89, color=RED)
arrow(9.14, 6.25, 9.46, 5.71, color=VIOLET)

# Domain supervision is supporting, visually quieter.
label_box(9.53, 4.30, 1.04, 0.72, "Rig heads",
          r"$D(z^d)\to m$" + "\n" + r"$A(GRL(\bar z^h))\to m$",
          fc=VIOLET_L, ec=VIOLET, tc=VIOLET, tfs=5.7, bfs=5.0)
arrow(10.05, 5.30, 10.05, 5.02, color=VIOLET, lw=0.7)

# Shared classifier and router form the central decision diamond.
label_box(7.73, 4.45, 1.42, 0.96, "Shared classifier",
          "$o_v=C(z_v^h)$\nper-view opinion",
          fc=RED_L, ec=RED, tc=RED, tfs=6.05, bfs=5.0)
arrow(9.46, 6.89, 8.61, 5.41, color=RED, rad=0.16)

label_box(5.90, 2.33, 3.40, 1.63, "Evidence router $R$",
          r"$c_v=\|z_v^h-\bar z^h\|_2$    $e_v=H[\mathrm{softmax}(o_v)]$" + "\n"
          r"$r=R([\phi;c;e])+\log\pi$     $w=\mathrm{softmax}(r)$" + "\n"
          r"$\mathcal{L}_{prior}=\mathbb{E}_i KL(w_i\|\pi)$  |  view dropout 0.15",
          fc=TEAL_L, ec=TEAL, tc=TEAL, tfs=6.45, bfs=5.15)
chip(6.13, 4.08, 2.92, r"$\mathcal{L}_{cons}$: router-weighted health consensus",
     fc="white", ec=TEAL, color=TEAL, fs=5.0)
arrow(8.44, 4.45, 8.44, 3.96, color=TEAL)
arrow(9.70, 6.48, 8.84, 3.96, color=TEAL, lw=0.72, rad=0.17)
arrow(5.05, 6.34, 6.18, 3.96, color=LINE, lw=0.7, rad=-0.13)

# Frozen prior crosses panel boundary clearly and only once.
arrow(3.58, 1.95, 5.90, 3.05, color=GREEN, lw=1.0, ls="--", rad=-0.18)
ax.text(4.86, 2.47, r"frozen $\log\pi$", fontsize=5.2, color=GREEN,
        fontweight="bold", ha="center", va="center")

label_box(9.65, 2.43, 1.05, 1.43, "Fusion",
          r"$o=\sum_v w_vo_v$" + "\n\n" + r"$\hat y=\arg\max$",
          fc=RED_L, ec=RED, tc=RED, tfs=6.3, bfs=5.2)
arrow(9.30, 3.15, 9.65, 3.15, color=TEAL)
arrow(9.15, 4.79, 10.18, 3.86, color=RED, rad=-0.12)

# Objective rail: exact default configuration, no inactive balance/DG term.
rounded(4.47, 0.93, 6.24, 0.88, fc="white", ec="#D8A0A0", lw=0.70, r=0.06)
ax.text(4.65, 1.56, "END-TO-END SOURCE TRAINING", fontsize=5.45,
        color=RED, fontweight="bold", ha="left", va="center")
ax.text(7.60, 1.27,
        r"$\mathcal{L}_{cls}+0.3\mathcal{L}_{view}+0.1\mathcal{L}_{cons}"
        r"+0.1\mathcal{L}_{adv}+0.1\mathcal{L}_{dom}+0.05\mathcal{L}_{orth}"
        r"+1.0\mathcal{L}_{prior}$",
        fontsize=5.05, ha="center", va="center")
ax.text(10.52, 1.56, "AdamW | batch 256 | 40 epochs", fontsize=5.0,
        color=MID, ha="right", va="center")
ax.plot([5.16, 10.16], [2.03, 2.03], color=RED, lw=0.65, ls="--", zorder=1)
for xx, yy in [(5.16, 5.82), (8.44, 2.33), (10.16, 2.43)]:
    arrow(xx, 2.03, xx, yy, color=RED, lw=0.65, ls="--", head=5.5, z=1)

# -------------------------------------------------------------------------
# c | Deployment protocol
# -------------------------------------------------------------------------
label_box(11.54, 6.32, 2.54, 0.88, "Unseen target recordings",
          "windows only; labels never used",
          fc=GREEN_L, ec=GREEN, tc=GREEN, tfs=6.25, bfs=5.15)
chip(11.78, 5.90, 2.06, "frozen last-epoch CIC-MAN", fc="white",
     ec=GREEN, color=GREEN, fs=5.0, bold=True)

# Window stream and probability cards.
for i, yy in enumerate([5.36, 4.95, 4.54]):
    rounded(11.58, yy, 0.70, 0.28, fc=GRAY_L, ec=LINE, lw=0.5, r=0.04)
    waveform(11.65, yy+0.04, 0.55, 0.18, BLUE, phase=i)
    rounded(12.58, yy, 1.25, 0.28, fc="white", ec="#B8C0C7", lw=0.5, r=0.04)
    vals = [[.72,.18,.10],[.31,.55,.14],[.42,.43,.15]][i]
    ax.text(13.205, yy+0.14, "p=" + "/".join(f"{v:.2f}" for v in vals),
            ha="center", va="center", fontsize=5.0, color=INK)
    arrow(12.28, yy+0.14, 12.58, yy+0.14, color=GREEN, lw=0.65, head=5.5)
ax.text(11.90, 5.69, "windows", fontsize=5.0, color=MID, ha="center")
ax.text(13.20, 5.69, "frozen probabilities", fontsize=5.0, color=MID, ha="center")

label_box(11.54, 2.72, 2.54, 1.18, "Official recording decision",
          "1  majority vote over windows\n2  tied classes: cumulative probability\n3  remaining tie: smallest class id",
          fc="white", ec=GREEN, tc=GREEN, tfs=6.15, bfs=5.0, align="left")
arrow(12.81, 4.54, 12.81, 3.90, color=GREEN)

label_box(11.54, 1.42, 2.54, 0.78, "3-class diagnosis",
          "normal  |  inner race  |  outer race",
          fc=GREEN_L, ec=GREEN, tc=GREEN, tfs=6.35, bfs=5.0)
arrow(12.81, 2.72, 12.81, 2.20, color=GREEN)

chip(11.66, 0.92, 2.30, "no adaptation  |  no target selection",
     fc="white", ec=GREEN, color=GREEN, fs=5.0)

# Minimal visual key.
ax.text(0.25, 0.32,
        "solid arrows: forward flow     dashed green: frozen prior     dashed red: optimization feedback",
        fontsize=5.0, color=MID, ha="left", va="center")
ax.text(14.32, 0.32, "all target metrics: recording macro-F1",
        fontsize=5.0, color=MID, ha="right", va="center")

fig.subplots_adjust(left=0.004, right=0.996, bottom=0.004, top=0.996)

for out in OUT_DIRS:
    out.mkdir(parents=True, exist_ok=True)
    base = out / "fig_framework_p1_submission"
    save_cns_figure(fig, str(base))
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pad_inches=0.025, pil_kwargs={"compression": "tiff_lzw"})
    # Canonical manuscript aliases.
    alias = out / "fig_framework_p1"
    fig.savefig(alias.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(alias.with_suffix(".png"), dpi=600, bbox_inches="tight",
                pad_inches=0.025, facecolor="white", transparent=False)
    fig.savefig(alias.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(alias.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pad_inches=0.025, facecolor="white", transparent=False,
                pil_kwargs={"compression": "tiff_lzw"})
    # Nature-family raster delivery is opaque RGB rather than an RGBA canvas.
    for raster in (alias.with_suffix(".png"), alias.with_suffix(".tiff"),
                   base.with_suffix(".tiff")):
        with Image.open(raster) as im:
            dpi = im.info.get("dpi", (600, 600))
            rgb = im.convert("RGB")
            kwargs = {"dpi": dpi}
            if raster.suffix.lower() in {".tif", ".tiff"}:
                kwargs["compression"] = "tiff_lzw"
            rgb.save(raster, **kwargs)

qa = {
    "core_conclusion": (
        "A source-only worst-variant reliability prior anchors six-agent evidence "
        "routing and frozen target inference without target adaptation or selection."
    ),
    "archetype": "schematic-led",
    "backend": "python/matplotlib",
    "final_size_mm": [183.0, 113.0],
    "panel_map": {
        "a": "offline intervention-consistent reliability prior",
        "b": "six-agent health/domain routing and exact training objective",
        "c": "last-epoch target inference and official recording aggregation",
    },
    "review_guards": [
        "target is inference only",
        "HP-800 data enter as a union and not through a paired loss",
        "16-D metadata enter only the router",
        "Stage-1 probe uses internal probability-sum aggregation",
        "official output uses majority with deterministic probability/id tie-break",
        "only active default v6ic objective terms are displayed",
    ],
    "minimum_nominal_font_pt": 5.0,
    "primary_editable_format": "SVG",
    "statistics_reproducibility": {
        "quantitative_panels": "none; this is a method/protocol schematic",
        "training_split": "two source datasets; recording-level 80/20 source train/validation",
        "test_split": "third dataset held out and unseen until final inference",
        "seeds": "five target-evaluation seeds; Stage-1 prior frozen from seed 42 probes",
        "metric": "recording macro-F1 after official majority/probability/id tie-break",
        "checkpoint": "last epoch (40); no target-based selection",
    },
}
(ROOT  / "outputs" / "figures" /
 "fig_framework_p1_submission_qa.json").write_text(
    json.dumps(qa, indent=2), encoding="utf-8"
)

plt.close(fig)
print("saved submission-grade CIC-MAN framework (SVG/PDF/TIFF/PNG)")
