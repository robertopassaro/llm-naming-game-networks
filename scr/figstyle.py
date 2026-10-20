"""Matplotlib style and colours shared by all figures."""
import os
import matplotlib as mpl

import paths

FIGDIR = paths.FIGURES

# Okabe-Ito colours (colour-blind safe)
TOPO_COLORS = {
    "fc":   "#000000",
    "ucm":  "#0072B2",   # UCM gamma = 2.5
    "ucm5": "#56B4E9",   # UCM gamma = 5.0
    "rrn":  "#D55E00",
}
TOPO_LABELS = {"fc": "FC (well-mixed)", "ucm": r"UCM ($\gamma$=2.5)",
               "ucm5": r"UCM ($\gamma$=5.0)", "rrn": "RRN"}

PLACE_COLORS = {"hub": "#CC79A7", "random": "#666666", "leaf": "#009E73"}
PLACE_LABELS = {"hub": "hub", "random": "random", "leaf": "low-degree nodes"}

DATA_COLOR = "#222222"
THEORY_COLOR = "#D7301F"


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 11,
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "lines.linewidth": 1.8,
    })


def savefig(fig, name):
    """Write <name>.pdf and <name>.png into the figures directory."""
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    return os.path.join(FIGDIR, name)
