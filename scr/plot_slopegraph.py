"""Convergence time of every (model, word pair) cell on FC, UCM 2.5, UCM 5.0
and RRN, one panel per model (Figure 1 and Figure A1).

Reads tables/friedman_cells.csv, written by analyze_friedman.py.

    python plot_slopegraph.py
"""
import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs
import paths

HERE = paths.ROOT
CELLS = os.path.join(paths.TABLES, "friedman_cells.csv")

MODELS = ["llama31_70B", "gpt-4o", "QWQ_32B", "phi_4"]
MODEL_LABELS = {"llama31_70B": "Llama-3.1-70B", "gpt-4o": "GPT-4o",
                "QWQ_32B": "QwQ-32B", "phi_4": "Phi-4"}
TOPOS = ["fc", "ucm_2.5", "ucm_5.0", "rrn"]
TOPO_TICKS = ["FC", r"UCM$_{\gamma=2.5}$", r"UCM$_{\gamma=5.0}$", "RRN"]

# Word pairs in CrowS-Pairs ID order; a pair keeps its style in every figure.
PAIR_ORDER = ["African/White", "gay/straight", "his/her", "young/old",
              "White/Black", "Mexican/American", "she/he", "wife/husband",
              "woman/man", "tall/short", "more/less"]

# 11 pairs: six colour-blind safe hues, each with a filled circle or an open square
HUES = ["#009E73", "#0072B2", "#CC79A7", "#882255", "#B8860B", "#8C510A"]
MARKERS = ["o", "s"]


def pair_style(words):
    """(color, marker, filled) for a word pair."""
    i = PAIR_ORDER.index(words)
    return HUES[i % len(HUES)], MARKERS[i // len(HUES)], i < len(HUES)


def load_cells():
    with open(CELLS) as fh:
        return list(csv.DictReader(fh))


def panel_data(rows, rule, N, model, stat):
    """{word pair: np.array of 4 values in TOPOS order} for one panel."""
    sub = [r for r in rows
           if r["rule"] == rule and int(r["N"]) == N and r["model"] == model]
    by_id = {}
    for r in sub:
        by_id.setdefault(r["words"], {})[r["topo"]] = float(r[stat])
    return {w: np.array([v[t] for t in TOPOS]) for w, v in by_id.items()}


def draw_panel(ax, cells):
    x = np.arange(len(TOPOS))
    for words in PAIR_ORDER:
        if words not in cells:
            continue                          # pair excluded for this model
        color, marker, filled = pair_style(words)
        ax.plot(x, cells[words], color=color, lw=1.3, alpha=0.9, zorder=3,
                marker=marker, markersize=3.6, markeredgewidth=0.9,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color, solid_capstyle="round")

    ax.set_xticks(x)
    ax.set_xticklabels(TOPO_TICKS, fontsize=9)
    ax.set_xlim(-0.35, len(TOPOS) - 0.65)
    ax.set_yscale("log")
    ax.grid(axis="y", which="major", color="#DDDDDD", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)


def build(rows, rule, N, stat, name):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), sharey=True)
    for ax, model in zip(axes.ravel(), MODELS):
        draw_panel(ax, panel_data(rows, rule, N, model, stat))
        ax.set_title(MODEL_LABELS[model], fontsize=12)
    for r in (0, 1):
        axes[r, 0].set_ylabel("population round", fontsize=9.5)

    handles = []
    for words in PAIR_ORDER:
        color, marker, filled = pair_style(words)
        handles.append(Line2D([], [], color=color, lw=1.6, marker=marker,
                              markersize=4.4, markeredgewidth=0.9,
                              markerfacecolor=color if filled else "white",
                              markeredgecolor=color, label=words))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 0.005), fontsize=8.5,
               handlelength=2.6, columnspacing=1.4, labelspacing=0.55)
    fig.suptitle(f"Convergence time by topology "
                 f"({stat} per cell, {rule} update, $N$={N:,})",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0.155, 1, 0.96))
    fs.savefig(fig, name)
    plt.close(fig)
    return os.path.join(fs.FIGDIR, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="edge", choices=["edge", "node"])
    ap.add_argument("--N", type=int, default=10000)
    args = ap.parse_args()

    fs.apply_style()
    rows = load_cells()

    jobs = [(args.rule, args.N, stat, f"fig_slopegraph_{stat}_{args.rule}_N{args.N}")
            for stat in ("mode", "mean")]
    for rule, N, stat, name in jobs:
        print(f"{stat}/{rule}/N={N}  ->  "
              f"{os.path.relpath(build(rows, rule, N, stat, name), HERE)}")


if __name__ == "__main__":
    main()
