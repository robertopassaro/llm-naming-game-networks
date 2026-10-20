"""
Convergence time on RRN against the degree K at N = 10000, relative to FC,
one panel per model and one line per word pair (Figure A2).

Reads tables/density_cells.csv, written by `python analyze_friedman.py density`.

    python plot_density.py [--stat mode|mean|both]
"""
import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs
import paths
from plot_slopegraph import PAIR_ORDER, pair_style, MODELS, MODEL_LABELS

CELLS = os.path.join(paths.TABLES, "density_cells.csv")
NAME = "fig_density_rrn"


def load(stat="mean"):
    """(Ks ascending, {model: {word pair: T(K)/T_FC}})."""
    rows = list(csv.DictReader(open(CELLS)))
    Ks = sorted({int(r["topo"].split("_k")[1])
                 for r in rows if r["topo"].startswith("rrn_k")})
    by = {}
    for r in rows:
        by.setdefault((r["model"], r["words"]), {})[r["topo"]] = float(r[stat])
    out = {}
    for (model, words), d in by.items():
        fc = d["fc"]
        out.setdefault(model, {})[words] = np.array(
            [d[f"rrn_k{K}"] / fc for K in Ks])
    return np.array(Ks, float), out


def draw_panel(ax, cells, Ks):
    for words in PAIR_ORDER:
        if words not in cells:
            continue                          # pair excluded for this model
        color, marker, filled = pair_style(words)
        ax.plot(Ks, cells[words], color=color, lw=1.3, alpha=0.9, zorder=3,
                marker=marker, markersize=3.6, markeredgewidth=0.9,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color, solid_capstyle="round")

    ax.axhline(1.0, color="#000000", ls="--", lw=1.2, zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([f"{int(k)}" for k in Ks], fontsize=7.5)
    ax.set_yticks([1, 1.5, 2, 3, 5, 10])
    ax.set_yticklabels(["1", "1.5", "2", "3", "5", "10"], fontsize=9)
    ax.grid(axis="y", which="major", color="#DDDDDD", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="minor", bottom=False, top=False,
                   left=False, right=False)


def build(stat):
    Ks, data = load(stat)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4), sharey=True, sharex=True)
    for ax, model in zip(axes.ravel(), MODELS):
        draw_panel(ax, data[model], Ks)
        n = len(data[model])
        ax.set_title(f"{MODEL_LABELS[model]}  ({n} word pairs)", fontsize=12)
    for r in (0, 1):
        axes[r, 0].set_ylabel(r"$T(K)\,/\,T_{\mathrm{FC}}$", fontsize=10)
    for c in (0, 1):
        axes[1, c].set_xlabel(r"mean degree $K$", fontsize=10)

    # label the FC line once, above it, on the first panel
    axes[0, 0].annotate("FC (well-mixed)", xy=(Ks[-1], 1.0), xytext=(-4, 5),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=8.5, color="#000000")

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
    fig.suptitle("Convergence time vs edge density on RRN "
                 f"({stat} per cell, edge update, " r"$N$=10,000)",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0.145, 1, 0.96))
    name = f"{NAME}_{stat}"
    fs.savefig(fig, name)
    plt.close(fig)
    print(f"wrote {os.path.join(fs.FIGDIR, name)}.pdf / .png")
    for model in MODELS:
        d = data[model]
        lo = min(d.items(), key=lambda kv: kv[1][0])
        hi = max(d.items(), key=lambda kv: kv[1][0])
        print(f"  {MODEL_LABELS[model]:14s} {len(d):2d} pairs   "
              f"T(K=4)/T_FC {lo[1][0]:5.2f} ({lo[0]}) .. "
              f"{hi[1][0]:5.2f} ({hi[0]})")
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", choices=["mean", "mode", "both"], default="mode")
    args = ap.parse_args()
    fs.apply_style()
    for stat in (["mean", "mode"] if args.stat == "both" else [args.stat]):
        build(stat)


if __name__ == "__main__":
    main()
