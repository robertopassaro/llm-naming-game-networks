"""Critical committed fraction p* at N = 10000 (Figure 3).

  fig3a_topology   UCM 2.5, UCM 5.0, RRN and FC, random placement
  fig3b_placement  hub, random and low-degree placement on UCM 2.5

The strong convention of a pair is the word the population converges to more
often on FC at N = 100000, without committed agents. The "strong" lane shows
the p* needed to overturn it, the "weak" lane the p* needed to overturn the
other word. Conditions with p* = 0 (the convention does not hold even without
committed agents) are not drawn; they are counted in the printed summary.

    python plot_pstar.py
    python plot_pstar.py --xmax 0.6 --layout offset
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

import figstyle as fs
import paths

RESULTS = paths.GRID

MODELS = ["llama31_70B", "gpt-4o", "QWQ_32B", "phi_4"]
MODEL_LABELS = {"llama31_70B": "Llama-3.1-70B", "gpt-4o": "GPT-4o",
                "QWQ_32B": "QwQ-32B", "phi_4": "Phi-4"}
IDS = [92, 102, 191, 219, 344, 349, 397, 410, 505, 626, 1161]
N_FIX = 10000

# panel -> [(topo, placement, label, short, color, marker), ...]
PANELS = {
    "a": ("fig3a_topology", "Topology (random placement)", [
        ("ucm_2.5", "random", r"UCM ($\gamma$=2.5)", "ucm_2.5", fs.TOPO_COLORS["ucm"], "o"),
        ("ucm_5.0", "random", r"UCM ($\gamma$=5.0)", "ucm_5.0", fs.TOPO_COLORS["ucm5"], "D"),
        ("rrn",     "random", "RRN",              "rrn", fs.TOPO_COLORS["rrn"], "^"),
        ("fc",      "random", "FC (well-mixed)",  "fc",  fs.TOPO_COLORS["fc"],  "s"),
    ]),
    "b": ("fig3b_placement", r"Placement on UCM ($\gamma$=2.5)", [
        ("ucm_2.5", "hub",    "hub",    "hub",    fs.PLACE_COLORS["hub"],    "^"),
        ("ucm_2.5", "random", "random", "random", fs.PLACE_COLORS["random"], "o"),
        ("ucm_2.5", "leaf",   "low-degree nodes", "leaf",   fs.PLACE_COLORS["leaf"],   "s"),
    ]),
}
DIRECTIONS = ["strong", "weak"]
# (row gap, block gap, figure height) per layout
GEOM = {"shared": (0.55, 1.15, 5.4), "offset": (1.0, 2.0, 7.5)}
OFFSET = 0.25        # between conditions inside a sub-lane (offset only)


def load(model, ID, topo, placement, w):
    p = os.path.join(RESULTS, f"RQ2_{model}_ID_{ID}_{topo}_N{N_FIX}_"
                              f"{placement}_dir{w}_edge_H5.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def direction_labels():
    """(model, ID) -> {'strong': w, 'weak': w'}: the committed word of the RQ2
    file to load for each lane.

    The strong word comes from the FC consensus counts at N = 100000. The two
    Llama pairs that never converge on FC use the majority word at t_max
    instead. An RQ2 file dir{w} starts from consensus on the other word, so the
    "strong" lane loads the file committed to the weak word, and vice versa.
    """
    out = {}
    for model in MODELS:
        for ID in IDS:
            p = os.path.join(RESULTS, f"RQ1_{model}_ID_{ID}_fc_N100000_edge_H5.pkl")
            with open(p, "rb") as f:
                d = pickle.load(f)
            n0, n1 = d["consensus"][0], d["consensus"][1]
            if n0 == n1 == 0:
                n0 = d["non_consensus"][0]["count"]
                n1 = d["non_consensus"][1]["count"]
            strong_word = 1 if n1 >= n0 else 0   # ties go to word 1
            weak_word = 1 - strong_word
            out[(model, ID)] = {"strong": weak_word, "weak": strong_word}
    return out


def track(model, topo, placement, w_of, direction):
    """One lane: (p* values > 0, IDs with p* = 0, IDs with p* None)."""
    vals, spontaneous, never = [], [], []
    for ID in IDS:
        d = load(model, ID, topo, placement, w_of[(model, ID)][direction])
        if d is None:
            continue
        p = d["p_star"]
        if p is None:
            never.append(ID)
        elif p == 0.0:
            spontaneous.append(ID)
        else:
            vals.append(p)
    return np.array(vals), spontaneous, never


def plot(panel, xmax, layout):
    name, subtitle, conds = PANELS[panel]
    w_of = direction_labels()
    fs.apply_style()
    ROW_GAP, BLOCK_GAP, height = GEOM[layout]
    fig, ax = plt.subplots(figsize=(8.5, height))

    yticks, ylabels, rows = [], [], []
    for m, model in enumerate(MODELS):
        for r, direction in enumerate(DIRECTIONS):
            base = m * BLOCK_GAP + r * ROW_GAP
            yticks.append(base)
            ylabels.append(direction)
            for c, (topo, placement, lab, _s, color, mk) in enumerate(conds):
                # offset layout: one sub-track per condition
                mid = (len(conds) - 1) / 2
                y = base + ((c - mid) * OFFSET if layout == "offset" else 0.0)
                vals, spont, never = track(model, topo, placement, w_of, direction)
                rows.append((model, direction, lab, vals, spont, never))

                if len(vals):
                    ax.plot(vals, np.full(len(vals), y), marker=mk, ms=3, lw=0,
                            mfc=color, mec=color, alpha=0.3, zorder=3)
                    med = np.median(vals)
                    if med <= xmax:
                        ax.plot([med], [y], marker=mk, ms=9, lw=0, mfc=color,
                                mec="white", mew=0.8, zorder=5)
                    else:                       # off the axis: arrow at the edge
                        ax.plot([xmax], [y], marker=">", ms=8, lw=0, mfc=color,
                                mec=color, clip_on=False, zorder=5)

        if m % 2:                  # shaded band for every other model
            half = BLOCK_GAP / 2
            ax.axhspan(m * BLOCK_GAP + ROW_GAP / 2 - half,
                       m * BLOCK_GAP + ROW_GAP / 2 + half,
                       color="0.945", lw=0, zorder=0)
        ax.text(-0.11, m * BLOCK_GAP + ROW_GAP / 2, MODEL_LABELS[model],
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=11, rotation=90)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9.5)
    ax.tick_params(axis="y", which="both", length=0, pad=6)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    pad = 0.45 * ROW_GAP + (OFFSET if layout == "offset" else 0)
    ax.set_ylim(-pad, (len(MODELS) - 1) * BLOCK_GAP + ROW_GAP + pad)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel(r"critical committed fraction  $p^*$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="0.88", lw=0.6, zorder=0)

    handles = [Line2D([0], [0], color=c, marker=mk, ms=8, lw=0, label=lab)
               for _t, _p, lab, _s, c, mk in conds]
    fig.legend(handles=handles, loc="upper center", ncol=len(conds),
               bbox_to_anchor=(0.5, 1.045), frameon=False)

    key = [Line2D([0], [0], color="0.35", marker="o", ms=8, lw=0, mec="white",
                  mew=0.8, label="median over the 11 word pairs"),
           Line2D([0], [0], color="0.35", marker="o", ms=3, lw=0, alpha=0.5,
                  label="one word pair")]
    fig.legend(handles=key, loc="upper center", ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.985), frameon=False)
    ax.set_title(subtitle, pad=8)
    # fixed margins: tight_layout does not leave room for the model names
    fig.subplots_adjust(left=0.20, right=0.98, top=0.86, bottom=0.10)
    path = fs.savefig(fig, name)
    plt.close(fig)
    print(f"wrote {path}.{{pdf,png}}")
    summarize(rows, conds)


def summarize(rows, conds):
    print("\n--- per-lane median p* ---")
    print(f"{'model':<14s}{'dir':<8s}"
          + "".join(f"{sh:>16s}" for _t, _p, _l, sh, *_ in conds))
    by = {(m, d): {} for m in MODELS for d in DIRECTIONS}
    for model, direction, lab, vals, spont, never in rows:
        by[(model, direction)][lab] = (vals, spont, never)
    for model in MODELS:
        for direction in DIRECTIONS:
            cells = []
            for _t, _p, lab, *_ in conds:
                vals, spont, _ = by[(model, direction)][lab]
                s = f"{np.median(vals):.3f}" if len(vals) else "  -  "
                cells.append(s + (f" +{len(spont)}@0" if spont else ""))
            print(f"{MODEL_LABELS[model]:<14s}{direction:<8s}"
                  + "".join(f"{c:>16s}" for c in cells))
    n0 = sum(len(s) for *_r, s, _n in rows)
    nn = sum(len(n) for *_r, _s, n in rows)
    print(f"\np*==0 (spontaneous) lane-entries: {n0};  p* is None (never flips): {nn}")


def pooled_table():
    """Mean and median p* per (topo, placement), both directions pooled."""
    print("\n--- pooled over both directions and 11 pairs ---")
    print(f"{'condition':<22s}{'mean':>8s}{'median':>9s}")
    got = {}
    for topo, places in [("ucm_2.5", ["hub", "random", "leaf"]),
                         ("ucm_5.0", ["hub", "random", "leaf"]),
                         ("rrn", ["random"]), ("fc", ["random"])]:
        for pl in places:
            v = [d["p_star"] for m in MODELS for ID in IDS for w in (0, 1)
                 if (d := load(m, ID, topo, pl, w)) and d["p_star"] is not None]
            v = np.array(v)
            got[(topo, pl)] = v.mean()
            print(f"{topo + ' ' + pl:<22s}{v.mean():>8.3f}{np.median(v):>9.3f}")
    for t in ("ucm_2.5", "ucm_5.0"):
        ok = got[(t, "hub")] <= got[(t, "random")] <= got[(t, "leaf")]
        print(f"  hub <= random <= leaf on {t}: {ok}")
    ok = got[("rrn", "random")] < got[("ucm_2.5", "random")] < got[("fc", "random")]
    print(f"  rrn < ucm_2.5 < fc at random placement: {ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["a", "b", "both"], default="both")
    ap.add_argument("--xmax", type=float, default=1.0,
                    help="x clip; medians beyond it are drawn as edge arrows")
    ap.add_argument("--layout", choices=["shared", "offset"], default="shared",
                    help="shared: the 3 conditions share one line per sub-lane")
    a = ap.parse_args()
    for panel in (["a", "b"] if a.panel == "both" else [a.panel]):
        plot(panel, a.xmax, a.layout)
    pooled_table()


if __name__ == "__main__":
    main()
