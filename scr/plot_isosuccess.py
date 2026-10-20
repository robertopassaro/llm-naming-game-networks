"""Iso-success contours by degree on UCM 2.5 at N = 10000, one panel per word
pair and one figure per model (Figure 2 and Figures A5-A7).

For each degree bin, S_k is the cumulative success rate of the interactions its
nodes took part in, from counters summed over all runs. t* is the round at
which S_k first reaches a level (0.6, 0.7, 0.8), interpolated in log t, and
n* = 2 kbar t* / <k> the mean number of interactions of a node of the bin by
then. The tilt is t*(lowest bin) / t*(highest bin) and
tau = ln(tilt) / ln(k_hub / k_low): 0 for a vertical contour (every bin at the
same round), 1 for a horizontal one (every bin after the same number of own
interactions). Quenched (edge rule) and annealed runs are shown together.

Writes tables/isosuccess_cells.csv (t*, n* per bin) and
tables/isosuccess_contours.csv (tilt, tau per contour).

    python plot_isosuccess.py
"""
import argparse
import csv
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs
import paths
from plot_slopegraph import PAIR_ORDER

RESULTS = paths.GRID

MODELS = ["llama31_70B", "gpt-4o", "QWQ_32B", "phi_4"]
MODEL_LABELS = {"llama31_70B": "Llama-3.1-70B", "gpt-4o": "GPT-4o",
                "QWQ_32B": "QwQ-32B", "phi_4": "Phi-4"}
IDS = [92, 102, 191, 219, 344, 349, 397, 410, 505, 626, 1161]

# Degree bins [a, b): 4-5, 6-9, 10-17, 18-35, 36-100
EDGES = [4, 6, 10, 18, 36, 101]
LEVELS = [0.6, 0.7, 0.8]
CONTOUR_COLOR = "#B2182B"

# (rule, label, colour, marker, linestyle)
ARMS = [("edge", "quenched", CONTOUR_COLOR, "o", "-"),
        ("annealed", "annealed", "#2166AC", "s", "--")]

LEVEL_COLORS = {0.6: "#F4A582", 0.7: "#D6604D", 0.8: CONTOUR_COLOR}
LEVEL_COLORS_ANN = {0.6: "#92C5DE", 0.7: "#4393C3", 0.8: "#2166AC"}

WORDS = dict(zip(IDS, PAIR_ORDER))

GRID_COLOR = "#BBBBBB"
NCOL = 4

# the two Llama pairs that do not converge
EXCLUDE = {("llama31_70B", 219), ("llama31_70B", 1161)}

MIN_SPAN = 1.6                # smallest panel window, decades
PAD = 0.18                    # margin around the contour, decades


def load(model, ID, topo, N, rule):
    p = os.path.join(RESULTS, f"RQ1_{model}_ID_{ID}_{topo}_N{N}_{rule}_H5.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def bins_of(deg_hist, edges):
    """[(lo, hi_inclusive, mask)] for the bins that contain nodes."""
    kk = np.arange(len(deg_hist))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (kk >= a) & (kk < b) & (deg_hist > 0)
        if m.any():
            out.append((a, min(b - 1, int(kk[m].max())), m))
    return out


def binned(d, edges):
    """(labels, S[nbin, nsnap], n[nbin, nsnap]).

    S is the cumulative success rate over all interactions of the bin's nodes,
    n the cumulative number of interactions per node of the bin.
    """
    ss = d["succ_snap_sum"].astype(float)
    ts = d["tot_snap_sum"].astype(float)
    dh = d["deg_hist_sum"].astype(float)
    labels, S, n = [], [], []
    for lo, hi, m in bins_of(dh, edges):
        labels.append(f"$k$ = {lo}" if lo == hi else f"$k$ = {lo}–{hi}")
        S.append(ss[:, m].sum(1) / ts[:, m].sum(1))
        n.append(ts[:, m].sum(1) / dh[m].sum())
    return labels, np.array(S), np.array(n)


def s0_of(model, ID):
    """Success rate at t = 0, q^2 + (1-q)^2 with q the empty-memory policy.

    The same for every bin. It is prepended to the snapshots because some bins
    are already above 0.6 at the first snapshot (t = 0.1 rounds).
    """
    with open(paths.q_dict_path(model, ID), "rb") as f:
        q = pickle.load(f)[""]
    return q * q + (1.0 - q) * (1.0 - q)


def crossings(S, n, rounds, level, s0=None):
    """(t*, n*) at which each bin's S first reaches `level`, interpolated in
    log-log; NaN for bins that never reach it. `s0` is placed at t = rounds[0]/10."""
    if s0 is not None:
        t0 = rounds[0] / 10.0
        S = np.column_stack([np.full(len(S), float(s0)), S])
        n = np.column_stack([n[:, 0] * (t0 / rounds[0]), n])
        rounds = np.concatenate([[t0], rounds])
    t_star = np.full(len(S), np.nan)
    n_star = np.full(len(S), np.nan)
    for b in range(len(S)):
        if S[b, 0] < level <= S[b, -1]:
            t_star[b] = np.exp(np.interp(level, S[b], np.log(rounds)))
            n_star[b] = np.exp(np.interp(level, S[b], np.log(n[b])))
    return t_star, n_star


def fit_loglog(t, n, npts=50):
    """Reduced-major-axis line through the contour points in log-log space.

    Returns (x, y, slope, r), or None with fewer than two usable points. The
    segment spans the projections of the points onto the line.
    """
    t = np.asarray(t, dtype=float)
    n = np.asarray(n, dtype=float)
    m = np.isfinite(t) & np.isfinite(n) & (t > 0) & (n > 0)
    if m.sum() < 2:
        return None
    lt, ln = np.log10(t[m]), np.log10(n[m])
    sd_t, sd_n = lt.std(), ln.std()
    if sd_t == 0 or sd_n == 0:
        return None
    r = float(np.corrcoef(lt, ln)[0, 1])
    slope = np.sign(r) * sd_n / sd_t
    mt, mn = lt.mean(), ln.mean()
    ux, uy = 1.0 / np.hypot(1.0, slope), slope / np.hypot(1.0, slope)
    s = (lt - mt) * ux + (ln - mn) * uy
    ss = np.linspace(s.min(), s.max(), npts)
    return (10.0 ** (mt + ss * ux), 10.0 ** (mn + ss * uy), float(slope), r)


def kbars(d, edges):
    """Node-weighted mean degree of each bin."""
    dh = d["deg_hist_sum"].astype(float)
    kk = np.arange(len(dh))
    return np.array([(dh[m] * kk[m]).sum() / dh[m].sum()
                     for _, _, m in bins_of(dh, edges)])


def mean_degree(d):
    dh = d["deg_hist_sum"].astype(float)
    return float((dh * np.arange(len(dh))).sum() / dh.sum())


def contours(model, ID, topo, N, edges, levels):
    """dict(kbar, kmean, labels, arms={rule: {level: (t*, n*)}}), or None."""
    arms, kbar, kmean, labels = {}, None, None, None
    for rule, _, _, _, _ in ARMS:
        d = load(model, ID, topo, N, rule)
        if d is None:
            continue
        labels, S, n = binned(d, edges)
        kbar, kmean = kbars(d, edges), mean_degree(d)
        rounds = np.array(d["snap_rounds"], dtype=float)
        s0 = s0_of(model, ID)
        arms[rule] = {L: crossings(S, n, rounds, L, s0) for L in levels}
    if not arms:
        return None
    return dict(kbar=kbar, kmean=kmean, labels=labels, arms=arms)


def has_data(c):
    return any(np.isfinite(t).any()
               for lv in c["arms"].values() for t, _ in lv.values())


def tilt_of(t):
    """t*(lowest bin) / t*(highest bin); > 1 when hubs get there first."""
    ok = np.nonzero(np.isfinite(t))[0]
    return t[ok[0]] / t[ok[-1]] if len(ok) >= 2 else np.nan


def tau_of(t, kbar):
    """ln(tilt) / ln(k_hub / k_low); NaN unless every bin crossed."""
    if not np.isfinite(t).all():
        return np.nan
    return float(np.log(t[0] / t[-1]) / np.log(kbar[-1] / kbar[0]))


def alpha_of(t, kbar):
    """Exponent in t* ~ k^-alpha. 0 = vertical contour, 1 = horizontal."""
    ok = np.isfinite(t)
    if ok.sum() < 2:
        return np.nan
    return float(-np.polyfit(np.log(kbar[ok]), np.log(t[ok]), 1)[0])


def window(c, levels):
    """(xlim, ylim) for one panel: the contour bounding box, padded and square
    in decades, since panels use equal log aspect."""
    lt, ln = [], []
    for lv in c["arms"].values():
        for L in levels:
            t, n = lv[L]
            ok = np.isfinite(t) & np.isfinite(n)
            lt.extend(np.log10(t[ok]))
            ln.extend(np.log10(n[ok]))
    lt, ln = np.array(lt), np.array(ln)
    span = max(np.ptp(lt), np.ptp(ln)) + 2 * PAD
    span = max(span, MIN_SPAN)
    cx, cy = lt.mean(), ln.mean()
    return ((10 ** (cx - span / 2), 10 ** (cx + span / 2)),
            (10 ** (cy - span / 2), 10 ** (cy + span / 2)))


def draw_grid(ax, kbar, kmean, labels, xlim, ylim, label_rays=True):
    """Grey lines n = 2 k t / <k>, one per degree bin, labelled near the
    lower-left end of their visible part."""
    lx0, lx1 = np.log10(xlim)
    ly0, ly1 = np.log10(ylim)
    for b, k in enumerate(kbar):
        off = np.log10(2.0 * k / kmean)            # log n - log t along the ray
        x = np.array(xlim)
        ax.plot(x, 2.0 * k * x / kmean, "-", color=GRID_COLOR, lw=0.8,
                zorder=0, solid_capstyle="butt")
        if not label_rays:
            continue
        xa, xb = max(lx0, ly0 - off), min(lx1, ly1 - off)
        if xb <= xa:
            continue                                # ray misses the window
        xt = xa + 0.06 * (xb - xa)
        ax.text(10 ** xt, 10 ** (xt + off), labels[b].replace("$k$ = ", ""),
                fontsize=5.8, color="#9A9A9A", rotation=45,
                rotation_mode="anchor", ha="left", va="bottom", zorder=1)


def draw_panel(ax, c, levels, equal_aspect, xlim=None, ylim=None):
    if xlim is None:
        xlim, ylim = window(c, levels)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    draw_grid(ax, c["kbar"], c["kmean"], c["labels"], xlim, ylim)

    for rule, _, color, mk, ls in ARMS:
        lv = c["arms"].get(rule)
        if lv is None:
            continue
        for L in levels:
            t, n = lv[L]
            if not np.isfinite(t).any():
                continue
            ax.plot(t, n, marker=mk, ls="none", color=color, ms=3.2,
                    zorder=5 if rule == "edge" else 4)
            ft = fit_loglog(t, n)
            if ft is not None:
                ax.plot(ft[0], ft[1], ls=ls, color=color, lw=1.0, alpha=0.8,
                        zorder=3 if rule == "edge" else 2)
            if rule == "edge":
                j = np.nonzero(np.isfinite(t))[0][0]
                ax.annotate(f"{L:g}", (t[j], n[j]), xytext=(3, -9), fontsize=6.5,
                            textcoords="offset points", color=color, zorder=6)

    # quenched tilts, one slot per level (a dash where the level is not reached)
    if "edge" in c["arms"]:
        tl = [tilt_of(c["arms"]["edge"][L][0]) for L in levels]
        if any(np.isfinite(v) for v in tl):
            ax.text(0.03, 0.97, "tilt " + "/".join(
                f"{v:.2f}" if np.isfinite(v) else "–" for v in tl),
                transform=ax.transAxes, fontsize=6.5, ha="left", va="top",
                color="0.3", zorder=6)

    ax.tick_params(labelsize=7)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


def draw_dist(ax, cs, use, levels):
    """Tilts of all panels, one row per level, quenched above and annealed
    below. Only contours on which every bin crossed. The top axis is tau."""
    kbar = cs[use[0]]["kbar"]
    K = kbar[-1] / kbar[0]
    rng = np.random.default_rng(0)          # vertical jitter
    for i, L in enumerate(levels):
        for rule, dy, mk, cols in (("edge", 0.2, "o", LEVEL_COLORS),
                                   ("annealed", -0.2, "s", LEVEL_COLORS_ANN)):
            v = []
            for ID in use:
                lv = cs[ID]["arms"].get(rule)
                if lv is None:
                    continue
                t = lv[L][0]
                if np.isfinite(t).all():
                    v.append(t[0] / t[-1])
            v = np.array(v)
            y0 = -i + dy
            col = cols[L]
            ax.scatter(v, y0 + rng.uniform(-0.1, 0.1, len(v)), s=13, marker=mk,
                       color=col, edgecolor="0.25", lw=0.4, zorder=3)
            if len(v):
                m = np.median(v)
                ax.plot([m, m], [y0 - 0.17, y0 + 0.17], color="0.15", lw=2.2,
                        zorder=4, solid_capstyle="butt")
    for x, ls in ((1.0, "--"), (np.sqrt(K), ":"), (K, "--")):
        ax.axvline(x, color="0.55", lw=0.8, ls=ls, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.45, K * 1.35)
    ax.set_ylim(-len(levels) + 0.45, 0.72)
    ax.set_xticks([1, np.sqrt(K), K])
    ax.set_xticklabels(["1", f"{np.sqrt(K):.2f}", f"{K:.2f}"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([-i for i in range(len(levels))])
    ax.set_yticklabels([f"{L:g}" for L in levels], fontsize=7.5)
    ax.tick_params(axis="y", which="both", length=0)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel(r"tilt  $t^*_{\rm low}/t^*_{\rm hub}$", fontsize=8)
    # top axis: ticks at tau = 0, 0.5, 1
    top = ax.twiny()
    top.set_xscale("log")
    top.set_xlim(ax.get_xlim())
    top.set_xticks([1, np.sqrt(K), K])
    top.set_xticklabels(["0", "0.5", "1"])
    top.xaxis.set_minor_locator(plt.NullLocator())
    top.tick_params(labelsize=7, direction="in")
    top.set_xlabel(r"normalised tilt $\tau$", fontsize=8)
    ax.set_title("distribution of tilts", fontsize=9)


def plot(model, topo, N, edges, levels, equal_aspect, shared):
    cs = {ID: contours(model, ID, topo, N, edges, levels) for ID in IDS
          if (model, ID) not in EXCLUDE}
    use = [ID for ID in IDS if cs.get(ID) is not None and has_data(cs[ID])]
    skipped = [ID for ID in IDS if ID not in use]
    if not use:
        print(f"  {model:<13} no usable word pair, skipped")
        return None

    xl = yl = None
    if shared:
        ws = [window(cs[ID], levels) for ID in use]
        xl = (min(w[0][0] for w in ws), max(w[0][1] for w in ws))
        yl = (min(w[1][0] for w in ws), max(w[1][1] for w in ws))

    # one extra slot for the distribution panel, bottom right
    nrow = int(np.ceil((len(use) + 1) / NCOL))
    fig, axes = plt.subplots(nrow, NCOL, figsize=(2.55 * NCOL, 2.75 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, ID in zip(axes, use):
        draw_panel(ax, cs[ID], levels, equal_aspect, xl, yl)
        ax.set_title(WORDS[ID], fontsize=9)
    for ax in axes[len(use):]:
        ax.set_axis_off()

    # x label on the lowest occupied panel of each column
    bottom = {j % NCOL: j for j in range(len(use))}
    for j, ax in enumerate(axes[:len(use)]):
        if j % NCOL == 0:
            ax.set_ylabel(r"own interactions $n$", fontsize=8)
        if bottom[j % NCOL] == j:
            ax.set_xlabel(r"population rounds $t$", fontsize=8)

    gam = topo.split("_")[1] if topo.startswith("ucm") else topo
    fig.suptitle(rf"{MODEL_LABELS.get(model, model)}   —   iso-success contours "
                 rf"by word pair,  $N$ = {N},  $\gamma$ = {gam}",
                 y=0.995, fontsize=12.5)
    handles = [Line2D([], [], color=GRID_COLOR, lw=1.6,
                      label=r"degree bin, $n_k = 2kt/\langle k\rangle$")]
    handles += [Line2D([], [], color=col, marker=mk, ls=ls, lw=1.6, ms=7,
                       label=f"iso-$S$ {tag}")
                for _, tag, col, mk, ls in ARMS]

    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / (2.75 * nrow)), h_pad=1.4)

    # The distribution panel is added after tight_layout so that its extra
    # labels do not change the spacing of the grid. It takes the width of the
    # panel above it and is slightly shorter.
    ax_above = axes[nrow * NCOL - 1 - NCOL]
    ax_above.apply_aspect()
    above = ax_above.get_position()
    cell = axes[nrow * NCOL - 1].get_position()
    y0 = cell.y0 + (cell.height - above.height) / 2
    dist_ax = fig.add_axes([above.x0, y0, above.width, 0.75 * above.height])
    draw_dist(dist_ax, cs, use, levels)
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1 - 0.30 / (2.75 * nrow)),
               handlelength=2.4, columnspacing=2.0, framealpha=0.9)
    path = fs.savefig(fig, f"fig1e_isosuccess_{model}_{topo}_N{N}")
    plt.close(fig)
    print(f"  {model:<13} {len(use):>2} panels"
          + (f", omitted {[WORDS[i] for i in skipped]}" if skipped else "")
          + f"  ->  {path}.{{pdf,png}}")
    return cs


def check_on_grid(allcs, levels, tol=0.01):
    """Assert that every plotted point lies on its line n* = 2 kbar t* / <k>."""
    worst = 0.0
    for (model, ID), c in allcs.items():
        for lv in c["arms"].values():
            for L in levels:
                t, n = lv[L]
                ok = np.isfinite(t) & np.isfinite(n)
                if not ok.any():
                    continue
                pred = 2.0 * c["kbar"][ok] * t[ok] / c["kmean"]
                worst = max(worst, float(np.abs(n[ok] / pred - 1).max()))
    assert worst < tol, f"markers off the degree grid by {worst:.1%}"
    print(f"\n  grid check: every plotted n* is within {worst:.2%} of "
          f"2*k*t*/<k>  (tolerance {tol:.0%})")


def write_csvs(allcs, topo, N, levels):
    os.makedirs(paths.TABLES, exist_ok=True)
    cells = os.path.join(paths.TABLES, "isosuccess_cells.csv")
    cont = os.path.join(paths.TABLES, "isosuccess_contours.csv")
    nc = nr = 0
    with open(cells, "w", newline="") as f1, open(cont, "w", newline="") as f2:
        w1 = csv.writer(f1)
        w1.writerow(["model", "ID", "words", "topo", "N", "rule", "level",
                     "bin", "kbar", "t_star", "n_star"])
        w2 = csv.writer(f2)
        w2.writerow(["model", "ID", "words", "topo", "N", "rule", "level",
                     "n_bins", "tilt", "n_ratio", "alpha", "rma_slope", "rma_r",
                     "tau"])
        for (model, ID), c in allcs.items():
            for rule, lv in c["arms"].items():
                for L in levels:
                    t, n = lv[L]
                    ok = np.nonzero(np.isfinite(t))[0]
                    if not len(ok):
                        continue
                    for b in ok:
                        w1.writerow([model, ID, WORDS[ID], topo, N, rule,
                                     L, b, f"{c['kbar'][b]:.4f}",
                                     f"{t[b]:.6g}", f"{n[b]:.6g}"])
                        nc += 1
                    ft = fit_loglog(t, n)
                    w2.writerow([
                        model, ID, WORDS[ID], topo, N, rule, L, len(ok),
                        f"{tilt_of(t):.4f}",
                        f"{n[ok[-1]] / n[ok[0]]:.4f}",
                        f"{alpha_of(t, c['kbar']):.4f}",
                        f"{ft[2]:.4f}" if ft else "",
                        f"{ft[3]:.4f}" if ft else "",
                        f"{tau_of(t, c['kbar']):.4f}"])
                    nr += 1
    print(f"  wrote {os.path.basename(cells)} ({nc} rows) and "
          f"{os.path.basename(cont)} ({nr} rows)")


def summarize(allcs, levels):
    """Print the tilts of every word pair."""
    print("\n--- iso-success contours per word pair "
          "(tilt = t*(low-degree)/t*(hub); > 1 when hubs are first) ---")
    head = "".join(f"{'tilt ' + f'{L:g}':>11}{'alpha':>8}" for L in levels)
    print(f"{'model':<14}{'word pair':<18}{'arm':<10}" + head)
    within = tot = 0
    for model in MODELS:
        for ID in IDS:
            c = allcs.get((model, ID))
            if c is None:
                continue
            rows = {}
            for rule, tag, _, _, _ in ARMS:
                lv = c["arms"].get(rule)
                if lv is None:
                    continue
                rows[rule] = [(tilt_of(lv[L][0]), alpha_of(lv[L][0], c["kbar"]))
                              for L in levels]
            for i, (rule, vals) in enumerate(rows.items()):
                tag = next(t for r, t, _, _, _ in ARMS if r == rule)
                cells = "".join(
                    (f"{v:>11.2f}" if np.isfinite(v) else f"{'-':>11}")
                    + (f"{a:>8.3f}" if np.isfinite(a) else f"{'-':>8}")
                    for v, a in vals)
                print(f"{model if i == 0 else '':<14}"
                      f"{WORDS[ID] if i == 0 else '':<18}{tag:<10}" + cells)
            if "edge" in rows and "annealed" in rows:
                for (q, _), (a, _) in zip(rows["edge"], rows["annealed"]):
                    if np.isfinite(q) and np.isfinite(a):
                        tot += 1
                        within += abs(a / q - 1) <= 0.10
    if tot:
        print(f"\n  annealed tilt within 10% of the quenched tilt in {within} "
              f"of {tot} conditions ({100 * within / tot:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="ucm_2.5")
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--edges", type=int, nargs="+", default=EDGES)
    ap.add_argument("--levels", type=float, nargs="+", default=LEVELS)
    ap.add_argument("--no-equal-aspect", dest="equal", action="store_false",
                    help="do not force equal aspect on the log axes")
    ap.add_argument("--shared-axes", action="store_true",
                    help="same axis limits for every panel of a figure")
    a = ap.parse_args()

    fs.apply_style()
    allcs = {}
    for model in MODELS:
        cs = plot(model, a.topo, a.N, a.edges, a.levels, a.equal, a.shared_axes)
        if cs:
            allcs.update({(model, ID): c for ID, c in cs.items()
                          if c is not None and has_data(c)})
    check_on_grid(allcs, a.levels)
    write_csvs(allcs, a.topo, a.N, a.levels)
    summarize(allcs, a.levels)


if __name__ == "__main__":
    main()
