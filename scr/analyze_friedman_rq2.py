"""Friedman tests on the critical committed fraction p* (N = 10000, edge rule).

The block is a (model, word pair) cell observed under every level of the factor:

  placement   hub < random < low-degree, on UCM 2.5 and UCM 5.0   (k = 3)
  topology    RRN < UCM 5.0 < UCM 2.5 < FC, random placement       (k = 4)

Each family is tested per model and direction, and pooled over models per
direction. "strong" is the cost of overturning the convention the population
prefers on FC (see plot_pstar.direction_labels).

p* is rounded to the 0.01 bisection resolution before ranking. A block is
dropped when any of its cells has p* = 0 (no convention to overturn).

    python analyze_friedman_rq2.py [--no-csv]
"""
import argparse
import csv
import os
import pickle

import numpy as np
from scipy.stats import friedmanchisquare, page_trend_test, wilcoxon

import paths
from analyze_friedman import (ALPHA, IDS, MODELS, fmt_p, holm, posthoc_floor,
                              rank_rows, words)
from plot_pstar import direction_labels

HERE = paths.ROOT
GRID = paths.GRID
OUTDIR = paths.TABLES
N_FIX = 10000
GRID_STEP = 0.01
DIRECTIONS = ["strong", "weak"]
MODEL_LABELS = {"llama31_70B": "Llama-3.1-70B", "gpt-4o": "GPT-4o",
                "QWQ_32B": "QwQ-32B", "phi_4": "Phi-4"}

# family -> (topo, placement) columns in the predicted ascending order
FAMILIES = {
    "placement_ucm_2.5": [("ucm_2.5", "hub"), ("ucm_2.5", "random"), ("ucm_2.5", "leaf")],
    "placement_ucm_5.0": [("ucm_5.0", "hub"), ("ucm_5.0", "random"), ("ucm_5.0", "leaf")],
    "topology_random":   [("rrn", "random"), ("ucm_5.0", "random"),
                          ("ucm_2.5", "random"), ("fc", "random")],
}
COL_LABEL = {("ucm_2.5", "hub"): "hub", ("ucm_2.5", "random"): "random",
             ("ucm_2.5", "leaf"): "low-degree", ("ucm_5.0", "hub"): "hub",
             ("ucm_5.0", "random"): "random", ("ucm_5.0", "leaf"): "low-degree",
             ("rrn", "random"): "RRN", ("fc", "random"): "FC"}
TOPO_COL = {"rrn": "RRN", "ucm_5.0": "UCM 5.0", "ucm_2.5": "UCM 2.5", "fc": "FC"}


def col_name(family, c):
    return COL_LABEL[c] if family.startswith("placement") else TOPO_COL[c[0]]


# Loading
def load_cells():
    """One dict per (model, ID, direction, topo, placement)."""
    w_of = direction_labels()
    conds = sorted({c for cols in FAMILIES.values() for c in cols})
    cells = []
    for model in MODELS:
        for ID in IDS:
            for d in DIRECTIONS:
                w = w_of[(model, ID)][d]
                for topo, pl in conds:
                    p = os.path.join(GRID, f"RQ2_{model}_ID_{ID}_{topo}_N{N_FIX}_"
                                           f"{pl}_dir{w}_edge_H5.pkl")
                    with open(p, "rb") as f:
                        ps = pickle.load(f)["p_star"]
                    assert ps is not None, p
                    cells.append({"model": model, "ID": ID, "words": words(ID),
                                  "direction": d, "committed_word": w,
                                  "topo": topo, "placement": pl,
                                  "p_star": float(ps),
                                  "p_round": round(round(float(ps) / GRID_STEP) * GRID_STEP, 2)})
    return cells


def index(cells):
    return {(c["model"], c["ID"], c["direction"], c["topo"], c["placement"]): c
            for c in cells}


# Tests
def run_block_test(mat, names):
    """Friedman test, Kendall's W, Page's L and Holm-corrected Wilcoxon
    post-hocs on a matrix whose columns are in the predicted order."""
    n, k = mat.shape
    out = {"n": n, "k": k, "mat": mat, "names": names}
    if n < 2 or np.all(mat == mat[:, :1]):
        out.update(Q=float("nan"), p=1.0, W=float("nan"), ranks=rank_rows(mat),
                   rank_sums=rank_rows(mat).sum(axis=0), page_L=float("nan"),
                   page_p=1.0, posthoc=[], n_tied_rows=int(n))
        return out
    Q, p = friedmanchisquare(*mat.T)
    ranks = rank_rows(mat)
    page = page_trend_test(mat, ranked=False)
    idx_pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    post = []
    if p < ALPHA:
        raw, stats_, zeros, n_eff = [], [], [], []
        for a, b in idx_pairs:
            d = mat[:, a] - mat[:, b]
            zeros.append(int((d == 0).sum()))
            n_eff.append(int((d != 0).sum()))
            if np.all(d == 0):
                stats_.append(float("nan"))
                raw.append(1.0)
            else:
                res = wilcoxon(mat[:, a], mat[:, b], zero_method="pratt")
                stats_.append(float(res.statistic))
                raw.append(float(res.pvalue))
        adj = holm(raw)
        post = [{"pair": (names[a], names[b]), "stat": s, "p_raw": r, "p_holm": h,
                 "n_zero": z, "n_eff": ne,
                 "decidable": min(1.0, posthoc_floor(ne) * len(idx_pairs)) < ALPHA,
                 "median_diff": float(np.median(mat[:, a] - mat[:, b]))}
                for (a, b), s, r, h, z, ne in zip(idx_pairs, stats_, raw, adj,
                                                  zeros, n_eff)]
    out.update(Q=float(Q), p=float(p), W=float(Q / (n * (k - 1))), ranks=ranks,
               rank_sums=ranks.sum(axis=0), page_L=float(page.statistic),
               page_p=float(page.pvalue), posthoc=post,
               n_tied_rows=int(np.sum(np.all(mat == mat[:, :1], axis=1))))
    return out


def run_all(cells):
    """{(family, model|'ALL', direction): test}."""
    idx = index(cells)
    tests, excluded = {}, []
    for fam, cols in FAMILIES.items():
        names = [col_name(fam, c) for c in cols]
        for d in DIRECTIONS:
            for model in MODELS + ["ALL"]:
                models = MODELS if model == "ALL" else [model]
                rows, mat = [], []
                for m in models:
                    for ID in IDS:
                        vals = [idx[(m, ID, d, t, pl)] for t, pl in cols]
                        if any(v["p_star"] == 0 for v in vals):
                            if model != "ALL":
                                excluded.append((fam, m, ID, d,
                                                 [v["p_star"] for v in vals]))
                            continue
                        rows.append((m, ID))
                        mat.append([v["p_round"] for v in vals])
                t = run_block_test(np.array(mat), names)
                t.update(family=fam, model=model, direction=d, rows=rows,
                         n_ordered=int(np.sum(np.all(np.diff(np.array(mat), axis=1) >= 0, axis=1))))
                tests[(fam, model, d)] = t
    return tests, excluded


# Descriptives
def grid_mode(x):
    """Mode on the 0.01-rounded values; ties -> smallest value."""
    vals, counts = np.unique(np.round(np.asarray(x), 2), return_counts=True)
    return float(vals[np.argmax(counts)]), int(counts.max())


def describe(cells):
    """Per (topo, placement) pooled over models + directions, and per model x
    direction x topo x placement.  Both with and without the p* = 0 cells."""
    rows = []
    conds = sorted({(c["topo"], c["placement"]) for c in cells})
    groups = [("ALL", "pooled")] + [(m, d) for m in MODELS for d in DIRECTIONS]
    for model, d in groups:
        for topo, pl in conds:
            sub = [c for c in cells if c["topo"] == topo and c["placement"] == pl
                   and (model == "ALL" or (c["model"] == model and c["direction"] == d))]
            for zeros in ("with_zeros", "no_zeros"):
                s = sub if zeros == "with_zeros" else [c for c in sub if c["p_star"] > 0]
                raw = np.array([c["p_star"] for c in s])
                rnd = np.array([c["p_round"] for c in s])
                mo, mc = grid_mode(rnd)
                rows.append({"model": model, "direction": d, "topo": topo,
                             "placement": pl, "zeros": zeros, "n": len(s),
                             "mean": raw.mean(), "median": float(np.median(raw)),
                             "mode": mo, "mode_count": mc,
                             "n_zero": int(np.sum(raw == 0)),
                             "n_one": int(np.sum(rnd >= 1.0))})
    return rows


# Printing
def lab(m):
    return "ALL" if m == "ALL" else MODEL_LABELS[m]


def print_tests(tests):
    for fam in FAMILIES:
        t0 = next(t for (f, _, _), t in tests.items() if f == fam)
        print(f"\n=== {fam}   columns (predicted ascending): {' < '.join(t0['names'])}")
        print(f"{'model':<14s}{'dir':<7s}{'n':>3s}{'tied':>5s}{'ord':>5s}"
              f"{'Q':>8s}{'p':>11s}{'W':>7s}{'L':>8s}{'p(L)':>11s}   rank sums")
        for d in DIRECTIONS:
            for model in MODELS + ["ALL"]:
                t = tests[(fam, model, d)]
                print(f"{lab(model):<14s}{d:<7s}{t['n']:>3d}{t['n_tied_rows']:>5d}"
                      f"{t['n_ordered']:>5d}{t['Q']:>8.2f}{fmt_p(t['p']):>11s}"
                      f"{t['W']:>7.3f}{t['page_L']:>8.0f}{fmt_p(t['page_p']):>11s}   "
                      + " ".join(f"{r:6.1f}" for r in t["rank_sums"]))


def print_posthoc(tests):
    print("\n=== Post-hocs (Wilcoxon, pratt, Holm), where Friedman p < .05")
    for (fam, model, d), t in tests.items():
        for ph in t["posthoc"]:
            flag = ("sig" if ph["p_holm"] < ALPHA
                    else ("n.s." if ph["decidable"] else "UNDECIDABLE"))
            print(f"{fam:<19s}{lab(model):<14s}{d:<7s}{ph['pair'][0]:>10s} vs "
                  f"{ph['pair'][1]:<11s} med.diff {ph['median_diff']:+.3f}  "
                  f"ties {ph['n_zero']:>2d}  p_holm {fmt_p(ph['p_holm']):>9s}  {flag}")


def print_descriptives(desc):
    print("\n=== Pooled over models and directions (88 values per condition)")
    for zeros in ("with_zeros", "no_zeros"):
        print(f"--- {zeros}")
        for r in desc:
            if r["model"] == "ALL" and r["zeros"] == zeros:
                print(f"{r['topo']:<9s}{r['placement']:<8s} n={r['n']:>3d}  "
                      f"mean {r['mean']:.3f}  median {r['median']:.3f}  "
                      f"mode {r['mode']:.2f} (x{r['mode_count']})  "
                      f"zeros {r['n_zero']}  ones {r['n_one']}")


# CSV
def write_csvs(cells, tests, excluded, desc):
    os.makedirs(OUTDIR, exist_ok=True)
    written = []
    excl = {(m, ID, d) for _, m, ID, d, _ in excluded}

    f = os.path.join(OUTDIR, "rq2_cells.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "ID", "words", "direction", "committed_word", "topo",
                    "placement", "p_star", "p_round", "block_has_zero"])
        for c in cells:
            w.writerow([c["model"], c["ID"], c["words"], c["direction"],
                        c["committed_word"], c["topo"], c["placement"],
                        f"{c['p_star']:.6f}", f"{c['p_round']:.2f}",
                        int((c["model"], c["ID"], c["direction"]) in excl)])
    written.append(f)

    f = os.path.join(OUTDIR, "rq2_tests.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "model", "direction", "n", "n_tied_rows",
                    "n_rows_fully_ordered", "Q", "df", "p", "kendall_W",
                    "rank_sums_ascending_order", "rank_sums", "page_L", "page_p",
                    "columns"])
        for (fam, model, d), t in tests.items():
            w.writerow([fam, model, d, t["n"], t["n_tied_rows"], t["n_ordered"],
                        f"{t['Q']:.4f}", t["k"] - 1, f"{t['p']:.6g}",
                        f"{t['W']:.4f}",
                        int(np.all(np.diff(t["rank_sums"]) > 0)),
                        " ".join(f"{r:.1f}" for r in t["rank_sums"]),
                        f"{t['page_L']:.1f}", f"{t['page_p']:.6g}",
                        " < ".join(t["names"])])
    written.append(f)

    f = os.path.join(OUTDIR, "rq2_ranks.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "model", "direction", "ID", "words", "columns",
                    "values_round", "ranks"])
        for (fam, model, d), t in tests.items():
            if model == "ALL":
                continue
            for (m, ID), vals, rks in zip(t["rows"], t["mat"], t["ranks"]):
                w.writerow([fam, m, d, ID, words(ID), " < ".join(t["names"]),
                            " ".join(f"{v:.2f}" for v in vals),
                            " ".join(f"{r:.1f}" for r in rks)])
    written.append(f)

    f = os.path.join(OUTDIR, "rq2_posthoc.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "model", "direction", "pair", "median_diff",
                    "n_zero", "n_eff", "p_raw", "p_holm", "significant",
                    "decidable"])
        for (fam, model, d), t in tests.items():
            for ph in t["posthoc"]:
                w.writerow([fam, model, d, f"{ph['pair'][0]}_vs_{ph['pair'][1]}",
                            f"{ph['median_diff']:.4f}", ph["n_zero"], ph["n_eff"],
                            f"{ph['p_raw']:.6g}", f"{ph['p_holm']:.6g}",
                            int(ph["p_holm"] < ALPHA), int(ph["decidable"])])
    written.append(f)

    f = os.path.join(OUTDIR, "rq2_descriptives.csv")
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(desc[0]))
        w.writeheader()
        for r in desc:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    written.append(f)

    f = os.path.join(OUTDIR, "rq2_excluded_blocks.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "model", "ID", "words", "direction", "p_star_values"])
        for fam, m, ID, d, vals in excluded:
            w.writerow([fam, m, ID, words(ID), d, " ".join(f"{v:.3f}" for v in vals)])
    written.append(f)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-csv", action="store_true", help="print tables only")
    args = ap.parse_args()

    cells = load_cells()
    assert len(cells) == 704, len(cells)

    raw = {(t, pl): np.mean([c["p_star"] for c in cells
                             if c["topo"] == t and c["placement"] == pl])
           for t, pl in [("ucm_2.5", "hub"), ("ucm_2.5", "random"), ("ucm_2.5", "leaf"),
                         ("ucm_5.0", "hub"), ("ucm_5.0", "random"), ("ucm_5.0", "leaf")]}
    print("mean p* (hub, random, leaf on UCM 2.5, then UCM 5.0):",
          " ".join(f"{v:.3f}" for v in raw.values()))

    tests, excluded = run_all(cells)
    print(f"\nExcluded blocks (p* = 0 in at least one cell): {len(excluded)}")
    for fam, m, ID, d, vals in excluded:
        print(f"  {fam:<19s}{MODEL_LABELS[m]:<14s}{words(ID):<14s}{d:<7s}"
              + " ".join(f"{v:.3f}" for v in vals))

    print_tests(tests)
    print_posthoc(tests)
    desc = describe(cells)
    print_descriptives(desc)

    if not args.no_csv:
        for f in write_csvs(cells, tests, excluded, desc):
            print("wrote", os.path.relpath(f, HERE))


if __name__ == "__main__":
    main()
