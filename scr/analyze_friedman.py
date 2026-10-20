"""
Friedman tests on convergence time across topologies.

Topology is the within-subject factor and a (model, word pair) cell is the
block, observed on FC, UCM 2.5, UCM 5.0 and RRN. Tests are run separately for
each population size, activation rule (edge, node) and per-cell statistic
(mean, mode of the convergence time).

Per-run times are taken from data/tmax where the non-converged runs were
replayed with a longer t_max, and from data/grid otherwise. After the merge no
run is censored, so every cell statistic is computed over all runs.

The two Llama pairs {young, old} and {more, less} are excluded: they never
converge on FC, even with t_max = 1e5.

`python analyze_friedman.py density` runs the same tests on the RRN degree
sweep (data/density), with FC and K = 256 ... 4 as the columns.

    python analyze_friedman.py [density] [--no-csv]
"""
import argparse
import csv
import glob
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import friedmanchisquare, gaussian_kde, page_trend_test, wilcoxon

import paths

HERE = paths.ROOT
GRID = paths.GRID
TMAX = paths.TMAX

MODELS = ["QWQ_32B", "gpt-4o", "llama31_70B", "phi_4"]
IDS = [92, 102, 191, 219, 344, 349, 397, 410, 505, 626, 1161]
# Columns in the predicted ascending order, as page_trend_test expects.
TOPOS = ["fc", "ucm_2.5", "ucm_5.0", "rrn"]
TOPO_LABEL = {"fc": "FC", "ucm_2.5": "UCM 2.5", "ucm_5.0": "UCM 5.0", "rrn": "RRN"}
NS = [1000, 5000, 10000, 100000]
RULES = ["edge", "node"]
STATS = ["mean", "mode"]
H = 5

# Density mode: RRN at N = 10000 across the degree K, with FC as the K -> N-1
# limit. Columns run in decreasing K so that the predicted convergence time
# increases from left to right.
DENSITY = paths.DENSITY
DENSITY_OUT = paths.TABLES
DENSITY_N = 10000
KS_DESC = [256, 192, 128, 96, 64, 48, 32, 24, 20, 16, 12, 10, 8, 6, 4]
DENSITY_TOPOS = ["fc"] + [f"rrn_k{K}" for K in KS_DESC]
TOPO_LABEL.update({f"rrn_k{K}": f"k={K}" for K in KS_DESC})

DENSITY_MODE = False      # set by main()
CSV_PREFIX = "friedman"
CSV_DIR = paths.TABLES

# Post-hoc families in density mode. With 11 blocks the smallest signed-rank p
# is 9.8e-4, so Holm over all 120 column pairs could never reach 0.05. Two
# smaller families are used instead:
#   vs_fc : each K against FC
#   adj   : each K against the next larger K
FAM_VS_FC = "vs_fc"
FAM_ADJ = "adj"


def density_pairs(family, k):
    """Column-index pairs for a family.  Column 0 is fc; 1..k-1 are K descending."""
    if family == FAM_VS_FC:
        return [(0, j) for j in range(1, k)]
    return [(j, j + 1) for j in range(1, k - 1)]


def src_dir(topo):
    """Directory of a cell's pickle. In density mode FC comes from data/grid."""
    return GRID if (not DENSITY_MODE or topo == "fc") else DENSITY

# Pairs that never converge on FC, so their block can never be complete.
EXCLUDED = {
    ("llama31_70B", 219): "FC consensus fraction 0.000 at every N, also at "
                          "t_max=1e5 (only UCM 2.5 at N=1000 escapes)",
    ("llama31_70B", 1161): "FC consensus fraction 0.000 at every N, also at "
                           "t_max=1e5 (only RRN at N=1000 escapes)",
}

MODE_EMPIRICAL_MAX_UNIQUE = 20   # <= this many distinct t -> empirical mode
KDE_GRID = 2048
ALPHA = 0.05

crows = pickle.load(open(paths.WORD_PAIRS, "rb"))


def words(ID):
    return "/".join(crows[ID]["differences"])


# Loading
def file_rule(topo, rule):
    """On FC both rules are the same process, and only the edge file exists."""
    return "edge" if topo == "fc" else rule


def load_long():
    """Long-t_max replays, keyed by (model, ID, topo, N, rule)."""
    out = {}
    for f in glob.glob(os.path.join(TMAX, "RQ1LONG_*.pkl")):
        d = pickle.load(open(f, "rb"))
        out[(d["model"], d["ID"], d["topo"], d["N"], d["rule"])] = d
    return out


def cell_times(model, ID, topo, N, rule, long_files):
    """Convergence times of the converged runs of one cell.

    Returns (times, source, n_censored, n_unreplayed, n_runs_total).
    """
    fr = file_rule(topo, rule)
    key = (model, ID, topo, N, fr)
    if key in long_files:
        d = long_files[key]
        merged = d["conv_times_merged"]
        censored = d["censored"]
        t = np.array([v for v in merged if v is not None and v != -1], dtype=np.int64)
        n_unrep = sum(1 for v in merged if v is None)
        return t, "merged", int(sum(censored)), n_unrep, len(merged)

    path = os.path.join(src_dir(topo),
                        f"RQ1_{model}_ID_{ID}_{topo}_N{N}_{fr}_H{H}.pkl")
    d = pickle.load(open(path, "rb"))
    ct = np.asarray(d["conv_times_all"], dtype=np.int64)
    return ct[ct >= 0], "grid", int((ct < 0).sum()), 0, len(ct)


# Per-cell summary statistics
def cell_mode(t):
    """Mode of t (in rounds, not log rounds). Returns (mode, method).

    With at most 20 distinct values the empirical mode is used (ties go to the
    smallest t); otherwise the peak of a Gaussian KDE.
    """
    vals, counts = np.unique(t, return_counts=True)
    if len(vals) <= MODE_EMPIRICAL_MAX_UNIQUE:
        return float(vals[np.argmax(counts)]), "empirical"
    kde = gaussian_kde(t.astype(float))          # Scott bandwidth
    grid = np.linspace(t.min(), t.max(), KDE_GRID)
    return float(grid[np.argmax(kde(grid))]), "kde"


def build_cells(long_files):
    """One record per (rule, N, model, ID, topo)."""
    jobs = [(rule, N, m, i, topo)
            for rule in RULES for N in NS
            for m in MODELS for i in IDS if (m, i) not in EXCLUDED
            for topo in TOPOS]

    def one(job):
        rule, N, m, i, topo = job
        t, src, n_cens, n_unrep, n_tot = cell_times(m, i, topo, N, rule, long_files)
        mode, method = cell_mode(t)
        return {
            "rule": rule, "N": N, "model": m, "ID": i, "words": words(i),
            "topo": topo, "R": len(t), "runs_total": n_tot,
            "n_censored": n_cens, "n_unreplayed": n_unrep,
            "mean": float(t.mean()), "mode": mode, "mode_method": method,
            "n_unique": int(len(np.unique(t))),
            "median": float(np.median(t)), "t_min": int(t.min()), "t_max": int(t.max()),
            "source": src,
        }

    # threads hide the per-file latency of network or cloud-synced disks
    with ThreadPoolExecutor(max_workers=64) as ex:
        cells = list(ex.map(one, jobs))

    for c in cells:
        assert c["n_censored"] == 0 and c["n_unreplayed"] == 0, \
            f"censored/unreplayed runs left in {c}"
        assert c["R"] == c["runs_total"], f"run count mismatch in {c}"
    return cells


# Tests
def rank_rows(mat):
    """Ascending ranks within each row (1 = fastest), midranks for ties."""
    from scipy.stats import rankdata
    return np.vstack([rankdata(row) for row in mat])


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def posthoc_floor(n):
    """Smallest two-sided signed-rank p reachable with n blocks."""
    if n < 6:
        return 1.0
    x = np.arange(1.0, n + 1.0)
    return float(wilcoxon(x, x + 1.0, zero_method="pratt").pvalue)


def run_test(cells, rule, stat, N, model=None, family=None):
    """Friedman test, Kendall's W, Page's L and Holm-corrected Wilcoxon post-hocs.

    `model` restricts the blocks to one model (None pools all models);
    `family` restricts the post-hoc pairs (see density_pairs).
    """
    by_cell = {}
    for c in cells:
        if c["rule"] == rule and c["N"] == N and (model is None or c["model"] == model):
            by_cell.setdefault((c["model"], c["ID"]), {})[c["topo"]] = c
    rows = sorted(by_cell)
    mat = np.array([[by_cell[r][t][stat] for t in TOPOS] for r in rows])

    n, k = mat.shape
    Q, p = friedmanchisquare(*mat.T)
    W = Q / (n * (k - 1))
    ranks = rank_rows(mat)
    rank_sums = ranks.sum(axis=0)

    page = page_trend_test(mat, ranked=False)

    post = []
    idx_pairs = (density_pairs(family, k) if family
                 else [(a, b) for a in range(k) for b in range(a + 1, k)])
    if p < ALPHA:
        pairs, raw, zeros, stats_ = [], [], [], []
        n_eff = []
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
            pairs.append((TOPOS[a], TOPOS[b]))
        adj = holm(raw)
        post = [{"pair": pr, "stat": s, "p_raw": pr_, "p_holm": pa, "n_zero": z,
                 "n_eff": ne,
                 # with many ties a pair may be unable to reach alpha at all
                 "decidable": min(1.0, posthoc_floor(ne) * len(idx_pairs)) < ALPHA,
                 "median_diff": float(np.median(mat[:, TOPOS.index(pr[0])]
                                                - mat[:, TOPOS.index(pr[1])]))}
                for pr, s, pr_, pa, z, ne in zip(pairs, stats_, raw, adj,
                                                 zeros, n_eff)]

    floor = posthoc_floor(n)
    return {"rule": rule, "stat": stat, "N": N, "model": model or "ALL",
            "rows": rows, "mat": mat,
            "ranks": ranks, "n": n, "k": k, "Q": float(Q), "df": k - 1,
            "p": float(p), "W": float(W), "rank_sums": rank_sums,
            "page_L": float(page.statistic), "page_p": float(page.pvalue),
            "posthoc": post, "family": family or "all_pairs",
            "n_pairs": len(idx_pairs),
            "posthoc_floor": min(1.0, floor * len(idx_pairs))}


# Printing
def fmt_p(p):
    return "<1e-16" if p < 1e-16 else f"{p:.3g}"


def print_exclusions(cells):
    print("=" * 100)
    print("EXCLUDED CELLS")
    print("=" * 100)
    for (m, i), why in EXCLUDED.items():
        print(f"  {m:13s} ID {i:<5d} {words(i):16s} {why}")
    kept = len({(c['model'], c['ID']) for c in cells})
    print(f"\n  Same exclusion set for edge and node, and at every N.")
    print(f"  Rows kept: {kept} of {len(MODELS) * len(IDS)} "
          f"({len(MODELS)} models x {len(IDS)} word pairs).\n")


def print_cells(cells):
    print("=" * 100)
    print("PER-CELL SUMMARIES   (mean / mode of raw t; 'k' marks a KDE mode)")
    print("=" * 100)
    for rule in RULES:
        for N in NS:
            sub = [c for c in cells if c["rule"] == rule and c["N"] == N]
            R = sub[0]["R"]
            print(f"\n--- rule={rule}  N={N}  (R={R} runs/cell) ---")
            head = f"{'model':13s}{'ID':>5s} {'words':16s}"
            for t in TOPOS:
                head += f"{TOPO_LABEL[t]:>19s}"
            print(head)
            by = {(c["model"], c["ID"], c["topo"]): c for c in sub}
            for m in MODELS:
                for i in IDS:
                    if (m, i) in EXCLUDED:
                        continue
                    line = f"{m:13s}{i:5d} {words(i):16s}"
                    for t in TOPOS:
                        c = by[(m, i, t)]
                        flag = "k" if c["mode_method"] == "kde" else " "
                        line += f"{c['mean']:10.1f}/{c['mode']:7.1f}{flag}"
                    print(line)


def print_ranks(tests):
    print("\n" + "=" * 100)
    print("RANK MATRICES   (rank within each row, 1 = fastest)")
    print("=" * 100)
    for key in sorted(tests, key=lambda k: (k[2], k[0], k[1])):
        t = tests[key]
        print(f"\n--- rule={t['rule']}  stat={t['stat']}  N={t['N']}  "
              f"({t['n']} rows) ---")
        head = f"{'model':13s}{'ID':>5s} {'words':16s}"
        for tp in TOPOS:
            head += f"{TOPO_LABEL[tp]:>10s}"
        print(head)
        for (m, i), row in zip(t["rows"], t["ranks"]):
            line = f"{m:13s}{i:5d} {words(i):16s}"
            for r in row:
                line += f"{r:10.1f}"
            print(line)
        line = f"{'RANK SUM':34s}"
        for r in t["rank_sums"]:
            line += f"{r:10.1f}"
        print(line + f"    (total {t['rank_sums'].sum():.0f}, "
                     f"expected {t['n'] * t['k'] * (t['k'] + 1) / 2:.0f})")


def print_summary(tests):
    print("\n" + "=" * 100)
    print("SUMMARY -- one table per N, four (rule x statistic) combinations")
    print("=" * 100)
    for N in NS:
        print(f"\n### N = {N}")
        head = (f"{'rule':6s}{'stat':6s}{'n':>4s}"
                + "".join(f"{'R:' + TOPO_LABEL[t]:>12s}" for t in TOPOS)
                + f"{'Q':>10s}{'df':>4s}{'p':>11s}{'W':>8s}"
                + f"{'Page L':>10s}{'p(L)':>11s}")
        print(head)
        for rule in RULES:
            for stat in STATS:
                t = tests[(rule, stat, N)]
                print(f"{rule:6s}{stat:6s}{t['n']:4d}"
                      + "".join(f"{r:12.1f}" for r in t["rank_sums"])
                      + f"{t['Q']:10.2f}{t['df']:4d}{fmt_p(t['p']):>11s}"
                      f"{t['W']:8.3f}{t['page_L']:10.0f}{fmt_p(t['page_p']):>11s}")


def print_posthoc(tests):
    print("\n" + "=" * 100)
    print("POST-HOC pairwise Wilcoxon signed-rank (Holm-Bonferroni over 6 pairs)")
    print("=" * 100)
    for N in NS:
        for rule in RULES:
            for stat in STATS:
                t = tests[(rule, stat, N)]
                print(f"\n--- rule={rule}  stat={stat}  N={N} ---")
                if not t["posthoc"]:
                    print(f"  Friedman not significant (p={fmt_p(t['p'])}); "
                          "no post-hoc run.")
                    continue
                print(f"  {'pair':22s}{'med diff':>11s}{'zeros':>7s}"
                      f"{'p raw':>11s}{'p Holm':>11s}   verdict")
                for ph in t["posthoc"]:
                    a, b = ph["pair"]
                    lab = f"{TOPO_LABEL[a]} vs {TOPO_LABEL[b]}"
                    v = "differ" if ph["p_holm"] < ALPHA else "NOT distinguishable"
                    print(f"  {lab:22s}{ph['median_diff']:11.1f}{ph['n_zero']:7d}"
                          f"{fmt_p(ph['p_raw']):>11s}{fmt_p(ph['p_holm']):>11s}   {v}")


# Density mode: the saturation table
NAME = {"none": "none -- every K separates from FC",
        "undetermined": "undetermined (post-hocs cannot resolve)",
        "no_posthoc": "n/a (Friedman not significant)"}


def thresholds(t, med, fc):
    """(saturation K, plateau K) from a test's post-hocs.

    Both are the end of the run of non-significant comparisons that starts at
    the largest K.
    """
    ph = {tuple(sorted(x["pair"])): x for x in t["posthoc"]}

    def verdict(a, b):
        """'same', 'differ', or None if the comparison cannot reach alpha."""
        x = ph.get(tuple(sorted((a, b))))
        if x is None:
            return None
        if x["p_holm"] < ALPHA:
            return "differ"
        return "same" if x["decidable"] else None

    def walk(seq):
        """K, 'none' (the largest K already differs) or 'undetermined'."""
        if not t["posthoc"]:
            return "no_posthoc"          # Friedman not significant
        out = None
        for a, b, K in seq:
            v = verdict(a, b)
            if v == "same":
                out = K
            elif v == "differ":
                break
            else:
                return "undetermined" if out is None else out
        return out if out is not None else "none"

    sat = walk([(f"rrn_k{K}", "fc", K) for K in KS_DESC])
    plat = walk([(f"rrn_k{KS_DESC[j]}", f"rrn_k{KS_DESC[j - 1]}", KS_DESC[j])
                 for j in range(1, len(KS_DESC))])
    return sat, plat


def within(med, fc, thr):
    """Smallest K whose median excess over FC is <= thr (ascending K)."""
    hit = [K for K in sorted(KS_DESC) if med[f"rrn_k{K}"] / fc - 1 <= thr]
    return min(hit) if hit else None


def print_saturation(tests_fc, tests_adj):
    """Per-model table of T(K) against FC."""
    print("\n" + "=" * 100)
    print(f"DEGREE SWEEP   (RRN, N={DENSITY_N}, edge rule)")
    print("=" * 100)
    print("  Post-hocs, Holm-corrected within each family:")
    print(f"    'vs FC'  {len(KS_DESC)} pairs, each K against FC")
    print(f"    'adj'    {len(KS_DESC) - 1} pairs, each K against the next larger K\n")

    for stat in STATS:
        for model in ["ALL"] + MODELS:
            t = tests_fc[(stat, model)]
            ta = tests_adj[(stat, model)]
            col = {tp: i for i, tp in enumerate(TOPOS)}
            med = {tp: float(np.median(t["mat"][:, col[tp]])) for tp in TOPOS}
            fc = med["fc"]
            ph = {tuple(sorted(x["pair"])): x for x in t["posthoc"]}
            pha = {tuple(sorted(x["pair"])): x for x in ta["posthoc"]}
            sat, plat = thresholds(t, med, fc)

            label = "ALL MODELS POOLED" if model == "ALL" else model
            print("-" * 100)
            print(f"stat={stat}   {label}   n={t['n']} word pairs   "
                  f"post-hoc floor: vs FC {t['posthoc_floor']:.4f}, "
                  f"adj {ta['posthoc_floor']:.4f}")
            if t["posthoc_floor"] >= ALPHA:
                print("  no post-hoc can reach alpha=0.05 with this many blocks")
            print(f"{'K':>6s}{'median T':>11s}{'T/T_FC':>9s}{'p(vs FC)':>12s}"
                  f"{'  verdict':<22s}{'p(vs next K up)':>17s}")
            print(f"{'FC':>6s}{fc:11.2f}{1.0:9.3f}{'--':>12s}{'':<22s}{'--':>17s}")
            for j, K in enumerate(KS_DESC):
                tp = f"rrn_k{K}"
                x = ph.get(tuple(sorted((tp, "fc"))))
                xa = (pha.get(tuple(sorted((tp, f"rrn_k{KS_DESC[j - 1]}"))))
                      if j > 0 else None)
                if x is None:
                    v, q = "n/a", None
                elif x["p_holm"] < ALPHA:
                    v, q = "slower than FC", x["p_holm"]
                elif x["decidable"]:
                    v, q = "indistinguishable", x["p_holm"]
                else:
                    v, q = f"UNDECIDABLE (n_eff={x['n_eff']})", x["p_holm"]
                print(f"{K:>6d}{med[tp]:11.2f}{med[tp] / fc:9.3f}"
                      f"{(fmt_p(q) if q is not None else 'n/a'):>12s}"
                      f"  {v:<20s}"
                      f"{(fmt_p(xa['p_holm']) if xa is not None else '--'):>17s}")
            print(f"\n  Friedman: Q={t['Q']:.2f}, df={t['df']}, p={fmt_p(t['p'])}, "
                  f"W={t['W']:.3f};  Page's L={t['page_L']:.0f}, "
                  f"p={fmt_p(t['page_p'])}")
            print(f"  saturation K (contiguous, indistinguishable from FC): "
                  f"{NAME.get(sat, sat)}")
            print(f"  plateau onset (contiguous, no adjacent-K difference): "
                  f"{NAME.get(plat, plat)}")
            print(f"  magnitude: within 10%/5%/1% of FC from K >= "
                  f"{within(med, fc, .10)}/{within(med, fc, .05)}/"
                  f"{within(med, fc, .01)};  "
                  f"T(K=4)/T_FC = {med['rrn_k4'] / fc:.2f}x;  "
                  f"residual at K={KS_DESC[0]}: "
                  f"{(med[f'rrn_k{KS_DESC[0]}'] / fc - 1) * 100:+.2f}%")
            print()


def _inversions(rank_sums):
    """(count, 'a>=b;...') of adjacent columns out of the predicted order."""
    bad = [i for i, d in enumerate(np.diff(rank_sums)) if d <= 0]
    return len(bad), ";".join(f"{TOPOS[i]}>={TOPOS[i + 1]}" for i in bad)


def write_density_tests(tests_fc, tests_adj):
    """density_tests.csv: one row per (statistic, model), plus the pooled row."""
    os.makedirs(CSV_DIR, exist_ok=True)
    path = os.path.join(CSV_DIR, "density_tests.csv")
    ks = KS_DESC
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "stat", "rule", "N", "n_word_pairs", "n_columns",
                    "Q", "df", "p", "kendall_W", "order_correct",
                    "n_inversions", "inversions", "page_L", "page_p",
                    "saturation_K", "plateau_K",
                    "K_within_10pct_FC", "K_within_5pct_FC", "K_within_1pct_FC",
                    "T_FC", "T_K4", "T_K4_over_FC", "T_Kmax_over_FC",
                    "posthoc_family_vs_fc_n", "posthoc_floor_vs_fc",
                    "posthoc_family_adj_n", "posthoc_floor_adj"]
                   + [f"median_T_k{K}" for K in ks]
                   + [f"ranksum_k{K}" for K in ks] + ["ranksum_fc"]
                   + [f"holm_p_k{K}_vs_fc" for K in ks]
                   + [f"holm_p_k{ks[j]}_vs_k{ks[j - 1]}"
                      for j in range(1, len(ks))])
        for stat in STATS:
            for model in ["ALL"] + MODELS:
                t = tests_fc[(stat, model)]
                ta = tests_adj[(stat, model)]
                col = {tp: i for i, tp in enumerate(TOPOS)}
                med = {tp: float(np.median(t["mat"][:, col[tp]])) for tp in TOPOS}
                fc = med["fc"]
                ph = {tuple(sorted(x["pair"])): x for x in t["posthoc"]}
                pha = {tuple(sorted(x["pair"])): x for x in ta["posthoc"]}
                sat, plat = thresholds(t, med, fc)

                def g(d, a, b):
                    x = d.get(tuple(sorted((a, b))))
                    if x is None:
                        return ""
                    return (f"{x['p_holm']:.6g}" if x["decidable"]
                            else f"undecidable({x['p_holm']:.3g})")
                w.writerow(
                    [model, stat, "edge", DENSITY_N, t["n"], t["k"],
                     f"{t['Q']:.4f}", t["df"], f"{t['p']:.6g}", f"{t['W']:.4f}",
                     int(all(np.diff(t["rank_sums"]) > 0)),
                     *_inversions(t["rank_sums"]),
                     f"{t['page_L']:.0f}", f"{t['page_p']:.6g}",
                     sat, plat,
                     within(med, fc, .10) or "", within(med, fc, .05) or "",
                     within(med, fc, .01) or "",
                     f"{fc:.4f}", f"{med['rrn_k4']:.4f}",
                     f"{med['rrn_k4'] / fc:.4f}",
                     f"{med[f'rrn_k{ks[0]}'] / fc:.4f}",
                     t["n_pairs"], f"{t['posthoc_floor']:.6g}",
                     ta["n_pairs"], f"{ta['posthoc_floor']:.6g}"]
                    + [f"{med[f'rrn_k{K}']:.4f}" for K in ks]
                    + [f"{t['rank_sums'][col[f'rrn_k{K}']]:.1f}" for K in ks]
                    + [f"{t['rank_sums'][col['fc']]:.1f}"]
                    + [g(ph, f"rrn_k{K}", "fc") for K in ks]
                    + [g(pha, f"rrn_k{ks[j]}", f"rrn_k{ks[j - 1]}")
                       for j in range(1, len(ks))])
    return path


def write_density_posthoc(tests_fc, tests_adj):
    """density_posthoc.csv: one row per comparison. `decidable` is 0 when the
    comparison could not reach alpha with its number of non-tied blocks."""
    os.makedirs(CSV_DIR, exist_ok=True)
    path = os.path.join(CSV_DIR, "density_posthoc.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "rule", "stat", "N", "family", "pair",
                    "median_diff", "n_zero", "n_eff", "p_raw", "p_holm",
                    "decidable", "significant"])
        for stat in STATS:
            for model in ["ALL"] + MODELS:
                for fam, tt in ((FAM_VS_FC, tests_fc), (FAM_ADJ, tests_adj)):
                    t = tt[(stat, model)]
                    for x in t["posthoc"]:
                        a, b = x["pair"]
                        w.writerow([model, "edge", stat, DENSITY_N, fam,
                                    f"{a}_vs_{b}", f"{x['median_diff']:.4f}",
                                    x["n_zero"], x["n_eff"],
                                    f"{x['p_raw']:.6g}", f"{x['p_holm']:.6g}",
                                    int(x["decidable"]),
                                    int(x["decidable"] and x["p_holm"] < ALPHA)])
    return path


# CSV
def write_csvs(cells, tests):
    f1 = os.path.join(CSV_DIR, f"{CSV_PREFIX}_cells.csv")
    cols = ["rule", "N", "model", "ID", "words", "topo", "R", "mean", "mode",
            "mode_method", "n_unique", "median", "t_min", "t_max", "source",
            "n_censored", "n_unreplayed"]
    with open(f1, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(cells)

    f2 = os.path.join(CSV_DIR, f"{CSV_PREFIX}_ranks.csv")
    with open(f2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "stat", "N", "model", "ID", "words"]
                   + [f"value_{t}" for t in TOPOS] + [f"rank_{t}" for t in TOPOS])
        for key in sorted(tests, key=lambda k: (k[2], k[0], k[1])):
            t = tests[key]
            for (m, i), vals, rks in zip(t["rows"], t["mat"], t["ranks"]):
                w.writerow([t["rule"], t["stat"], t["N"], m, i, words(i)]
                           + [f"{v:.4f}" for v in vals] + [f"{r:.1f}" for r in rks])

    f3 = os.path.join(CSV_DIR, f"{CSV_PREFIX}_tests.csv")
    pair_names = [f"{TOPOS[a]}_vs_{TOPOS[b]}"
                  for a in range(len(TOPOS)) for b in range(a + 1, len(TOPOS))]
    with open(f3, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "stat", "N", "n_cells", "Q", "df", "p", "kendall_W"]
                   + [f"ranksum_{t}" for t in TOPOS]
                   + ["page_L", "page_p"]
                   + [f"holm_p_{p}" for p in pair_names])
        for key in sorted(tests, key=lambda k: (k[2], k[0], k[1])):
            t = tests[key]
            ph = {f"{a}_vs_{b}": v["p_holm"]
                  for v in t["posthoc"] for a, b in [v["pair"]]}
            w.writerow([t["rule"], t["stat"], t["N"], t["n"], f"{t['Q']:.4f}",
                        t["df"], f"{t['p']:.6g}", f"{t['W']:.4f}"]
                       + [f"{r:.1f}" for r in t["rank_sums"]]
                       + [f"{t['page_L']:.0f}", f"{t['page_p']:.6g}"]
                       + [("" if p not in ph else f"{ph[p]:.6g}") for p in pair_names])
    return f1, f2, f3


def main():
    global DENSITY_MODE, CSV_PREFIX, CSV_DIR, TOPOS, NS, RULES

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="topology",
                    choices=["topology", "density"],
                    help="topology: FC / UCM / RRN (default). "
                         "density: RRN degree sweep at N=10000.")
    ap.add_argument("--no-csv", action="store_true", help="print tables only")
    args = ap.parse_args()

    if args.mode == "density":
        DENSITY_MODE = True
        CSV_PREFIX = "density"
        TOPOS = DENSITY_TOPOS
        CSV_DIR = DENSITY_OUT
        NS = [DENSITY_N]
        RULES = ["edge"]           # on a regular graph node == edge
        if not os.path.isdir(DENSITY):
            raise SystemExit(f"{DENSITY}/ not found; run run_density.py first")

    long_files = load_long()
    cells = build_cells(long_files)
    n_merged = sum(1 for c in cells if c["source"] == "merged")
    n_kde = sum(1 for c in cells if c["mode_method"] == "kde")

    src = "data/density + data/grid (fc)" if DENSITY_MODE else "data/grid"
    print(f"loaded {len(cells)} cells "
          f"({n_merged} merged with data/tmax, {len(cells) - n_merged} from "
          f"{src}); 0 censored, 0 unreplayed runs.")
    print(f"mode: {len(cells) - n_kde} empirical, {n_kde} KDE "
          f"(threshold {MODE_EMPIRICAL_MAX_UNIQUE} distinct t)\n")

    print_exclusions(cells)
    print_cells(cells)

    tests = {(rule, stat, N): run_test(cells, rule, stat, N)
             for rule in RULES for stat in STATS for N in NS}

    print_ranks(tests)
    print_summary(tests)
    if not DENSITY_MODE:
        print_posthoc(tests)
    else:
        tests_fc = {(stat, m): run_test(cells, "edge", stat, DENSITY_N,
                                        None if m == "ALL" else m, FAM_VS_FC)
                    for stat in STATS for m in ["ALL"] + MODELS}
        tests_adj = {(stat, m): run_test(cells, "edge", stat, DENSITY_N,
                                         None if m == "ALL" else m, FAM_ADJ)
                     for stat in STATS for m in ["ALL"] + MODELS}
        print_saturation(tests_fc, tests_adj)

    if not args.no_csv:
        os.makedirs(CSV_DIR, exist_ok=True)
        written = list(write_csvs(cells, tests))
        if DENSITY_MODE:
            # replace the generic tests file with the per-model one
            written[-1] = write_density_tests(tests_fc, tests_adj)
            written.append(write_density_posthoc(tests_fc, tests_adj))
        for f in written:
            print(f"wrote {os.path.relpath(f, HERE)}")


if __name__ == "__main__":
    main()
