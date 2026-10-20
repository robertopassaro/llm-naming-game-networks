"""Numbers quoted in the text that are not printed by the other scripts.

Reads data/grid and data/tmax, and tables/isosuccess_*.csv (run
plot_isosuccess.py first).

    python paper_numbers.py
"""
import csv
import glob
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import spearmanr, wilcoxon

import paths

MODELS = ["llama31_70B", "gpt-4o", "QWQ_32B", "phi_4"]
IDS = [92, 102, 191, 219, 344, 349, 397, 410, 505, 626, 1161]
TOPOS = ["fc", "ucm_2.5", "ucm_5.0", "rrn"]
NS = [1000, 5000, 10000, 100000]
NON_CONVERGING = {("llama31_70B", 219), ("llama31_70B", 1161)}
BINS = [(4, 5), (6, 9), (10, 17), (18, 35), (36, 100)]

with open(paths.WORD_PAIRS, "rb") as f:
    WORDS = {i: d["differences"] for i, d in pickle.load(f).items() if i in IDS}


def pair(ID):
    return "/".join(WORDS[ID])


def _load(p):
    with open(p, "rb") as f:
        return p, pickle.load(f)


def load_dir(pattern):
    with ThreadPoolExecutor(32) as ex:
        return dict(ex.map(_load, sorted(glob.glob(pattern))))


def load_rq1():
    out = {}
    rx = re.compile(r"RQ1_(.+)_ID_(\d+)_(fc|rrn|ucm_[0-9.]+)_N(\d+)_(edge|node|annealed)_H5\.pkl")
    for p, d in load_dir(os.path.join(paths.GRID, "RQ1_*.pkl")).items():
        m = rx.match(os.path.basename(p))
        out[(m[1], int(m[2]), m[3], int(m[4]), m[5])] = d
    return out


def load_tmax():
    return {(d["model"], d["ID"], d["topo"], d["N"], d["rule"]): d
            for d in load_dir(os.path.join(paths.TMAX, "RQ1LONG_*.pkl")).values()}


def load_rq2():
    out = {}
    rx = re.compile(r"RQ2_(.+)_ID_(\d+)_(fc|rrn|ucm_[0-9.]+)_N10000_(random|hub|leaf)_dir(\d)_edge_H5\.pkl")
    for p, d in load_dir(os.path.join(paths.GRID, "RQ2_*_N10000_*.pkl")).items():
        m = rx.match(os.path.basename(p))
        out[(m[1], int(m[2]), m[3], m[4], int(m[5]))] = d
    return out


def run_times(key, grid, tmax):
    """Per-run convergence times, -1 where a run did not converge."""
    if key in tmax:
        return np.array([-1 if t is None else t for t in tmax[key]["conv_times_merged"]])
    return np.array(grid[key]["conv_times_all"])


def mode(t):
    vals, counts = np.unique(t, return_counts=True)
    return vals[np.argmax(counts)]


def header(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


# --------------------------------------------------------------------------
def degrees(G):
    header("Degree statistics (Table 1 and the degree bins)")
    for topo in ["ucm_2.5", "ucm_5.0", "rrn"]:
        for N in NS:
            h = G[("gpt-4o", 92, topo, N, "edge")]["deg_hist_sum"].astype(float)
            k = np.arange(len(h))
            P = h / h.sum()
            k1, k2 = (k * P).sum(), (k * k * P).sum()
            print(f"  {topo:8s} N={N:<6d} <k> = {k1:5.2f}   <k^2>/<k> = {k2 / k1:6.2f}"
                  f"   share with k <= 5 = {P[k <= 5].sum():.3f}")
    h = G[("gpt-4o", 92, "ucm_2.5", 10000, "edge")]["deg_hist_sum"].astype(float)
    k = np.arange(len(h))
    kb = []
    print("\n  UCM 2.5, N = 10000")
    for a, b in BINS:
        m = (k >= a) & (k <= b)
        kb.append((k[m] * h[m]).sum() / h[m].sum())
        print(f"    k = {a:>2d}-{b:<3d}  share {h[m].sum() / h.sum():.3f}   mean k {kb[-1]:.2f}")
    print(f"    k_hub / k_low = {kb[-1] / kb[0]:.2f}")


def consensus(G, T):
    header("Consensus in every run and the winning word (section 3.1)")
    full = []
    for model in MODELS:
        for ID in IDS:
            ok = all((run_times((model, ID, topo, N, rule), G, T) >= 0).all()
                     for topo in TOPOS for N in NS
                     for rule in (["edge"] if topo == "fc" else ["edge", "node"]))
            if ok:
                full.append((model, ID))
    print(f"  model x pair combinations converging in every run: {len(full)}")

    def share0(d):
        n0, n1 = d["consensus"][0], d["consensus"][1]
        return n0 / (n0 + n1)

    changed = []
    for model, ID in full:
        winners = {np.sign(share0(G[(model, ID, topo, N, rule)]) - 0.5)
                   for topo in TOPOS for N in NS
                   for rule in (["edge"] if topo == "fc" else ["edge", "node"])}
        if len(winners) > 1:
            changed.append((model, ID))
    print(f"  same majority word everywhere: {len(full) - len(changed)}")
    for model, ID in changed:
        print(f"\n  {model} {pair(ID)}: share of runs won by '{WORDS[ID][0]}'")
        for topo in TOPOS:
            for rule in (["edge"] if topo == "fc" else ["edge", "node"]):
                row = "  ".join(f"{share0(G[(model, ID, topo, N, rule)]):.3f}" for N in NS)
                print(f"    {topo:8s} {rule:5s} {row}")


def metastable(G, T):
    header("Llama {young/old} and {more/less} (Tables 2 and A3), edge rule")
    for ID in (219, 1161):
        for topo in TOPOS:
            cf, med = [], []
            for N in NS:
                t = np.array(G[("llama31_70B", ID, topo, N, "edge")]["conv_times_all"])
                cf.append(f"{(t >= 0).mean():.2f}")
                med.append(f"{np.median(t[t >= 0]):.1f}" if (t >= 0).any() else "-")
            print(f"  {pair(ID):10s} {topo:8s} converged {' '.join(cf)}   median {' '.join(med)}")
    print("\n  replays with t_max = 1e5 (1e4 at N = 1e5) of runs that failed within 1000 rounds")
    for key, d in sorted(T.items()):
        if key[0] == "llama31_70B" and key[1] in (219, 1161) and d["n_escaped"]:
            lc = np.array([v for v in d["long_conv"].values() if v >= 0])
            print(f"    {pair(key[1]):10s} {key[2]:8s} N={key[3]:<6d} {key[4]:8s} "
                  f"{d['n_escaped']}/{d['R_rerun']} replayed runs converge "
                  f"(median {np.median(lc):.0f} rounds; {d['n_fail_orig']} failed originally)")


def llama_modes(G, T):
    header("Llama mode convergence time per pair, N = 10000, edge rule")
    print(f"  {'pair':18s}" + "".join(f"{t:>9s}" for t in TOPOS))
    for ID in IDS:
        if ("llama31_70B", ID) in NON_CONVERGING:
            continue
        row = [mode(run_times(("llama31_70B", ID, t, 10000, "edge"), G, T)) for t in TOPOS]
        print(f"  {pair(ID):18s}" + "".join(f"{v:9.0f}" for v in row))


def annealed_speed(G, T):
    header("Annealed vs quenched convergence time, UCM 2.5, N = 10000")
    mq, ma = [], []
    for model in MODELS:
        for ID in IDS:
            tq = run_times((model, ID, "ucm_2.5", 10000, "edge"), G, T)
            ta = run_times((model, ID, "ucm_2.5", 10000, "annealed"), G, T)
            if (tq < 0).any() or (ta < 0).any():
                continue
            mq.append(np.median(tq))
            ma.append(np.median(ta))
    mq, ma = np.array(mq), np.array(ma)
    rel = ma / mq - 1
    print(f"  conditions: {len(mq)}; annealed faster {np.sum(ma < mq)}, "
          f"equal {np.sum(ma == mq)}, slower {np.sum(ma > mq)}")
    print(f"  median change {np.median(rel):+.1%} (range {rel.min():+.1%} to {rel.max():+.1%})")
    w = wilcoxon(np.log(ma), np.log(mq), zero_method="pratt")
    print(f"  Wilcoxon signed-rank on log medians (pratt): p = {w.pvalue:.1e}")
    r = spearmanr(mq, rel)
    print(f"  Spearman(quenched median, change): rho = {r.statistic:.2f}, p = {r.pvalue:.1e}")
    o = np.argsort(mq)[-10:]
    print(f"  ten slowest (quenched median {mq[o].min():.0f}-{mq[o].max():.0f} rounds): "
          f"median change {np.median(rel[o]):+.1%}")


def tilts():
    header("Normalised tilt tau (section 3.1.2 and Table A5)")
    cont = os.path.join(paths.TABLES, "isosuccess_contours.csv")
    cells = os.path.join(paths.TABLES, "isosuccess_cells.csv")
    if not os.path.exists(cont):
        print("  tables/isosuccess_contours.csv not found; run plot_isosuccess.py")
        return
    tau, tstar = {}, {}
    for r in csv.DictReader(open(cont)):
        if r["tau"] != "nan":
            tau[(r["model"], int(r["ID"]), r["rule"], float(r["level"]))] = float(r["tau"])
    for r in csv.DictReader(open(cells)):
        tstar[(r["model"], int(r["ID"]), r["rule"], float(r["level"]), int(r["bin"]))] = float(r["t_star"])
    q = {k: v for k, v in tau.items() if k[2] == "edge"}
    v = np.array(list(q.values()))
    print(f"  quenched contours: {len(v)}; median {np.median(v):.2f} "
          f"(IQR {np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f}); "
          f"tau < 0.5 in {np.sum(v < 0.5)}, tau > 0 in {np.sum(v > 0)}")
    g = [x for k, x in q.items() if k[0] == "gpt-4o"]
    print(f"  GPT-4o range {min(g):.3f} to {max(g):.2f}")
    print(f"  hub / low-degree interactions at the median tau: "
          f"{12.56 ** (1 - np.median(v)):.1f}")
    for k, x in sorted(q.items()):
        if x >= 0.5:
            print(f"  tau >= 0.5: {k[0]} {pair(k[1])} S={k[3]} tau={x:.2f}")
    print("\n  negative tau (S = 0.6), with tau at S = 0.8 and the annealed delay of t*")
    for k, x in sorted(q.items()):
        if x < 0:
            m, ID, _, L = k
            dlo = tstar[(m, ID, "annealed", L, 0)] / tstar[(m, ID, "edge", L, 0)] - 1
            dhi = tstar[(m, ID, "annealed", L, 4)] / tstar[(m, ID, "edge", L, 4)] - 1
            print(f"    {m:12s} {pair(ID):14s} S={L}  tau={x:+.3f}  "
                  f"tau(0.8)={q[(m, ID, 'edge', 0.8)]:.2f}  low {dlo:+.0%}  hub {dhi:+.0%}")
    for L in (0.6, 0.8):
        d, lo, hi = [], [], []
        for (m, ID, rule, LL), x in q.items():
            if LL != L or (m, ID, "annealed", L) not in tau:
                continue
            d.append(tau[(m, ID, "annealed", L)] - x)
            lo.append(tstar[(m, ID, "annealed", L, 0)] / tstar[(m, ID, "edge", L, 0)] - 1)
            hi.append(tstar[(m, ID, "annealed", L, 4)] / tstar[(m, ID, "edge", L, 4)] - 1)
        d = np.array(d)
        print(f"\n  S = {L}: annealed - quenched tau over {len(d)} contours: median {np.median(d):.3f} "
              f"(IQR {np.percentile(d, 25):.3f} to {np.percentile(d, 75):.3f}), positive in {np.sum(d > 0)}")
        print(f"          median annealed delay of t*: lowest bin {np.median(lo):+.0%}, "
              f"highest bin {np.median(hi):+.0%}")


def committed(G, P):
    header("Committed minorities, N = 10000 (section 3.2, Tables 3, 4 and A8)")
    ps = {k: d["p_star"] for k, d in P.items()}
    strong = {}
    for model in MODELS:
        for ID in IDS:
            d = G[(model, ID, "fc", 100000, "edge")]
            n0, n1 = d["consensus"][0], d["consensus"][1]
            if n0 + n1 == 0:
                n0, n1 = d["non_consensus"][0]["count"], d["non_consensus"][1]["count"]
            strong[(model, ID)] = 1 if n1 >= n0 else 0

    def cw(model, ID, direction):
        """Committed word that overturns the strong (or weak) convention."""
        s = strong[(model, ID)]
        return 1 - s if direction == "strong" else s

    print("  placement, both directions pooled (88 values):")
    for topo in ["ucm_2.5", "ucm_5.0"]:
        for pl in ["hub", "random", "leaf"]:
            v = np.array([ps[(m, i, topo, pl, w)] for m in MODELS for i in IDS for w in (0, 1)])
            print(f"    {topo} {pl:6s} mean {v.mean():.3f}  median {np.median(v):.3f}")

    bad, gap25, gap5 = 0, [], []
    for m in MODELS:
        for i in IDS:
            for w in (0, 1):
                for topo in ["ucm_2.5", "ucm_5.0"]:
                    h, r, l = (ps[(m, i, topo, pl, w)] for pl in ["hub", "random", "leaf"])
                    bad += not (h <= r <= l)
                gap25.append(ps[(m, i, "ucm_2.5", "leaf", w)] - ps[(m, i, "ucm_2.5", "hub", w)])
                gap5.append(ps[(m, i, "ucm_5.0", "leaf", w)] - ps[(m, i, "ucm_5.0", "hub", w)])
    g = [(a, b) for a, b in zip(gap25, gap5) if a or b]
    print(f"  hub <= random <= leaf violated in {bad} of 176 cells")
    print(f"  leaf - hub gap larger on UCM 2.5 than on UCM 5.0 in "
          f"{sum(a > b for a, b in g)} of {len(g)} pairs")

    zero = sorted(k for k, v in ps.items() if v == 0)
    print(f"\n  p* = 0 in {len(zero)} conditions:")
    for m, i, topo, pl, w in zero:
        print(f"    {m} {pair(i)} {topo} {pl}, established word '{WORDS[i][1 - w]}'")

    # Committed agents count in the success rate. With the other agents still
    # on the old word, success = phi^2 + (1 - phi)^2, phi being the committed
    # share of edge endpoints, which reaches 0.98 at phi = 0.9899.
    phi_c = 0.5 + 0.5 * np.sqrt(0.96)
    high = sorted(((k, v) for k, v in ps.items() if v is not None and v >= 0.95),
                  key=lambda kv: -kv[1])
    print(f"\n  p* >= 0.95 in {len(high)} conditions, {sum(v == 1 for _, v in high)} at 1.00. "
          f"phi = committed share of edge endpoints; phi >= {phi_c:.4f} meets the "
          f"convergence criterion without converting anyone.")
    for (m, i, topo, pl, w), v in high:
        mstar = P[(m, i, topo, pl, w)]["m_star"]
        if topo == "fc" or pl == "random":
            phi = mstar / 10000
        else:
            h = G[("gpt-4o", 92, topo, 10000, "edge")]["deg_hist_sum"].astype(float)
            seq = np.sort(np.repeat(np.arange(len(h)), np.round(h / h.sum() * 10000).astype(int)))
            sel = seq[-mstar:] if pl == "hub" else seq[:mstar]
            phi = sel.sum() / seq.sum()
        print(f"    {m:12s} {pair(i):14s} {topo:8s} {pl:6s} p* = {v:.4f}  phi = {phi:.4f}"
              f"{'  (criterion met by committed agents)' if phi >= phi_c else ''}")

    print("\n  overturning the strong convention costs more than the weak one (random placement):")
    rnd = lambda x: round(round(x / 0.01) * 0.01, 2)
    for topo in ["fc", "ucm_2.5", "ucm_5.0", "rrn"]:
        a = [(ps[(m, i, topo, "random", cw(m, i, "strong"))], ps[(m, i, topo, "random", cw(m, i, "weak"))])
             for m in MODELS for i in IDS]
        gt = sum(rnd(x) > rnd(y) for x, y in a)
        eq = sum(rnd(x) == rnd(y) for x, y in a)
        print(f"    {topo:8s} {gt} of 44 pairs ({eq} ties at the 0.01 resolution)")

    print("\n  median p* under random placement, p* = 0 excluded (Table 4):")
    print(f"    {'':12s} {'':6s}" + "".join(f"{t:>9s}" for t in ["rrn", "ucm_5.0", "ucm_2.5", "fc"]))
    for m in MODELS:
        for d in ("strong", "weak"):
            row = []
            for topo in ["rrn", "ucm_5.0", "ucm_2.5", "fc"]:
                v = [ps[(m, i, topo, "random", cw(m, i, d))] for i in IDS]
                row.append(np.median([x for x in v if x > 0]))
            print(f"    {m:12s} {d:6s}" + "".join(f"{x:9.3f}" for x in row))

    print("\n  Llama, weak direction: committed agents m* in each of the 8 conditions")
    for i in IDS:
        w = cw("llama31_70B", i, "weak")
        ms = [P[("llama31_70B", i, t, pl, w)]["m_star"]
              for t, pl in [("ucm_2.5", "hub"), ("ucm_2.5", "random"), ("ucm_2.5", "leaf"),
                            ("ucm_5.0", "hub"), ("ucm_5.0", "random"), ("ucm_5.0", "leaf"),
                            ("rrn", "random"), ("fc", "random")]]
        print(f"    {pair(i):18s} {ms}")


def main():
    G = load_rq1()
    T = load_tmax()
    P = load_rq2()
    degrees(G)
    consensus(G, T)
    metastable(G, T)
    llama_modes(G, T)
    annealed_speed(G, T)
    tilts()
    committed(G, P)


if __name__ == "__main__":
    main()
