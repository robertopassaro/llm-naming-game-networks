"""
Analysis of the small-N RRN degree sweep (run_density_metastable.py).

    python analyze_density_metastable.py

Writes
  tables/metastable_K_cells.csv   one row per (arm, pair, N, K)
  tables/metastable_K_tests.csv   KS tests: annealed vs FC, K = N-1 vs FC,
                                  stub vs swap generator
  tables/metastable_K_fits.csv    T(4), T_FC, K50, K90 and fits of log T vs K
  figures/T_vs_K.{png,pdf}
"""
import csv
import glob
import os
import pickle

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle as fs
import paths
from run_density_metastable import (IDS, NS, KS, TIME, ITERS,
                                    RUNS_DIR, stub_max)

PAIR = {219: "young/old", 1161: "more/less", 92: "african/white (control)"}
ANALYSIS = paths.TABLES
FIGS = paths.FIGURES
RNG = np.random.default_rng(0)
N_BOOT = 5000


def load_all():
    cells = {}
    for arm in ["quenched", "annealed", "fc", "quenched_swap"]:
        for f in glob.glob(os.path.join(RUNS_DIR, arm, "*.pkl")):
            with open(f, "rb") as fh:
                d = pickle.load(fh)
            K = d["N"] - 1 if arm == "fc" else d["K"]
            cells[(arm, d["ID"], d["N"], K)] = d
    return cells


def summarize(d):
    t = np.array(d["conv_times_all"], dtype=float)
    conv = t >= 0
    tc = t[conv]
    row = {"runs": len(t), "censored": int((~conv).sum()),
           "frac_by_1000": float(np.mean(conv & (t < 1000))),
           "frac_by_tmax": float(conv.mean()),
           "mean_T": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
           "median_T": np.nan, "mean_T_lb": np.nan}
    if len(tc):
        # censored runs counted at t_max: a lower bound on the true mean
        row["mean_T_lb"] = float((tc.sum() + (~conv).sum() * d["T"]) / len(t))
        boot = RNG.choice(tc, size=(N_BOOT, len(tc))).mean(axis=1)
        row.update(mean_T=float(tc.mean()), median_T=float(np.median(tc)),
                   ci_lo=float(np.percentile(boot, 2.5)),
                   ci_hi=float(np.percentile(boot, 97.5)))
    return row, tc


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    cells = load_all()
    rows, times = [], {}
    for key in sorted(cells, key=lambda k: (k[2], IDS.index(k[1]), k[0], k[3])):
        arm, ID, N, K = key
        r, tc = summarize(cells[key])
        times[key] = tc
        rows.append({"arm": arm, "ID": ID, "pair": PAIR[ID], "N": N, "K": K,
                     "T_max": cells[key]["T"], **r})
    fc_mean = {(r["ID"], r["N"]): r["mean_T"] for r in rows if r["arm"] == "fc"}
    for r in rows:
        r["T_over_FC"] = r["mean_T"] / fc_mean.get((r["ID"], r["N"]), np.nan)

    cols = ["arm", "ID", "pair", "N", "K", "T_max", "runs", "censored",
            "frac_by_1000", "frac_by_tmax", "mean_T", "ci_lo", "ci_hi",
            "median_T", "mean_T_lb", "T_over_FC"]
    with open(os.path.join(ANALYSIS, "metastable_K_cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.4g}" if isinstance(r[c], float) else r[c])
                        for c in cols})

    incomplete = [r for r in rows if r["runs"] < ITERS]
    censored = [r for r in rows if r["censored"]]
    ctrl_fail = [r for r in rows if r["ID"] == 92 and r["frac_by_tmax"] < 1]
    print(f"cells: {len(rows)}   incomplete: {len(incomplete)}   "
          f"cells with censoring: {len(censored)}   control failures: {len(ctrl_fail)}")
    for r in censored:
        print(f"   CENSORED {r['arm']} {r['pair']} N{r['N']} K{r['K']}: "
              f"{r['censored']}/{r['runs']}  mean(converged)={r['mean_T']:.0f}  "
              f"lower bound with censored at t_max={r['mean_T_lb']:.0f} "
              f"(+{100 * (r['mean_T_lb'] / r['mean_T'] - 1):.1f}%)")

    # KS tests
    tests = []

    def ks(label, a, b, ID, N, K):
        if len(a) and len(b):
            s, p = stats.ks_2samp(a, b)
            tests.append({"test": label, "ID": ID, "pair": PAIR[ID], "N": N,
                          "K": K, "n_a": len(a), "n_b": len(b),
                          "mean_a": a.mean(), "mean_b": b.mean(),
                          "KS_D": s, "p": p})

    for N in NS:
        for ID in IDS:
            fc = times.get(("fc", ID, N, N - 1), np.array([]))
            ks("quenched K=N-1 vs FC", times.get(("quenched", ID, N, N - 1),
                                                 np.array([])), fc, ID, N, N - 1)
            for K in KS[N]:
                ks("annealed vs FC", times.get(("annealed", ID, N, K),
                                               np.array([])), fc, ID, N, K)
            Ks = stub_max(N)
            ks("stub vs swap generator", times.get(("quenched", ID, N, Ks), np.array([])),
               times.get(("quenched_swap", ID, N, Ks), np.array([])), ID, N, Ks)
    # Holm within each test family
    for fam in {t["test"] for t in tests}:
        ft = sorted([t for t in tests if t["test"] == fam], key=lambda t: t["p"])
        m, run = len(ft), 0.0
        for i, t in enumerate(ft):
            run = max(run, min(1.0, (m - i) * t["p"]))
            t["p_holm"] = run
    tcols = ["test", "ID", "pair", "N", "K", "n_a", "n_b", "mean_a", "mean_b",
             "KS_D", "p", "p_holm"]
    with open(os.path.join(ANALYSIS, "metastable_K_tests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tcols)
        w.writeheader()
        for t in tests:
            w.writerow({c: (f"{t[c]:.4g}" if isinstance(t[c], float) else t[c])
                        for c in tcols})
    print("\nKS tests (Holm within family):")
    for fam in sorted({t["test"] for t in tests}):
        ft = [t for t in tests if t["test"] == fam]
        nsig = sum(t["p_holm"] < 0.05 for t in ft)
        print(f"  {fam:26s} {len(ft):3d} tests, {nsig} significant after Holm, "
              f"min raw p = {min(t['p'] for t in ft):.3g}")
        for t in ft:
            if t["p_holm"] < 0.05:
                print(f"     SIG {t['pair']} N{t['N']} K{t['K']}: "
                      f"{t['mean_a']:.1f} vs {t['mean_b']:.1f}, p_holm={t['p_holm']:.3g}")

    # fits on the quenched arm
    fits = []
    for N in NS:
        for ID in IDS[:2]:
            q = sorted([r for r in rows if r["arm"] == "quenched" and r["ID"] == ID
                        and r["N"] == N], key=lambda r: r["K"])
            K = np.array([r["K"] for r in q], float)
            T = np.array([r["mean_T"] for r in q], float)
            Tfc = fc_mean[(ID, N)]
            # progress on a log scale from T(4) to T_FC
            prog = (np.log(T) - np.log(T[0])) / (np.log(Tfc) - np.log(T[0]))
            k50 = float(np.interp(0.5, np.maximum.accumulate(prog), K))
            k90 = float(np.interp(0.9, np.maximum.accumulate(prog), K))
            # rising part: K up to (and including) the first K past 90% progress
            stop = int(np.argmax(prog >= 0.9)) + 1 if (prog >= 0.9).any() else len(K)
            stop = max(stop, 3)
            e = stats.linregress(K[:stop], np.log(T[:stop]))
            p = stats.linregress(np.log(K[:stop]), np.log(T[:stop]))
            fits.append({"ID": ID, "pair": PAIR[ID], "N": N, "T_K4": T[0],
                         "T_FC": Tfc, "T_Kmax": T[-1], "FC_over_K4": Tfc / T[0],
                         "K50": k50, "K90": k90, "fit_Kmax": int(K[stop - 1]),
                         "exp_slope": e.slope, "exp_R2": e.rvalue ** 2,
                         "pow_exponent": p.slope, "pow_R2": p.rvalue ** 2,
                         "monotone": bool(np.all(np.diff(T) > -1e-9)),
                         "n_decreases": int(np.sum(np.diff(T) < 0))})
    fcols = list(fits[0].keys())
    with open(os.path.join(ANALYSIS, "metastable_K_fits.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fcols)
        w.writeheader()
        for r in fits:
            w.writerow({c: (f"{r[c]:.4g}" if isinstance(r[c], float) else r[c])
                        for c in fcols})

    # printed tables
    for N in NS:
        print(f"\n=== N = {N}  (t_max = {TIME[N]:,} rounds, {ITERS} runs/cell) ===")
        for ID in IDS:
            fcr = next(r for r in rows if r["arm"] == "fc" and r["ID"] == ID and r["N"] == N)
            print(f"\n{PAIR[ID]}   FC: mean T = {fcr['mean_T']:.1f} "
                  f"[{fcr['ci_lo']:.1f}, {fcr['ci_hi']:.1f}], median {fcr['median_T']:.0f}, "
                  f"frac by t=1000 {fcr['frac_by_1000']:.2f}")
            print(f"{'K':>5} | {'quenched mean [95% CI]':>28} {'med':>7} {'f@1000':>6} "
                  f"{'T/T_FC':>6} | {'annealed mean':>13} {'f@1000':>6} {'T/T_FC':>6}")
            for K in KS[N]:
                q = next((r for r in rows if r["arm"] == "quenched" and r["ID"] == ID
                          and r["N"] == N and r["K"] == K), None)
                a = next((r for r in rows if r["arm"] == "annealed" and r["ID"] == ID
                          and r["N"] == N and r["K"] == K), None)
                if q is None or a is None:
                    continue
                print(f"{K:>5} | {q['mean_T']:>10.1f} [{q['ci_lo']:>7.1f},{q['ci_hi']:>8.1f}] "
                      f"{q['median_T']:>7.0f} {q['frac_by_1000']:>6.2f} {q['T_over_FC']:>6.3f} | "
                      f"{a['mean_T']:>13.1f} {a['frac_by_1000']:>6.2f} {a['T_over_FC']:>6.3f}")
    print("\nFits on the quenched arm (rising part):")
    for r in fits:
        print(f"  {r['pair']:10s} N{r['N']}: T(4)={r['T_K4']:.1f}  T_FC={r['T_FC']:.1f} "
              f"(x{r['FC_over_K4']:.1f})  K50={r['K50']:.1f} K90={r['K90']:.1f}  "
              f"fit K<= {r['fit_Kmax']}: exp R2={r['exp_R2']:.3f} "
              f"(slope {r['exp_slope']:.4f}/K)  power R2={r['pow_R2']:.3f} "
              f"(exp {r['pow_exponent']:.2f})  decreases={r['n_decreases']}")

    plot(rows, fc_mean)


def plot(rows, fc_mean):
    """Mean convergence time against K, quenched and annealed, with the FC
    mean and its interval, for the two non-converging pairs (Figure A4)."""
    fs.apply_style()
    pairs = [219, 1161]
    fig, axes = plt.subplots(len(NS), len(pairs), figsize=(8.5, 3.6 * len(NS)),
                             sharey=True, squeeze=False)
    qc, ac = fs.TOPO_COLORS["rrn"], fs.TOPO_COLORS["ucm"]
    for i, N in enumerate(NS):
        for j, ID in enumerate(pairs):
            ax = axes[i, j]
            fcr = next(r for r in rows if r["arm"] == "fc" and r["ID"] == ID and r["N"] == N)
            ax.axhspan(fcr["ci_lo"], fcr["ci_hi"], color="0.85", lw=0, zorder=0)
            ax.axhline(fcr["mean_T"], color=fs.TOPO_COLORS["fc"], ls="--", lw=1.2,
                       label="FC (well-mixed)", zorder=1)
            for arm, c, mk, mfc, lab in [("quenched", qc, "o", qc, "RRN quenched"),
                                         ("annealed", ac, "s", "white", "RRN annealed")]:
                rr = sorted([r for r in rows if r["arm"] == arm and r["ID"] == ID
                             and r["N"] == N], key=lambda r: r["K"])
                K = np.array([r["K"] for r in rr])
                m = np.array([r["mean_T"] for r in rr])
                lo = m - np.array([r["ci_lo"] for r in rr])
                hi = np.array([r["ci_hi"] for r in rr]) - m
                ax.errorbar(K, m, yerr=[lo, hi], color=c, marker=mk, ms=5.5,
                            mfc=mfc, mew=1.3, lw=1.5, capsize=2, label=lab, zorder=3)
            ax.set_yscale("log")
            ax.set_ylim(20, 5e5)
            ax.set_xlim(0, N + 2)
            if i == len(NS) - 1:
                ax.set_xlabel("degree $K$")
            if j == 0:
                ax.set_ylabel(f"$N={N}$\nmean convergence time (rounds)")
            if i == 0:
                ax.set_title(PAIR[ID])
            ax.grid(True, which="major", color="0.9", lw=0.6)
            if i == 0 and j == 0:
                ax.legend(loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"T_vs_K.{ext}"))
    print(f"\nfigure: {os.path.join(FIGS, 'T_vs_K.png')}")


if __name__ == "__main__":
    main()
