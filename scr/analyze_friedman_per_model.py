"""Friedman tests on convergence time across topologies, one model at a time.

Same cells and test as analyze_friedman.py, but the blocks are the word pairs of
a single model: 11 per model, 9 for Llama-3.1-70B. 4 models x 2 rules x
2 statistics x 4 population sizes = 64 tests.

    python analyze_friedman_per_model.py [--no-csv]
"""
import argparse
import csv
import os

import numpy as np

import analyze_friedman as af
import paths
from analyze_friedman import (ALPHA, MODELS, NS, RULES, STATS, TOPOS,
                              TOPO_LABEL, fmt_p, words)

HERE = paths.ROOT
OUTDIR = paths.TABLES
MODEL_LABELS = {"llama31_70B": "Llama-3.1-70B", "gpt-4o": "GPT-4o",
                "QWQ_32B": "QwQ-32B", "phi_4": "Phi-4"}
PAIR_NAMES = [f"{TOPOS[a]}_vs_{TOPOS[b]}"
              for a in range(len(TOPOS)) for b in range(a + 1, len(TOPOS))]


def min_wilcoxon_p(n):
    """Smallest two-sided exact Wilcoxon p with n rows: 2 / 2**n."""
    return 2.0 / (2 ** n)


def run_all(cells):
    """{(model, rule, stat, N): test}."""
    out = {}
    for model in MODELS:
        sub = [c for c in cells if c["model"] == model]
        for rule in RULES:
            for stat in STATS:
                for N in NS:
                    t = af.run_test(sub, rule, stat, N)
                    t["model"] = model
                    out[(model, rule, stat, N)] = t
    return out


# Printing
def print_header(cells, tests):
    print("=" * 108)
    print("PER-MODEL FRIEDMAN TESTS (topology within subject, blocks = word pairs)")
    print("=" * 108)
    print(f"\n{len(tests)} tests = {len(MODELS)} models x {len(RULES)} rules "
          f"x {len(STATS)} statistics x {len(NS)} population sizes.\n")
    print(f"  {'model':16s}{'pairs (rows)':>14s}{'excluded':>10s}"
          f"{'min Wilcoxon p':>16s}{'x6 (Holm)':>12s}   post-hoc reachable?")
    for model in MODELS:
        n = tests[(model, RULES[0], STATS[0], NS[0])]["n"]
        exc = sum(1 for (m, _i) in af.EXCLUDED if m == model)
        mp = min_wilcoxon_p(n)
        holm6 = min(1.0, 6 * mp)
        print(f"  {MODEL_LABELS[model]:16s}{n:14d}{exc:10d}"
              f"{mp:16.5f}{holm6:12.4f}   "
              f"{'yes' if holm6 < ALPHA else 'no'}")
    print()


def print_summary(tests):
    print("=" * 108)
    print("SUMMARY, one table per N")
    print("=" * 108)
    for N in NS:
        print(f"\n### N = {N}")
        print(f"{'model':15s}{'rule':6s}{'stat':6s}{'n':>3s}"
              + "".join(f"{'R:' + TOPO_LABEL[t]:>11s}" for t in TOPOS)
              + f"{'Q':>9s}{'p':>11s}{'W':>7s}{'Page L':>9s}{'p(L)':>11s}"
              + "  order")
        for model in MODELS:
            for rule in RULES:
                for stat in STATS:
                    t = tests[(model, rule, stat, N)]
                    rs = t["rank_sums"]
                    ordered = "OK" if np.all(np.diff(rs) > 0) else "BROKEN"
                    print(f"{MODEL_LABELS[model]:15s}{rule:6s}{stat:6s}"
                          f"{t['n']:3d}"
                          + "".join(f"{r:11.1f}" for r in rs)
                          + f"{t['Q']:9.2f}{fmt_p(t['p']):>11s}{t['W']:7.3f}"
                          f"{t['page_L']:9.0f}{fmt_p(t['page_p']):>11s}"
                          f"  {ordered}")


def print_effect_sizes(tests):
    print("\n" + "=" * 108)
    print("KENDALL'S W BY MODEL   (W = Q / (n(k-1)))")
    print("=" * 108)
    for stat in STATS:
        for rule in RULES:
            print(f"\n--- statistic: {stat}, rule: {rule} ---")
            print(f"{'model':15s}" + "".join(f"{'N=' + str(N):>12s}" for N in NS)
                  + f"{'mean W':>10s}")
            for model in MODELS:
                ws = [tests[(model, rule, stat, N)]["W"] for N in NS]
                sig = ["*" if tests[(model, rule, stat, N)]["p"] < ALPHA else " "
                       for N in NS]
                print(f"{MODEL_LABELS[model]:15s}"
                      + "".join(f"{w:11.3f}{s}" for w, s in zip(ws, sig))
                      + f"{np.mean(ws):10.3f}")
            print("  * = Friedman significant at alpha=0.05")


def print_ranks(tests):
    print("\n" + "=" * 108)
    print("RANK MATRICES   (rank within each word pair, 1 = fastest)")
    print("=" * 108)
    for N in NS:
        for model in MODELS:
            for rule in RULES:
                for stat in STATS:
                    t = tests[(model, rule, stat, N)]
                    print(f"\n--- {MODEL_LABELS[model]}  rule={rule}  "
                          f"stat={stat}  N={N}  ({t['n']} rows) ---")
                    print(f"{'words':18s}"
                          + "".join(f"{TOPO_LABEL[tp]:>10s}" for tp in TOPOS))
                    for (_m, i), row in zip(t["rows"], t["ranks"]):
                        print(f"{words(i):18s}"
                              + "".join(f"{r:10.1f}" for r in row))
                    print(f"{'RANK SUM':18s}"
                          + "".join(f"{r:10.1f}" for r in t["rank_sums"])
                          + f"    (total {t['rank_sums'].sum():.0f}, expected "
                            f"{t['n'] * t['k'] * (t['k'] + 1) / 2:.0f})")


def print_posthoc(tests):
    print("\n" + "=" * 108)
    print("POST-HOC pairwise Wilcoxon signed-rank (Holm-Bonferroni over 6 pairs)")
    print("=" * 108)
    for N in NS:
        for model in MODELS:
            for rule in RULES:
                for stat in STATS:
                    t = tests[(model, rule, stat, N)]
                    print(f"\n--- {MODEL_LABELS[model]}  rule={rule}  "
                          f"stat={stat}  N={N} ---")
                    if not t["posthoc"]:
                        print(f"  Friedman not significant "
                              f"(p={fmt_p(t['p'])}); no post-hoc run.")
                        continue
                    print(f"  {'pair':22s}{'med diff':>11s}{'zeros':>7s}"
                          f"{'p raw':>11s}{'p Holm':>11s}   verdict")
                    for ph in t["posthoc"]:
                        a, b = ph["pair"]
                        v = ("differ" if ph["p_holm"] < ALPHA
                             else "NOT distinguishable")
                        print(f"  {TOPO_LABEL[a] + ' vs ' + TOPO_LABEL[b]:22s}"
                              f"{ph['median_diff']:11.2f}{ph['n_zero']:7d}"
                              f"{fmt_p(ph['p_raw']):>11s}"
                              f"{fmt_p(ph['p_holm']):>11s}   {v}")


def print_verdict(tests):
    print("\n" + "=" * 108)
    print("COUNTS over each model's 16 tests")
    print("=" * 108)
    print(f"\n{'model':15s}{'n':>3s}{'Friedman sig':>14s}{'order correct':>15s}"
          f"{'Page sig':>10s}{'W range':>16s}{'6/6 post-hoc':>14s}")
    for model in MODELS:
        ts = [tests[(model, r, s, N)] for r in RULES for s in STATS for N in NS]
        sig = sum(t["p"] < ALPHA for t in ts)
        order = sum(bool(np.all(np.diff(t["rank_sums"]) > 0)) for t in ts)
        page = sum(t["page_p"] < ALPHA for t in ts)
        ws = [t["W"] for t in ts]
        full = sum(bool(t["posthoc"]) and
                   all(p["p_holm"] < ALPHA for p in t["posthoc"]) for t in ts)
        print(f"{MODEL_LABELS[model]:15s}{ts[0]['n']:3d}{sig:>10d}/16"
              f"{order:>11d}/16{page:>6d}/16"
              f"{min(ws):9.3f}-{max(ws):.3f}{full:>10d}/16")


# CSV
def write_csvs(cells, tests):
    os.makedirs(OUTDIR, exist_ok=True)
    written = []

    f = os.path.join(OUTDIR, "friedman_per_model_cells.csv")
    cols = ["rule", "N", "model", "ID", "words", "topo", "R", "mean", "mode",
            "mode_method", "n_unique", "median", "t_min", "t_max", "source",
            "n_censored", "n_unreplayed"]
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(cells)
    written.append(f)

    f = os.path.join(OUTDIR, "friedman_per_model_tests.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "rule", "stat", "N", "n_pairs", "Q", "df", "p",
                    "kendall_W", "order_correct"]
                   + [f"ranksum_{t}" for t in TOPOS]
                   + ["page_L", "page_p"]
                   + [f"holm_p_{p}" for p in PAIR_NAMES])
        for key in sorted(tests, key=lambda k: (k[3], k[0], k[1], k[2])):
            t = tests[key]
            ph = {f"{a}_vs_{b}": v["p_holm"]
                  for v in t["posthoc"] for a, b in [v["pair"]]}
            w.writerow([t["model"], t["rule"], t["stat"], t["N"], t["n"],
                        f"{t['Q']:.4f}", t["df"], f"{t['p']:.6g}",
                        f"{t['W']:.4f}",
                        int(np.all(np.diff(t["rank_sums"]) > 0))]
                       + [f"{r:.1f}" for r in t["rank_sums"]]
                       + [f"{t['page_L']:.0f}", f"{t['page_p']:.6g}"]
                       + [("" if p not in ph else f"{ph[p]:.6g}")
                          for p in PAIR_NAMES])
    written.append(f)

    f = os.path.join(OUTDIR, "friedman_per_model_ranks.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "rule", "stat", "N", "ID", "words"]
                   + [f"value_{t}" for t in TOPOS]
                   + [f"rank_{t}" for t in TOPOS])
        for key in sorted(tests, key=lambda k: (k[3], k[0], k[1], k[2])):
            t = tests[key]
            for (m, i), vals, rks in zip(t["rows"], t["mat"], t["ranks"]):
                w.writerow([m, t["rule"], t["stat"], t["N"], i, words(i)]
                           + [f"{v:.4f}" for v in vals]
                           + [f"{r:.1f}" for r in rks])
    written.append(f)

    f = os.path.join(OUTDIR, "friedman_per_model_posthoc.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "rule", "stat", "N", "pair", "median_diff",
                    "n_zero", "p_raw", "p_holm", "significant"])
        for key in sorted(tests, key=lambda k: (k[3], k[0], k[1], k[2])):
            t = tests[key]
            for ph in t["posthoc"]:
                a, b = ph["pair"]
                w.writerow([t["model"], t["rule"], t["stat"], t["N"],
                            f"{a}_vs_{b}", f"{ph['median_diff']:.4f}",
                            ph["n_zero"], f"{ph['p_raw']:.6g}",
                            f"{ph['p_holm']:.6g}",
                            int(ph["p_holm"] < ALPHA)])
    written.append(f)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-csv", action="store_true", help="print tables only")
    args = ap.parse_args()

    cells = af.build_cells(af.load_long())
    tests = run_all(cells)

    print(f"loaded {len(cells)} cells from data/grid and data/tmax\n")
    print_header(cells, tests)
    print_summary(tests)
    print_effect_sizes(tests)
    print_verdict(tests)
    print_ranks(tests)
    print_posthoc(tests)

    if not args.no_csv:
        for f in write_csvs(cells, tests):
            print(f"\nwrote {os.path.relpath(f, HERE)}")


if __name__ == "__main__":
    main()
