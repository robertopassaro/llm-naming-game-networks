"""Mean convergence time against N for the two Llama pairs that do not converge
in large well-mixed populations, {young, old} (219) and {more, less} (1161), on
FC, UCM (gamma = 2.5 and 5.0) and RRN (K = 4), edge rule.

Each series climbs the N ladder with 200 runs per point, a fresh connected graph
per run and t_max = 1e4 rounds. It stops at the first N where a run does not
converge; that point is stored but not plotted. UCM starts at N = 25, the first
size with sqrt(N) >= kmin = 4.

The kernel is run_grid._run_rq1 (rules 0 and 2) without the per-round curves
and the degree counters, which would need arrays of length t_max. Those parts
draw no random numbers, so each run is identical to the full kernel's.

  python run_metastable_N.py run [--max-processes P]
  python run_metastable_N.py plotlog
"""
import argparse
import glob
import multiprocessing as mp
import os
import pickle

import numpy as np

import paths

RUNS_DIR = paths.METASTABLE_N

MODEL = "llama31_70B"
IDS = [219, 1161]
PAIR = {219: "young/old", 1161: "more/less"}
TOPOS = ["fc", "ucm_2.5", "ucm_5.0", "rrn"]
_LADDER = [10, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400,
           500, 600, 700, 800, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000]
N_MAX = {219: 1000, 1161: 5000}
LADDER = {(topo, ID): [N for N in _LADDER
                       if (not topo.startswith("ucm") or N >= 25) and N <= N_MAX[ID]]
          for topo in TOPOS for ID in IDS}
ITERS = 200
T_MAX = 10_000


_LEAN = None


def _lean_kernel():
    global _LEAN
    if _LEAN is None:
        from numba import njit

        @njit(cache=True)
        def _lean_rq1(rule, edges, q_s, P_key, empty_state, N, T, seed):
            np.random.seed(seed)
            M = edges.shape[0]
            pop = np.full(N, empty_state, dtype=np.int64)
            s0 = s1 = s2 = 0.0
            f0 = f1 = f2 = 0.0
            for t in range(T):
                success = 0
                all_words = 0
                for _ in range(N):
                    if rule == 0:
                        e = np.random.randint(0, M)
                        i = edges[e, 0]
                        j = edges[e, 1]
                    else:
                        i = np.random.randint(0, N)
                        j = np.random.randint(0, N)
                        while j == i:
                            j = np.random.randint(0, N)
                    first = 1 - int(np.random.random() < q_s[pop[i]])
                    second = 1 - int(np.random.random() < q_s[pop[j]])
                    si = pop[i]
                    sj = pop[j]
                    pop[i] = P_key[si, sj, 2 * first + second]
                    pop[j] = P_key[sj, si, 2 * second + first]
                    if first == second:
                        success += 1
                    all_words += first + second
                s0, s1, s2 = s1, s2, success / N
                f0, f1, f2 = f1, f2, all_words / (2 * N)
                if t >= 3 and (s0 + s1 + s2) / 3.0 >= 0.98:
                    return (1 if (f0 + f1 + f2) / 3.0 >= 0.5 else 0), t, True
            return (1 if (f0 + f1 + f2) / 3.0 >= 0.5 else 0), T - 1, False

        _LEAN = _lean_rq1
    return _LEAN


_AGENT = {}


def _agent(ID):
    if ID not in _AGENT:
        from run_grid import build_agent
        _AGENT[ID] = build_agent(MODEL, ID)
    return _AGENT[ID]


def _connected(edges, N):
    adj = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = np.zeros(N, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        for v in adj[stack.pop()]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return seen.all()


def graph(topo, N, it):
    """A connected graph for run `it`; disconnected draws are redrawn with the next seed."""
    from networks import generate_rrn, generate_ucm
    from run_grid import _stable_seed
    for attempt in range(1000):
        seed = _stable_seed("Nsweep", topo, N, it, attempt)
        if topo == "rrn":
            edges, _ = generate_rrn(N, 4, seed=seed)
        else:
            edges, _ = generate_ucm(N, float(topo.split("_")[1]), kmin=4, seed=seed)
        if _connected(edges, N):
            return edges.astype(np.int64), attempt
    raise RuntimeError(f"no connected graph {topo} N={N}")


def run_seed(topo, ID, N):
    from run_grid import _stable_seed
    return _stable_seed(MODEL, ID, topo, N, "edge", "Nsweep")


def run_one(args):
    topo, ID, N, it = args
    q_s, P_key, empty_state, _, _ = _agent(ID)
    if topo == "fc":
        edges, redraws, rule = np.zeros((1, 2), dtype=np.int64), 0, 2
    else:
        (edges, redraws), rule = graph(topo, N, it), 0
    state, ctime, conv = _lean_kernel()(rule, edges, q_s, P_key, empty_state,
                                        N, T_MAX, run_seed(topo, ID, N) + it)
    return int(state), int(ctime) if conv else -1, redraws


def out_path(topo, ID, N):
    return os.path.join(RUNS_DIR, f"RQ1_{MODEL}_ID_{ID}_{topo}_N{N}_edge_H5_Nsweep.pkl")


def _load(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp_{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, path)


def run_series(pool, topo, ID):
    for N in LADDER[(topo, ID)]:
        path = out_path(topo, ID, N)
        d = _load(path)
        if d is None or len(d["conv_times_all"]) < ITERS:
            res = pool.map(run_one, [(topo, ID, N, it) for it in range(ITERS)],
                           chunksize=1)
            d = {"model": MODEL, "ID": ID, "topo": topo, "N": N, "rule": "edge",
                 "T": T_MAX, "H": 5, "base_seed": run_seed(topo, ID, N),
                 "final_states": [r[0] for r in res],
                 "conv_times_all": [r[1] for r in res],
                 "graph_redraws": [r[2] for r in res]}
            _save(path, d)
        t = np.array(d["conv_times_all"])
        ncens = int((t < 0).sum())
        print(f"done  {topo:8s} ID{ID} N={N:5d}: {len(t)} runs, not converged {ncens}, "
              f"mean T={t[t >= 0].mean() if (t >= 0).any() else float('nan'):.4g}",
              flush=True)
        if ncens:
            print(f"stop  {topo:8s} ID{ID} at N={N}", flush=True)
            return


def run(max_processes):
    with mp.Pool(max_processes) as pool:
        for ID in IDS:
            for topo in TOPOS:
                run_series(pool, topo, ID)


def summarize():
    """Mean time and 95% bootstrap interval per (topo, pair, N)."""
    rows = []
    rng = np.random.default_rng(0)
    for f in sorted(glob.glob(os.path.join(RUNS_DIR, "*.pkl"))):
        d = _load(f)
        t = np.array(d["conv_times_all"], dtype=float)
        tc = t[t >= 0]
        if (not len(tc) or len(t) < ITERS
                or d["N"] not in LADDER.get((d["topo"], d["ID"]), ())):
            continue
        boot = rng.choice(tc, size=(5000, len(tc))).mean(axis=1)
        rows.append({"topo": d["topo"], "ID": d["ID"], "N": d["N"], "runs": len(t),
                     "censored": int((t < 0).sum()), "T_max": d["T"],
                     "mean_T": tc.mean(), "ci_lo": np.percentile(boot, 2.5),
                     "ci_hi": np.percentile(boot, 97.5), "median_T": np.median(tc)})
    rows.sort(key=lambda r: (r["ID"], r["topo"], r["N"]))
    return rows


def grid_extension(ID, topo, Ns=(2000, 5000, 10000, 100000)):
    """Points from data/grid (edge rule, t_max = 1000) beyond the ladder, used
    only where every run converged."""
    rng = np.random.default_rng(0)
    out = []
    for N in Ns:
        path = os.path.join(paths.GRID, f"RQ1_{MODEL}_ID_{ID}_{topo}_N{N}_edge_H5.pkl")
        d = _load(path)
        if d is None:
            continue
        t = np.array(d["conv_times_all"], dtype=float)
        if (t < 0).any():
            continue
        boot = rng.choice(t, size=(5000, len(t))).mean(axis=1)
        out.append((N, t.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)))
    return out


def plot_log():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle as fs

    rows = summarize()
    fs.apply_style()
    style = {"fc": (fs.TOPO_COLORS["fc"], "D", fs.TOPO_COLORS["fc"], "FC (well-mixed)"),
             "ucm_2.5": (fs.TOPO_COLORS["ucm"], "s", fs.TOPO_COLORS["ucm"], r"UCM ($\gamma$=2.5)"),
             "ucm_5.0": (fs.TOPO_COLORS["ucm5"], "s", "white", r"UCM ($\gamma$=5.0)"),
             "rrn": (fs.TOPO_COLORS["rrn"], "o", fs.TOPO_COLORS["rrn"], "RRN ($K$=4)")}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.9), sharey=True)
    for j, ID in enumerate(IDS):
        ax = axes[j]
        for topo in TOPOS:
            rr = [r for r in rows if r["ID"] == ID and r["topo"] == topo]
            stop = min([r["N"] for r in rr if r["censored"]], default=np.inf)
            rr = [r for r in rr if r["N"] < stop]
            if not rr:
                continue
            c, mk, mfc, lab = style[topo]
            N = np.array([r["N"] for r in rr])
            m = np.array([r["mean_T"] for r in rr])
            lo = m - np.array([r["ci_lo"] for r in rr])
            hi = np.array([r["ci_hi"] for r in rr]) - m
            ax.errorbar(N, m, yerr=[lo, hi], color=c, marker=mk, ms=5, mfc=mfc,
                        mew=1.3, lw=1.5, capsize=2, label=lab, zorder=3)
            # a series that never stopped is continued with the main-grid points
            if stop == np.inf:
                ext = grid_extension(ID, topo, [n for n in (2000, 5000, 10000, 100000)
                                                 if n > N.max()])
                if ext:
                    eN, em, elo, ehi = map(np.array, zip(*ext))
                    ax.plot([N[-1], eN[0]], [m[-1], em[0]], color=c, lw=1.5, ls="--",
                            zorder=2)
                    ax.errorbar(eN, em, yerr=[em - elo, ehi - em], color=c, marker=mk,
                                ms=5, mfc=mfc, mew=1.3, lw=1.5, ls="--", capsize=2,
                                zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(5, 5e3)
        ax.set_xlabel("system size $N$")
        if j == 0:
            ax.set_ylabel("mean convergence time (rounds)")
        else:
            ax.legend(loc="lower right")
        ax.set_title(PAIR[ID])
        ax.grid(True, which="major", color="0.9", lw=0.6)
    fig.tight_layout()
    os.makedirs(paths.FIGURES, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(paths.FIGURES, f"T_vs_N_log.{ext}"))
    print(f"figure: {os.path.join(paths.FIGURES, 'T_vs_N_log.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "plotlog"])
    ap.add_argument("--max-processes", type=int, default=16)
    args = ap.parse_args()
    if args.stage == "run":
        run(args.max_processes)
    else:
        plot_log()


if __name__ == "__main__":
    main()
