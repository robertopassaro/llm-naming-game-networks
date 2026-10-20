"""Replay the runs of data/grid that did not converge within 1000 rounds, with
a longer t_max (1e5 rounds; 1e4 at N = 1e5).

Seeds and graphs are pure functions of the run index, so replaying run `it`
reproduces the same trajectory and simply continues it past round 1000. Runs
that already converged keep their original time. At most R_CAP failed runs are
replayed per condition.

  python run_tmax.py [--max-processes P]
  python run_tmax.py --dry-run
"""
import argparse
import glob
import multiprocessing as mp
import os
import re

import numpy as np

import paths
import run_grid as G
from run_grid import (_run_rq1, get_graph, kmax_of, RULE_CODE, SNAP_ROUNDS, H,
                      build_agent, _load, _save)

T_LONG = {1000: 100_000, 5000: 100_000, 10000: 100_000, 100000: 10_000}
R_CAP = {1000: 300, 5000: 100, 10000: 100, 100000: 30}

GRID_NS = [1000, 5000, 10000, 100000]
RESULTS_DIR = paths.TMAX
SRC_DIR = paths.GRID

_FNAME = re.compile(r"RQ1_(.+)_ID_(\d+)_(ucm_[\d.]+|rrn|fc)_N(\d+)_(\w+)_H5\.pkl$")


def out_path(model, ID, topo, N, rule):
    return os.path.join(
        RESULTS_DIR,
        f"RQ1LONG_{model}_ID_{ID}_{topo}_N{N}_{rule}_T{T_LONG[N]}_H{H}.pkl")


def affected_conditions():
    """Grid conditions at the main population sizes with at least one failed run."""
    conds = []
    for f in sorted(glob.glob(os.path.join(SRC_DIR, "RQ1_*.pkl"))):
        m = _FNAME.search(os.path.basename(f))
        if not m:
            continue
        model, ID, topo, N, rule = m.groups()
        N = int(N)
        if N not in GRID_NS:
            continue
        d = _load(f)
        if d is None:
            continue
        if sum(v["count"] for v in d["non_consensus"].values()) == 0:
            continue
        conds.append((model, int(ID), topo, N, rule))
    return conds


def _replay(agent, topo, N, rule, base_seed, use_pool, indices, T):
    """Rerun the given run indices with t_max = T. Returns [(it, conv_time, freq, state)]."""
    q_s, P_key, empty_state, _, _ = agent
    KMAX = 0 if topo == "fc" else kmax_of(topo, N)
    snap_counts = np.array([max(1, round(f * N)) for f in SNAP_ROUNDS],
                           dtype=np.int64)
    code = RULE_CODE[rule if topo != "fc" else "fc"]
    out = []
    for it in indices:
        edges, degrees, adj_start, adj_list = get_graph(topo, N, it, use_pool)
        state, ctime, conv, _ss, _ts, _st, ft = _run_rq1(
            code, edges, degrees, adj_start, adj_list, q_s, P_key, empty_state,
            N, T, snap_counts, KMAX, False, base_seed + it)
        out.append((it, int(ctime) if conv else -1, float(ft[-1]), int(state)))
    return out


def run_condition(model, ID, topo, N, rule, agent):
    src = _load(os.path.join(SRC_DIR,
                             f"RQ1_{model}_ID_{ID}_{topo}_N{N}_{rule}_H{H}.pkl"))
    conv_all = list(src["conv_times_all"])
    failed = [i for i, c in enumerate(conv_all) if c == -1]
    T = T_LONG[N]
    R = min(len(failed), R_CAP[N])
    chosen = failed[:R]

    out = out_path(model, ID, topo, N, rule)
    prev = _load(out)
    if prev is not None and len(prev["long_conv"]) >= R:
        return f"skip  {model} {ID} {topo} N{N} {rule}"

    use_pool = src.get("pool_size", 0) > 0
    res = _replay(agent, topo, N, rule, src["base_seed"], use_pool, chosen, T)

    # conv_times_merged: original time, replayed time, -1 if still not
    # converged at T (censored), or None if the failed run was not replayed.
    long_conv = {it: ct for it, ct, _f, _s in res}
    merged = list(conv_all)
    censored = [False] * len(conv_all)
    for it in range(len(conv_all)):
        if conv_all[it] != -1:
            continue
        if it in long_conv and long_conv[it] != -1:
            merged[it] = long_conv[it]
        elif it in long_conv:
            merged[it] = -1
            censored[it] = True
        else:
            merged[it] = None

    n_escaped = sum(1 for v in long_conv.values() if v != -1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _save(out, {
        "model": model, "ID": ID, "topo": topo, "N": N, "rule": rule, "H": H,
        "T_long": T, "T_orig": G.TIME, "R_rerun": R,
        "n_fail_orig": len(failed), "iters_orig": src["total_attempts"],
        "base_seed": src["base_seed"], "pool_size": src.get("pool_size", 0),
        "long_conv": long_conv,
        "long_freq": {it: f for it, _c, f, _s in res},
        "long_state": {it: s for it, _c, _f, s in res},
        "conv_times_merged": merged, "censored": censored,
        "n_escaped": n_escaped, "n_censored": R - n_escaped,
    })
    return (f"done  {model} {ID} {topo} N{N} {rule}: "
            f"{n_escaped}/{R} converged within T={T}")


def _worker(args):
    model, ID, conds = args
    try:
        agent = build_agent(model, ID)
        msgs = []
        for (_m, _i, topo, N, rule) in conds:
            msgs.append(run_condition(model, ID, topo, N, rule, agent))
            print(msgs[-1], flush=True)
        return "\n".join(msgs)
    except Exception as e:
        return f"ERROR {model} {ID}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--ids", nargs="+", type=int, default=None)
    ap.add_argument("--topos", nargs="+", default=None)
    ap.add_argument("--ns", nargs="+", type=int, default=None)
    ap.add_argument("--rules", nargs="+", default=None)
    ap.add_argument("--max-processes", type=int, default=1)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conds = affected_conditions()
    for key, sel in ((0, args.models), (1, args.ids), (2, args.topos),
                     (3, args.ns), (4, args.rules)):
        if sel:
            conds = [c for c in conds if c[key] in sel]
    conds.sort(key=lambda c: -R_CAP[c[3]] * T_LONG[c[3]] * c[3])   # longest first
    mine = conds[args.task_id::args.num_tasks] if args.num_tasks > 1 else conds
    print(f"{len(mine)}/{len(conds)} conditions", flush=True)

    if args.dry_run:
        tot = 0.0
        for (model, ID, topo, N, rule) in mine:
            src = _load(os.path.join(
                SRC_DIR, f"RQ1_{model}_ID_{ID}_{topo}_N{N}_{rule}_H{H}.pkl"))
            R = min(sum(1 for c in src["conv_times_all"] if c == -1), R_CAP[N])
            tot += R * T_LONG[N] * N / G.RATE
            print(f"   {model:12s} {ID:<5d} {topo:8s} N{N:<7d} {rule:9s} "
                  f"R={R:<4d} T={T_LONG[N]}")
        print(f"upper-bound estimate: {tot / 3600:.1f} CPU-hours")
        return

    groups = {}
    for c in mine:
        groups.setdefault((c[0], c[1]), []).append(c)
    tasks = [(m, i, cs) for (m, i), cs in groups.items()]
    if args.max_processes > 1:
        with mp.Pool(min(args.max_processes, len(tasks))) as pool:
            for r in pool.imap_unordered(_worker, tasks):
                print(r, flush=True)
    else:
        for t in tasks:
            _worker(t)


if __name__ == "__main__":
    main()
