"""Convergence time on random regular networks as a function of the degree K,
at N = 10000 (edge rule).

K = 4 reuses the seeds and cached graphs of the `rrn` condition in data/grid,
so its runs are the same as the main grid's (with a longer t_max). The two
Llama pairs that never converge on FC are skipped.

  python run_density.py pool [--ks ...]          # pre-generate the graph cache
  python run_density.py run  [--max-processes P]
  python run_density.py run --dry-run

  --task-id I --num-tasks J   shard the condition list (Slurm arrays)

The graph cache for the dense K values is about 1 GB (paths.NETWORKS_DENSITY).
"""
import argparse
import multiprocessing as mp
import os

import numpy as np

import paths
import run_grid as G
from run_grid import (_run_rq1, _stable_seed, _write, _empty_rq1, RULE_CODE,
                      SNAP_ROUNDS, H, shard, _group_by_agent, build_agent, _load)
from networks import generate_rrn

MODELS = list(G.MODELS)
IDS = list(G.IDS)
N = 10000
KS = [4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256]

TIME = 20000        # t_max, population rounds
T_REC = 1000        # rounds of S(t) and freq(t) stored
ITERS = 100

SKIP_CELLS = {("llama31_70B", 219), ("llama31_70B", 1161)}

RULE = "edge"       # on a regular graph the node rule is the same process
RESULTS_DIR = paths.DENSITY
CACHE_DIR = paths.NETWORKS_DENSITY

# K = 4 reads the main grid's graph pool; matching its size keeps
# `it % pool_size` on the same graph indices.
GRID_CACHE_DIR = G.CACHE_DIR
GRID_POOL_SIZE = G.POOL_SIZE[N]


def topo_name(K):
    """Label used in file names."""
    return f"rrn_k{K}"


def topo_key(K):
    """Seed and cache namespace; K = 4 is the main grid's `rrn`."""
    return "rrn" if K == 4 else f"rrn_k{K}"


def pool_size(K):
    """Number of distinct graphs per K, limited by size (a K = 256 graph is ~20 MB)."""
    if K == 4:
        return GRID_POOL_SIZE
    return max(10, min(100, int(150e6 / (8 * N * K))))


def iters_for(model, ID):
    return ITERS


def _graph_path(K, idx):
    if K == 4:
        return os.path.join(GRID_CACHE_DIR, f"rrn_N{N}_i{idx}.npz")
    return os.path.join(CACHE_DIR, f"{topo_key(K)}_N{N}_i{idx}.npz")


def generate_pool(K, verbose=True):
    """Write pool_size(K) cached graphs. Existing files are kept."""
    if K == 4:
        missing = [i for i in range(pool_size(4))
                   if not os.path.exists(_graph_path(4, i))]
        if missing:
            raise RuntimeError(
                f"K=4 uses the main grid's graphs in {GRID_CACHE_DIR}/, but "
                f"{len(missing)} of {pool_size(4)} are missing. Run "
                f"`python run_grid.py pool --topos rrn --ns {N}`.")
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    for idx in range(pool_size(K)):
        path = _graph_path(K, idx)
        if os.path.exists(path):
            continue
        edges, degrees = generate_rrn(N, k=K, seed=_stable_seed(topo_key(K), N, idx))
        assert degrees.min() == degrees.max() == K, "degree sequence is not K-regular"
        tmp = f"{path}.tmp_{os.getpid()}"
        np.savez(tmp, edges=edges, degrees=degrees)   # the edge rule needs no CSR
        os.replace(tmp + ".npz", path)
        if verbose:
            print(f"pool K={K} {idx + 1}/{pool_size(K)}", flush=True)


_POOL_MEM = {}
_DUMMY = np.zeros(1, dtype=np.int64)      # adj_start / adj_list, unused by the edge rule


def get_graph(K, run_index):
    """(edges, degrees) for run `run_index`, taken from the cached pool."""
    idx = run_index % pool_size(K)
    key = (K, idx)
    if key not in _POOL_MEM:
        path = _graph_path(K, idx)
        if not os.path.exists(path):
            generate_pool(K, verbose=False)
        with np.load(path) as z:
            g = (z["edges"], z["degrees"])
        _POOL_MEM.clear()
        _POOL_MEM[key] = g
    return _POOL_MEM[key]


def out_path(model, ID, K):
    return os.path.join(
        RESULTS_DIR, f"RQ1_{model}_ID_{ID}_{topo_name(K)}_N{N}_{RULE}_H{H}.pkl")


def run_seed(model, ID, K):
    """Equal to the main grid's seed for (model, ID, 'rrn', N, 'edge') when K = 4."""
    return _stable_seed(model, ID, topo_key(K), N, RULE)


def run_condition(model, ID, K, agent, iters, record=True):
    q_s, P_key, empty_state, _, _ = agent
    snap_counts = np.array([max(1, round(f * N)) for f in SNAP_ROUNDS],
                           dtype=np.int64)
    out = out_path(model, ID, K)

    data = _load(out) if record else None
    if data is None:
        data = _empty_rq1(len(SNAP_ROUNDS), K, T_REC)
    if data["total_attempts"] >= iters:
        return f"skip  K={K:<3d} {model} {ID}"

    for it in range(data["total_attempts"], iters):
        edges, degrees = get_graph(K, it)
        state, ctime, conv, ss, ts, st, ft = _run_rq1(
            RULE_CODE[RULE], edges, degrees, _DUMMY, _DUMMY,
            q_s, P_key, empty_state, N, TIME, snap_counts, K,
            False, run_seed(model, ID, K) + it)
        if conv:
            data["consensus"][state] += 1
            data["time"][state].append(ctime)
        else:
            data["non_consensus"][state]["count"] += 1
            data["non_consensus"][state]["freq"].append(float(ft[-1]))
        data["conv_times_all"].append(int(ctime) if conv else -1)
        data["final_states"].append(int(state))
        data["succ_snap_sum"] += ss
        data["tot_snap_sum"] += ts
        data["succ_t_sum"] += st[:T_REC]
        data["succ_t_sq"] += st[:T_REC] * st[:T_REC]
        data["freq_t_sum"] += ft[:T_REC]
        data["freq_t_sq"] += ft[:T_REC] * ft[:T_REC]
        data["deg_hist_sum"] += np.bincount(degrees, minlength=K + 1)[:K + 1]
        data["total_attempts"] += 1

    data.update({"rule": RULE, "topo": topo_name(K), "K": K, "kbar": float(K),
                 "N": N, "T": TIME, "T_REC": T_REC, "no_break": False,
                 "ITERS": iters, "H": H, "pool_size": pool_size(K),
                 "base_seed": run_seed(model, ID, K),
                 "model": model, "ID": ID})
    if record:
        _write(out, data)
    nconv = data["consensus"][0] + data["consensus"][1]
    return (f"done  K={K:<3d} {model} {ID}: {nconv}/{data['total_attempts']} "
            f"converged, {data['consensus']}")


def conditions(models, ids, ks):
    return [(m, i, k) for m in models for i in ids for k in ks
            if (m, i) not in SKIP_CELLS]


def _worker(args):
    model, ID, conds = args
    try:
        agent = build_agent(model, ID)
        msgs = []
        for (_, _, K) in conds:
            msgs.append(run_condition(model, ID, K, agent, iters_for(model, ID)))
            print(msgs[-1], flush=True)
        return "\n".join(msgs)
    except Exception as e:
        return f"ERROR {model} {ID}: {type(e).__name__}: {e}"


RATE = G.RATE


def _estimate(conds):
    """Upper bound on CPU time, assuming every run lasts t_max."""
    total = sum(iters_for(m, i) * TIME * N / RATE for (m, i, _) in conds)
    print(f"upper-bound estimate: {total / 3600:.1f} CPU-hours "
          f"(runs that converge early cost far less)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "pool"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--ids", nargs="+", type=int, default=None)
    ap.add_argument("--ks", nargs="+", type=int, default=None)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--max-processes", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = args.models or MODELS
    ids = args.ids or IDS
    ks = args.ks or KS
    for K in ks:
        if K % 2 or K >= N:
            ap.error(f"K={K}: the generator needs an even K < N")

    if args.stage == "pool":
        for K in ks:
            generate_pool(K)
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    conds = conditions(models, ids, ks)
    mine = shard(conds, args.task_id, args.num_tasks)
    print(f"density sweep: {len(mine)}/{len(conds)} conditions "
          f"(task {args.task_id}/{args.num_tasks})", flush=True)
    if args.dry_run:
        for c in mine:
            print("   ", c, "iters=", iters_for(c[0], c[1]))
        _estimate(mine)
        return

    groups = _group_by_agent(mine)
    if args.max_processes > 1:
        with mp.Pool(min(args.max_processes, len(groups))) as pool:
            for r in pool.imap_unordered(_worker, groups):
                print(r, flush=True)
    else:
        for g in groups:
            _worker(g)


if __name__ == "__main__":
    main()
