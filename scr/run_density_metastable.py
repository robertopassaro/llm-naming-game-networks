"""RRN degree sweep at small N (100 and 200) for the two Llama pairs that do not
converge in large well-mixed populations, {young, old} (219) and
{more, less} (1161), with {African, White} (92) as a control.

Arms per (pair, N):
  quenched      : edge rule on a fresh K-regular graph per run
  annealed      : annealed rule on the same degree sequence
  fc            : well-mixed reference at the same N
  quenched_swap : quenched at K = stub_max(N) with the swap generator, to check
                  that the two graph generators agree

On a regular graph the annealed rule draws a uniform pair, so it is the fc
process for every K.

Graphs:
  K = 4           networks.generate_rrn(N, 4, seed=it+1)
  4 < K <= Ks     stub matching (networks.generate_network_numba)
  Ks < K < N-1    double-edge swaps from a circulant start; stub matching
                  rarely succeeds at this density
  K = N-1         the complete graph
with Ks = 24 (N = 100) and 48 (N = 200).

  python run_density_metastable.py run [--ns ..] [--ks ..] [--arms ..] [--max-processes P]
  python run_density_metastable.py run --dry-run
"""
import argparse
import multiprocessing as mp
import os

import numpy as np
from numba import njit

import paths
from run_grid import (_run_rq1, _stable_seed, _write, _empty_rq1, RULE_CODE,
                      SNAP_ROUNDS, H, shard, build_agent, _load)
from networks import generate_rrn, generate_network_numba, build_edge_array, _seed_all

MODEL = "llama31_70B"
IDS = [219, 1161, 92]
NS = [100, 200]
KS = {100: [4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96, 99],
      200: [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 199]}
TIME = {100: 100_000, 200: 1_000_000}     # t_max, population rounds
ITERS = 200
T_REC = 1000                              # rounds of S(t) and freq(t) stored
ARMS = ["quenched", "annealed", "fc", "quenched_swap"]
RULE_OF = {"quenched": "edge", "quenched_swap": "edge", "annealed": "annealed",
           "fc": "edge"}

RUNS_DIR = paths.METASTABLE_K
STUB_RESTARTS = 2000


STUB_MAX = {100: 24, 200: 48}              # largest K in the list with K <= N/4


def stub_max(N):
    return STUB_MAX[N]


@njit(cache=True)
def _swap_regular(N, K, seed):
    """Uniform simple K-regular graph by double-edge swaps (K even, K < N-1)."""
    np.random.seed(seed)
    M = N * K // 2
    edges = np.empty((M, 2), dtype=np.int64)
    adj = np.zeros((N, N), dtype=np.bool_)
    e = 0
    for i in range(N):                         # circulant start
        for d in range(1, K // 2 + 1):
            j = (i + d) % N
            edges[e, 0] = i
            edges[e, 1] = j
            adj[i, j] = True
            adj[j, i] = True
            e += 1
    Mc = N * (N - 1) // 2 - M                  # number of non-edges
    target = 10 * min(M, Mc) + 10 * N          # accepted swaps
    max_attempts = 2000 * M + 1000000
    accepted = 0
    attempts = 0
    while accepted < target and attempts < max_attempts:
        attempts += 1
        e1 = np.random.randint(0, M)
        e2 = np.random.randint(0, M)
        if e1 == e2:
            continue
        a = edges[e1, 0]
        b = edges[e1, 1]
        if np.random.random() < 0.5:
            c = edges[e2, 0]
            d = edges[e2, 1]
        else:
            c = edges[e2, 1]
            d = edges[e2, 0]
        # (a,b),(c,d) -> (a,d),(c,b)
        if a == d or c == b or adj[a, d] or adj[c, b]:
            continue
        adj[a, b] = False
        adj[b, a] = False
        adj[c, d] = False
        adj[d, c] = False
        adj[a, d] = True
        adj[d, a] = True
        adj[c, b] = True
        adj[b, c] = True
        edges[e1, 0] = a
        edges[e1, 1] = d
        edges[e2, 0] = c
        edges[e2, 1] = b
        accepted += 1
    return edges, accepted >= target


def _stub_regular(N, K, seed):
    _seed_all(seed)
    deg = np.full(N, K, dtype=np.int64)
    for _ in range(STUB_RESTARTS):
        ok, neighbors = generate_network_numba(deg)
        if ok:
            return build_edge_array(neighbors)
    raise RuntimeError(f"stub matching failed N={N} K={K} seed={seed}")


def _complete(N):
    i, j = np.triu_indices(N, 1)
    return np.stack([i, j], axis=1).astype(np.int64)


def _degrees(edges, N):
    deg = np.zeros(N, dtype=np.int64)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    return deg


def _validate(edges, N, K):
    assert np.all(edges[:, 0] != edges[:, 1]), "self-loop"
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    assert len(np.unique(lo * N + hi)) == len(edges), "multi-edge"
    deg = _degrees(edges, N)
    assert deg.min() == deg.max() == K, "not K-regular"
    adj = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = np.zeros(N, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    assert seen.all(), "disconnected"
    return deg


def make_graph(arm, N, K, it):
    """(edges, degrees, generator name) for run `it` of a quenched arm."""
    if K == N - 1:
        edges, gen = _complete(N), "complete"
    elif K == 4 and arm == "quenched":
        edges, _ = generate_rrn(N, 4, seed=it + 1)
        gen = "stub"
    elif arm == "quenched" and K <= stub_max(N):
        edges, gen = _stub_regular(N, K, _stable_seed("rrn_k", K, N, "graph", it)), "stub"
    else:
        edges, ok = _swap_regular(N, K, _stable_seed("rrn_k", K, N, "swap", it))
        assert ok, f"swap chain did not reach its target N={N} K={K}"
        gen = "swap"
    degrees = _validate(edges, N, K)
    return edges, degrees, gen


def topo_label(arm, K):
    return "fc" if arm == "fc" else f"rrn_k{K}"


def out_path(arm, ID, N, K):
    return os.path.join(
        RUNS_DIR, arm,
        f"RQ1_{MODEL}_ID_{ID}_{topo_label(arm, K)}_N{N}_{RULE_OF[arm]}_H{H}.pkl")


def run_seed(arm, ID, N, K):
    """K = 4 and fc use the same seeds as the small-N runs in data/grid."""
    rule = RULE_OF[arm]
    if arm == "fc":
        return _stable_seed(MODEL, ID, "fc", N, "edge")
    if arm == "quenched_swap":
        return _stable_seed(MODEL, ID, f"rrn_k{K}_swap", N, rule)
    topo = "rrn" if K == 4 else f"rrn_k{K}"
    return _stable_seed(MODEL, ID, topo, N, rule)


def conditions(ns, ks_override, arms, ids):
    conds = []
    for N in ns:
        ks = ks_override or KS[N]
        for ID in ids:
            for arm in arms:
                if arm == "fc":
                    conds.append((arm, ID, N, 0))
                elif arm == "quenched_swap":
                    conds.append((arm, ID, N, stub_max(N)))
                else:
                    conds.extend((arm, ID, N, K) for K in ks)
    return conds


_AGENT = {}


def _agent(ID):
    if ID not in _AGENT:
        _AGENT[ID] = build_agent(MODEL, ID)
    return _AGENT[ID]


_DUMMY = np.zeros(1, dtype=np.int64)


def simulate(arm, ID, N, K, it, T, snap_counts):
    """One run: the _run_rq1 tuple, the degree array and the generator name."""
    q_s, P_key, empty_state, _, _ = _agent(ID)
    if arm == "fc":
        edges, degrees, gen, KMAX = np.zeros((1, 2), dtype=np.int64), _DUMMY, "fc", 0
    elif arm == "annealed":
        # the annealed rule only reads the degree sequence
        edges, degrees, gen, KMAX = np.zeros((1, 2), dtype=np.int64), \
            np.full(N, K, dtype=np.int64), "annealed", K
    else:
        edges, degrees, gen = make_graph(arm, N, K, it)
        KMAX = K
    rule = "fc" if arm == "fc" else RULE_OF[arm]
    res = _run_rq1(RULE_CODE[rule], edges, degrees, _DUMMY, _DUMMY,
                   q_s, P_key, empty_state, N, T, snap_counts, KMAX, False,
                   run_seed(arm, ID, N, K) + it)
    return res, degrees, gen, KMAX


def run_condition(cond):
    arm, ID, N, K = cond
    try:
        T = TIME[N]
        out = out_path(arm, ID, N, K)
        snap_counts = np.array([max(1, round(f * N)) for f in SNAP_ROUNDS],
                               dtype=np.int64)
        KMAX = 0 if arm == "fc" else K
        data = _load(out) or _empty_rq1(len(SNAP_ROUNDS), KMAX, T_REC)
        data.setdefault("generators", [])
        if data["total_attempts"] >= ITERS:
            return f"skip  {arm:13s} ID{ID} N{N} K{K}"
        for it in range(data["total_attempts"], ITERS):
            (state, ctime, conv, ss, ts, st, ft), degrees, gen, _ = \
                simulate(arm, ID, N, K, it, T, snap_counts)
            if conv:
                data["consensus"][state] += 1
                data["time"][state].append(ctime)
            else:
                data["non_consensus"][state]["count"] += 1
                data["non_consensus"][state]["freq"].append(float(ft[-1]))
            data["conv_times_all"].append(int(ctime) if conv else -1)
            data["final_states"].append(int(state))
            data["generators"].append(gen)
            data["succ_snap_sum"] += ss
            data["tot_snap_sum"] += ts
            data["succ_t_sum"] += st[:T_REC]
            data["succ_t_sq"] += st[:T_REC] * st[:T_REC]
            data["freq_t_sum"] += ft[:T_REC]
            data["freq_t_sq"] += ft[:T_REC] * ft[:T_REC]
            if arm != "fc":
                data["deg_hist_sum"] += np.bincount(degrees, minlength=K + 1)[:K + 1]
            data["total_attempts"] += 1
        data.update({"arm": arm, "rule": RULE_OF[arm], "topo": topo_label(arm, K),
                     "K": (N - 1 if arm == "fc" else K), "N": N, "T": T,
                     "T_REC": T_REC, "no_break": False, "ITERS": ITERS, "H": H,
                     "pool_size": 0, "base_seed": run_seed(arm, ID, N, K),
                     "model": MODEL, "ID": ID})
        _write(out, data)
        n = data["consensus"][0] + data["consensus"][1]
        ts_ = [t for t in data["conv_times_all"] if t >= 0]
        med = np.median(ts_) if ts_ else float("nan")
        return (f"done  {arm:13s} ID{ID} N{N} K{K}: {n}/{data['total_attempts']} "
                f"converged, median T={med}")
    except Exception as e:
        return f"ERROR {arm} ID{ID} N{N} K{K}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--ns", nargs="+", type=int, default=None)
    ap.add_argument("--ks", nargs="+", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=None, choices=ARMS)
    ap.add_argument("--ids", nargs="+", type=int, default=None)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--max-processes", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ns = args.ns or NS
    ids = args.ids or IDS

    conds = conditions(ns, args.ks, args.arms or ARMS, ids)
    # most expensive first (large N, quenched, dense K)
    conds.sort(key=lambda c: (-c[2], c[0] not in ("fc", "annealed"), -c[3]))
    mine = shard(conds, args.task_id, args.num_tasks)
    print(f"metastable K sweep: {len(mine)}/{len(conds)} conditions", flush=True)
    if args.dry_run:
        for c in mine:
            print("   ", c, "->", out_path(*c))
        return
    if args.max_processes > 1:
        with mp.Pool(args.max_processes) as pool:
            for r in pool.imap_unordered(run_condition, mine):
                print(r, flush=True)
    else:
        for c in mine:
            print(run_condition(c), flush=True)


if __name__ == "__main__":
    main()
