"""Naming-game simulations on networks: consensus runs (RQ1) and the
committed-minority critical mass (RQ2).

Agents are the finite-memory policies of emergence_dynamics.py. Only the way
pairs are drawn differs between interaction rules:

  edge     : a uniform random edge; both endpoints speak and update
  node     : a uniform random node, then a uniform random neighbour
  fc       : two distinct uniform random agents (well mixed)
  annealed : two uniform random stubs (infinitely fast rewiring)

A run converges when the success rate averaged over the last 3 rounds reaches
0.98; the convergence word is the majority word over the same 3 rounds.

Usage
-----
  python run_grid.py pool                     # pre-generate the network cache
  python run_grid.py rq1 [--rule edge|node|annealed]
  python run_grid.py rq2

  --task-id I --num-tasks K   shard the condition list (Slurm arrays)
  --dry-run                   list the conditions and a cost estimate
"""
import argparse
import multiprocessing as mp
import os
import pickle
import shutil
import zlib

import numpy as np
from numba import njit

import paths
from emergence_dynamics import (H_reducer, integer_mapping,
                                integer_probabilities, state_transitions)
from networks import generate_rrn, generate_ucm

MODELS = ["llama31_70B", "gpt-4o", "QWQ_32B", "phi_4"]
IDS = [92, 102, 191, 219, 344, 349, 397, 410, 505, 626, 1161]

H = 5
TIME = 1000                      # t_max, population rounds

RQ1_TOPOS = ["ucm_2.5", "ucm_5.0", "rrn", "fc"]
RQ1_NS = [1000, 5000, 10000, 100000]
ITERS = {1000: 1000, 5000: 1000, 10000: 1000, 100000: 300}
NO_BREAK_NS = [10000]            # record full-length curves at this size

# Snapshot times (population rounds) of the cumulative degree-resolved counters.
# Analysis scripts index into this list, so only append to it.
SNAP_ROUNDS = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 1, 2, 4, 8, 16, 32,
               64, 128, 256, 512]

RQ2_N = 10000
RQ2_COMBOS = [("ucm_2.5", "random"), ("ucm_5.0", "random"),
              ("rrn", "random"), ("fc", "random"),
              ("ucm_2.5", "hub"), ("ucm_2.5", "leaf"),
              ("ucm_5.0", "hub"), ("ucm_5.0", "leaf")]
R_PER_CANDIDATE = 200            # runs per bisection probe
FLIP_THRESHOLD = 0.5             # flip fraction that defines p*
P_TOL = 0.01                     # stop when (m_hi - m_lo) / N <= P_TOL

# On a complete graph the node rule draws a uniform pair, i.e. it is the fc rule.
NODE_TOPOS = ["ucm_2.5", "ucm_5.0", "rrn"]
NODE_NS = list(RQ1_NS)

POOL_SIZE = {1000: 200, 5000: 200, 10000: 200, 100000: 50}

RESULTS_DIR = paths.GRID
CACHE_DIR = paths.NETWORKS

RULE_CODE = {"edge": 0, "node": 1, "fc": 2, "annealed": 3}


# Agents and graphs
def build_agent(model, ID):
    with open(paths.q_dict_path(model, ID), "rb") as f:
        q_total = pickle.load(f)
    q_H, keys_H, steady_0, steady_1 = H_reducer(q_total, H)
    F, _ = integer_mapping(keys_H)
    q_s = integer_probabilities(q_H, F)
    P_key, _ = state_transitions(q_H, H, F, _inv(F))
    return (q_s.astype(np.float64), P_key.astype(np.int64),
            F[""], F[steady_0], F[steady_1])


def _inv(F):
    return {v: k for k, v in F.items()}


def make_network(topo, N, seed):
    if topo == "rrn":
        return generate_rrn(N, k=4, seed=seed)
    gamma = float(topo.split("_")[1])
    return generate_ucm(N, gamma, kmin=4, seed=seed)


def kmax_of(topo, N):
    return 4 if topo == "rrn" else int(np.floor(np.sqrt(N)))


def build_csr(edges, degrees):
    """Neighbour lists in CSR form: adj_list[adj_start[i]:adj_start[i+1]]."""
    N = len(degrees)
    adj_start = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(degrees, out=adj_start[1:])
    fill = adj_start[:N].copy()
    adj_list = np.empty(adj_start[N], dtype=np.int64)
    for u, v in edges:
        adj_list[fill[u]] = v
        fill[u] += 1
        adj_list[fill[v]] = u
        fill[v] += 1
    return adj_start, adj_list


def _graph_path(topo, N, idx):
    return os.path.join(CACHE_DIR, f"{topo}_N{N}_i{idx}.npz")


def make_graph(topo, N, seed):
    """One graph and its CSR adjacency (not cached)."""
    edges, degrees = make_network(topo, N, seed)
    adj_start, adj_list = build_csr(edges, degrees)
    return edges, degrees, adj_start, adj_list


def generate_pool(topo, N, verbose=True):
    """Write POOL_SIZE[N] cached graphs for (topo, N). Existing files are kept."""
    if topo == "fc":
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    for idx in range(POOL_SIZE[N]):
        path = _graph_path(topo, N, idx)
        if os.path.exists(path):
            continue
        edges, degrees, adj_start, adj_list = make_graph(topo, N, _pool_seed(topo, N, idx))
        tmp = f"{path}.tmp_{os.getpid()}"
        np.savez(tmp, edges=edges, degrees=degrees,
                 adj_start=adj_start, adj_list=adj_list)
        os.replace(tmp + ".npz", path)
        if verbose:
            print(f"pool {topo} N{N} {idx + 1}/{POOL_SIZE[N]}", flush=True)


def _load(fname):
    try:
        with open(fname, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save(fname, data):
    tmp = f"{fname}.tmp_{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    shutil.move(tmp, fname)


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _save(path, data)


def _stable_seed(*key):
    """Seed derived from the condition key; unlike hash() it is the same in
    every process and session."""
    return (zlib.crc32(repr(key).encode()) % 10_000_000) + 1


def _pool_seed(topo, N, idx):
    return _stable_seed(topo, N, idx)


_POOL_MEM = {}


def get_graph(topo, N, run_index, use_pool=True):
    """Graph for run `run_index`: taken from the cached pool, or built fresh."""
    if topo == "fc":
        z = np.zeros(1, dtype=np.int64)
        return np.zeros((1, 2), dtype=np.int64), z, z, z
    if not use_pool:
        return make_graph(topo, N, run_index + 1)
    idx = run_index % POOL_SIZE[N]
    key = (topo, N, idx)
    if key not in _POOL_MEM:
        path = _graph_path(topo, N, idx)
        if os.path.exists(path):
            with np.load(path) as z:
                g = (z["edges"], z["degrees"], z["adj_start"], z["adj_list"])
        else:
            generate_pool(topo, N, verbose=False)
            with np.load(_graph_path(topo, N, idx)) as z:
                g = (z["edges"], z["degrees"], z["adj_start"], z["adj_list"])
        _POOL_MEM.clear()                         # keep one graph in memory
        _POOL_MEM[key] = g
    return _POOL_MEM[key]


# RQ1 kernel
@njit(cache=True)
def _run_rq1(rule, edges, degrees, adj_start, adj_list,
             q_s, P_key, empty_state, N, T, snap_counts, KMAX, no_break, seed):
    """One RQ1 run.

    Returns (final_state, conv_time, converged, succ_snap, tot_snap,
             succ_t, freq_t). succ_t[t] is the success rate in round t and
    freq_t[t] the fraction of outputs equal to word 1. With no_break=False the
    run stops at convergence and both curves are padded with their last value;
    with no_break=True it always runs T rounds.
    """
    np.random.seed(seed)
    M = edges.shape[0]
    # The annealed rule draws from a stub array built from the degree sequence
    # alone, so its runs do not depend on the wiring of the graph.
    stubs = adj_list
    L = adj_list.shape[0]
    if rule == 3:
        L = 0
        for u in range(N):
            L += degrees[u]
        stubs = np.empty(L, dtype=np.int64)
        w = 0
        for u in range(N):
            for _ in range(degrees[u]):
                stubs[w] = u
                w += 1
    pop = np.full(N, empty_state, dtype=np.int64)
    succ_k = np.zeros(KMAX + 1, dtype=np.int64)
    tot_k = np.zeros(KMAX + 1, dtype=np.int64)
    nsnap = snap_counts.shape[0]
    succ_snap = np.zeros((nsnap, KMAX + 1), dtype=np.int64)
    tot_snap = np.zeros((nsnap, KMAX + 1), dtype=np.int64)
    succ_t = np.zeros(T, dtype=np.float64)
    freq_t = np.zeros(T, dtype=np.float64)

    g = 0
    snap_idx = 0
    s0 = s1 = s2 = 0.0
    f0 = f1 = f2 = 0.0
    converged = False
    final_state = 0
    conv_time = T - 1
    last_t = T - 1

    for t in range(T):
        success = 0
        all_words = 0
        for _ in range(N):
            if rule == 0:                                   # edge
                e = np.random.randint(0, M)
                i = edges[e, 0]
                j = edges[e, 1]
            elif rule == 1:                                 # node
                i = np.random.randint(0, N)
                di_ = degrees[i]
                if di_ == 0:
                    continue
                j = adj_list[adj_start[i] + np.random.randint(0, di_)]
            elif rule == 3:                                 # annealed
                i = stubs[np.random.randint(0, L)]
                j = stubs[np.random.randint(0, L)]
                while j == i:
                    j = stubs[np.random.randint(0, L)]
            else:                                           # fully connected
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
            suc = 1 if first == second else 0
            success += suc
            all_words += first + second

            if rule != 2:
                di = degrees[i]
                dj = degrees[j]
                tot_k[di] += 1
                tot_k[dj] += 1
                if suc:
                    succ_k[di] += 1
                    succ_k[dj] += 1
            g += 1
            if snap_idx < nsnap and g == snap_counts[snap_idx]:
                for kk in range(KMAX + 1):
                    succ_snap[snap_idx, kk] = succ_k[kk]
                    tot_snap[snap_idx, kk] = tot_k[kk]
                snap_idx += 1

        succ_t[t] = success / N
        freq_t[t] = all_words / (2 * N)
        s0, s1, s2 = s1, s2, success / N
        f0, f1, f2 = f1, f2, all_words / (2 * N)
        if (not converged) and t >= 3 and (s0 + s1 + s2) / 3.0 >= 0.98:
            final_state = 1 if (f0 + f1 + f2) / 3.0 >= 0.5 else 0
            conv_time = t
            converged = True
            if not no_break:
                last_t = t
                break

    if not converged:
        final_state = 1 if (f0 + f1 + f2) / 3.0 >= 0.5 else 0
    for t in range(last_t + 1, T):
        succ_t[t] = succ_t[last_t]
        freq_t[t] = freq_t[last_t]
    return final_state, conv_time, converged, succ_snap, tot_snap, succ_t, freq_t


# RQ1 condition runner
def _empty_rq1(nsnap, KMAX, T):
    return {
        "consensus": {0: 0, 1: 0},
        "time": {0: [], 1: []},
        "non_consensus": {0: {"count": 0, "freq": []}, 1: {"count": 0, "freq": []}},
        "total_attempts": 0,
        "conv_times_all": [],
        "final_states": [],
        "succ_t_sum": np.zeros(T), "succ_t_sq": np.zeros(T),
        "freq_t_sum": np.zeros(T), "freq_t_sq": np.zeros(T),
        "snap_rounds": list(SNAP_ROUNDS),
        "snap_fracs": list(SNAP_ROUNDS),
        "KMAX": KMAX,
        "succ_snap_sum": np.zeros((nsnap, KMAX + 1), dtype=np.int64),
        "tot_snap_sum": np.zeros((nsnap, KMAX + 1), dtype=np.int64),
        "deg_hist_sum": np.zeros(KMAX + 1, dtype=np.int64),
    }


def rq1_path(model, ID, topo, N, rule):
    return os.path.join(RESULTS_DIR,
                        f"RQ1_{model}_ID_{ID}_{topo}_N{N}_{rule}_H{H}.pkl")


def run_rq1_condition(model, ID, topo, N, rule, agent, iters, use_pool=True):
    q_s, P_key, empty_state, _, _ = agent
    KMAX = 0 if topo == "fc" else kmax_of(topo, N)
    no_break = N in NO_BREAK_NS
    snap_counts = np.array([max(1, round(f * N)) for f in SNAP_ROUNDS],
                           dtype=np.int64)
    out = rq1_path(model, ID, topo, N, rule)

    data = _load(out) or _empty_rq1(len(SNAP_ROUNDS), KMAX, TIME)
    if data["total_attempts"] >= iters:
        return f"skip  RQ1 {model} {ID} {topo} N{N} {rule}"

    for it in range(data["total_attempts"], iters):
        edges, degrees, adj_start, adj_list = get_graph(topo, N, it, use_pool)
        seed = _run_seed(model, ID, topo, N, rule) + it
        state, ctime, conv, ss, ts, st, ft = _run_rq1(
            RULE_CODE[rule if topo != "fc" else "fc"],
            edges, degrees, adj_start, adj_list,
            q_s, P_key, empty_state, N, TIME, snap_counts, KMAX, no_break, seed)
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
        data["succ_t_sum"] += st
        data["succ_t_sq"] += st * st
        data["freq_t_sum"] += ft
        data["freq_t_sq"] += ft * ft
        if topo != "fc":
            data["deg_hist_sum"] += np.bincount(degrees, minlength=KMAX + 1)[:KMAX + 1]
        data["total_attempts"] += 1

    data.update({"rule": rule, "topo": topo, "N": N, "T": TIME,
                 "no_break": no_break, "ITERS": iters, "H": H,
                 "pool_size": (POOL_SIZE[N] if use_pool and topo != "fc" else 0),
                 "base_seed": _run_seed(model, ID, topo, N, rule),
                 "model": model, "ID": ID})
    _write(out, data)
    return f"done  RQ1 {model} {ID} {topo} N{N} {rule}: {data['consensus']}"


def _run_seed(*key):
    return _stable_seed(*key)


# RQ2: committed minorities
@njit(cache=True)
def _simulate_rq2(edges, q_s, P_key, cmask, committed_word, init_state,
                  N, TIME, seed):
    """One run on a graph (edge rule). Committed agents always say
    `committed_word` and never update. Returns True if the run converges with
    the committed word in the majority."""
    np.random.seed(seed)
    M = edges.shape[0]
    pop = np.full(N, init_state, dtype=np.int64)
    s0 = s1 = s2 = 0.0
    f0 = f1 = f2 = 0.0
    converged = False
    final_state = 0

    for t in range(TIME):
        success = 0
        all_words = 0
        for _ in range(N):
            e = np.random.randint(0, M)
            i = edges[e, 0]
            j = edges[e, 1]
            oi = committed_word if cmask[i] else 1 - int(np.random.random() < q_s[pop[i]])
            oj = committed_word if cmask[j] else 1 - int(np.random.random() < q_s[pop[j]])
            si = pop[i]
            sj = pop[j]
            if not cmask[i]:
                pop[i] = P_key[si, sj, 2 * oi + oj]
            if not cmask[j]:
                pop[j] = P_key[sj, si, 2 * oj + oi]
            success += 1 if oi == oj else 0
            all_words += oi + oj

        s0, s1, s2 = s1, s2, success / N
        f0, f1, f2 = f1, f2, all_words / (2 * N)
        if t >= 3 and (s0 + s1 + s2) / 3.0 >= 0.98:
            final_state = 1 if (f0 + f1 + f2) / 3.0 >= 0.5 else 0
            converged = True
            break

    if not converged:
        final_state = 1 if (f0 + f1 + f2) / 3.0 >= 0.5 else 0
    return converged and final_state == committed_word


@njit(cache=True)
def _simulate_fc_rq2(q_s, P_key, cmask, committed_word, init_state, N, TIME, seed):
    """Well-mixed version of _simulate_rq2."""
    np.random.seed(seed)
    pop = np.full(N, init_state, dtype=np.int64)
    s0 = s1 = s2 = 0.0
    f0 = f1 = f2 = 0.0
    converged = False
    final_state = 0

    for t in range(TIME):
        success = 0
        all_words = 0
        for _ in range(N):
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            while j == i:
                j = np.random.randint(0, N)
            oi = committed_word if cmask[i] else 1 - int(np.random.random() < q_s[pop[i]])
            oj = committed_word if cmask[j] else 1 - int(np.random.random() < q_s[pop[j]])
            si = pop[i]
            sj = pop[j]
            if not cmask[i]:
                pop[i] = P_key[si, sj, 2 * oi + oj]
            if not cmask[j]:
                pop[j] = P_key[sj, si, 2 * oj + oi]
            success += 1 if oi == oj else 0
            all_words += oi + oj

        s0, s1, s2 = s1, s2, success / N
        f0, f1, f2 = f1, f2, all_words / (2 * N)
        if t >= 3 and (s0 + s1 + s2) / 3.0 >= 0.98:
            final_state = 1 if (f0 + f1 + f2) / 3.0 >= 0.5 else 0
            converged = True
            break

    if not converged:
        final_state = 1 if (f0 + f1 + f2) / 3.0 >= 0.5 else 0
    return converged and final_state == committed_word


def _place(strategy, degrees, m, rng):
    """Boolean mask of the m committed nodes: random, the m highest-degree
    nodes (hub) or the m lowest-degree nodes (leaf); ties broken at random."""
    N = len(degrees)
    if m <= 0:
        return np.zeros(N, dtype=np.bool_)
    if strategy == "random":
        idx = rng.choice(N, size=m, replace=False)
    else:
        order = np.lexsort((rng.random(N), degrees))
        idx = order[-m:] if strategy == "hub" else order[:m]
    mask = np.zeros(N, dtype=np.bool_)
    mask[idx] = True
    return mask


def _bisect_pstar(eval_ff, N):
    """Smallest committed count m with flip fraction >= FLIP_THRESHOLD.

    eval_ff(m) returns the flip fraction. The count is doubled until the
    threshold is crossed, then bisected down to P_TOL * N agents. Returns
    (m_star or None, probes); m_star is the upper end of the final bracket.
    """
    probes = []

    def ff(m):
        v = eval_ff(m)
        probes.append((m / N, v, R_PER_CANDIDATE))
        return v

    if ff(0) >= FLIP_THRESHOLD:           # flips with no committed agents
        return 0, probes
    m_hi = 1
    while ff(m_hi) < FLIP_THRESHOLD:
        if m_hi >= N:
            return None, probes           # never flips
        m_hi = min(2 * m_hi, N)
    m_lo = m_hi // 2
    while (m_hi - m_lo) / N > P_TOL:
        m_mid = (m_lo + m_hi) // 2
        if ff(m_mid) >= FLIP_THRESHOLD:
            m_hi = m_mid
        else:
            m_lo = m_mid
    return m_hi, probes


def rq2_path(model, ID, topo, N, placement, w, rule):
    return os.path.join(
        RESULTS_DIR,
        f"RQ2_{model}_ID_{ID}_{topo}_N{N}_{placement}_dir{w}_{rule}_H{H}.pkl")


def _rq2_flip_fraction(q_s, P_key, topo, N, m, committed_word, init_state,
                       placement, base_seed, use_pool):
    flips = 0
    for r in range(R_PER_CANDIDATE):
        edges, degrees, _as, _al = get_graph(topo, N, r, use_pool)
        seed = base_seed + r
        rng = np.random.default_rng(seed)
        if topo == "fc":
            cmask = _place("random", np.zeros(N, dtype=np.int64), m, rng)
            flipped = _simulate_fc_rq2(q_s, P_key, cmask, committed_word,
                                       init_state, N, TIME, seed)
        else:
            cmask = _place(placement, degrees, m, rng)
            flipped = _simulate_rq2(edges, q_s, P_key, cmask, committed_word,
                                    init_state, N, TIME, seed)
        flips += 1 if flipped else 0
    return flips / R_PER_CANDIDATE


def run_rq2_condition(model, ID, topo, N, placement, w, rule, agent, use_pool=True):
    q_s, P_key, _, steady_0, steady_1 = agent
    init_state = steady_0 if w == 1 else steady_1     # consensus on the other word
    out = rq2_path(model, ID, topo, N, placement, w, rule)
    if _load(out) is not None:
        return f"skip  RQ2 {model} {ID} {topo} N{N} {placement} dir{w}"

    base = _run_seed(model, ID, topo, N, placement, w, rule)
    counter = {"i": 0}

    def eval_ff(m):
        bs = base + counter["i"] * R_PER_CANDIDATE
        counter["i"] += 1
        return _rq2_flip_fraction(q_s, P_key, topo, N, m, w, init_state,
                                  placement, bs, use_pool)

    m_star, probes = _bisect_pstar(eval_ff, N)
    data = {"p_star": (None if m_star is None else m_star / N),
            "m_star": m_star, "N": N, "R": R_PER_CANDIDATE, "P_TOL": P_TOL,
            "committed_word": w, "placement": placement, "topo": topo,
            "rule": rule, "model": model, "ID": ID, "H": H, "probes": probes}
    _write(out, data)
    return f"done  RQ2 {model} {ID} {topo} N{N} {placement} dir{w}: p*={data['p_star']}"


# Condition lists (fixed order, so Slurm shards are stable)
def rq1_conditions(models, ids, topos, ns, rule):
    return [(m, i, t, n) for m in models for i in ids for t in topos for n in ns]


def rq2_conditions(models, ids, combos, N):
    return [(m, i, t, pl, w) for m in models for i in ids
            for (t, pl) in combos for w in (0, 1)]


def shard(items, task_id, num_tasks):
    return items if num_tasks <= 1 else items[task_id::num_tasks]


def _worker(args):
    kind, model, ID, conds, use_pool, iters_override, rule = args
    try:
        agent = build_agent(model, ID)
        msgs = []
        for c in conds:
            if kind == "rq1":
                _, _, topo, N = c
                iters = iters_override or ITERS[N]
                msgs.append(run_rq1_condition(model, ID, topo, N, rule, agent,
                                              iters, use_pool))
            else:
                _, _, topo, pl, w = c
                msgs.append(run_rq2_condition(model, ID, topo, RQ2_N, pl, w,
                                              rule, agent, use_pool))
            print(msgs[-1], flush=True)
        return "\n".join(msgs)
    except Exception as e:
        return f"ERROR {kind} {model} {ID}: {type(e).__name__}: {e}"


def _group_by_agent(conds):
    """[(model, ID, [conditions...])], keeping the order."""
    groups = {}
    for c in conds:
        groups.setdefault((c[0], c[1]), []).append(c)
    return [(m, i, cs) for (m, i), cs in groups.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rq1", "rq2", "pool"])
    ap.add_argument("--rule", default="edge",
                    choices=["edge", "node", "annealed"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--ids", nargs="+", type=int, default=None)
    ap.add_argument("--topos", nargs="+", default=None)
    ap.add_argument("--ns", nargs="+", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--max-processes", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.rule != "edge":
        if args.stage == "rq2":
            ap.error("rq2 is implemented for the edge rule only")
        if args.topos and "fc" in args.topos:
            ap.error(f"{args.rule} on fc is the fc rule; use --rule edge")

    models = args.models or MODELS
    ids = args.ids or IDS
    use_pool = not args.no_pool
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.stage == "pool":
        topos = args.topos or [t for t in RQ1_TOPOS if t != "fc"]
        ns = args.ns or sorted(set(RQ1_NS + [RQ2_N]))
        for t in topos:
            for n in ns:
                generate_pool(t, n)
        return

    if args.stage == "rq1":
        default_topos = RQ1_TOPOS if args.rule == "edge" else NODE_TOPOS
        default_ns = RQ1_NS if args.rule == "edge" else NODE_NS
        conds = rq1_conditions(models, ids, args.topos or default_topos,
                               args.ns or default_ns, args.rule)
    else:
        combos = RQ2_COMBOS
        if args.topos:
            combos = [c for c in combos if c[0] in args.topos]
        conds = rq2_conditions(models, ids, combos, RQ2_N)

    mine = shard(conds, args.task_id, args.num_tasks)
    print(f"{args.stage} rule={args.rule}: {len(mine)}/{len(conds)} conditions "
          f"(task {args.task_id}/{args.num_tasks})", flush=True)
    if args.dry_run:
        for c in mine:
            print("  ", c)
        _estimate(args.stage, mine, args.iters)
        return

    groups = [(args.stage, m, i, cs, use_pool, args.iters, args.rule)
              for m, i, cs in _group_by_agent(mine)]
    if args.max_processes > 1:
        with mp.Pool(min(args.max_processes, len(groups))) as pool:
            for r in pool.imap_unordered(_worker, groups):
                print(r, flush=True)
    else:
        for gargs in groups:
            _worker(gargs)


RATE = 22e6      # interactions per second per core, measured


def _estimate(stage, conds, iters_override):
    """Upper bound on CPU time, assuming every run lasts T_max."""
    total = 0.0
    for c in conds:
        if stage == "rq1":
            N = c[3]
            runs = iters_override or ITERS[N]
        else:
            N = RQ2_N
            runs = R_PER_CANDIDATE * 12
        total += runs * TIME * N / RATE
    print(f"upper-bound estimate: {total / 3600:.1f} CPU-hours "
          f"(runs that converge early cost far less)")


if __name__ == "__main__":
    main()
