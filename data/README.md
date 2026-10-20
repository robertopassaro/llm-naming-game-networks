# Data

`q_dicts/` and `crows_pairs_stereo_sample.pkl` are the model inputs and are
part of the repository. The simulation outputs (`grid/`, `tmax/`, `density/`,
`metastable_K/`, `metastable_N/`) are archived on Zenodo
([doi:10.5281/zenodo.23497527](https://doi.org/10.5281/zenodo.23497527)); unzip the archive in the repository root.

All files are Python pickles of plain dicts, lists and numpy arrays.

Conventions used throughout:

- Time is measured in **population rounds**; one round is N pairwise interactions.
- Word 0 and word 1 of a pair are `differences[0]` and `differences[1]` in
  `crows_pairs_stereo_sample.pkl` (e.g. ID 397: word 0 = "she", word 1 = "he").
- A run **converges** at the first round t (counted from 0, t >= 3) at which the
  success rate averaged over rounds t-2, t-1, t is at least 0.98. Its final
  word is the majority word over the same three rounds. `-1` marks a run that
  did not converge within T rounds.
- `model` is one of `llama31_70B`, `gpt-4o`, `QWQ_32B`, `phi_4`; `ID` is the
  CrowS-Pairs sentence ID of the word pair.
- `topo` is `fc` (complete graph), `ucm_2.5`, `ucm_5.0`, `rrn` (K = 4) or
  `rrn_k{K}`. `rule` is `edge`, `node` or `annealed`.

## Inputs

**`q_dicts/Q_dict_{model}_ID_{ID}_-50_100_0.5tmp_BLANK_False.pkl`**: the agent
policy, extracted from the LLM by Flint et al. (2026). A dict from memory state to
the probability of producing word 0. A memory state is a string of the last up
to H = 5 interactions, oldest first, each written as two characters (own word,
partner's word); `""` is the empty memory. 1365 states per file.

**`crows_pairs_stereo_sample.pkl`**: dict `ID -> {"differences": [word0, word1],
"template", "bias_type", "stereo_antistereo"}`, taken from CrowS-Pairs (Nangia et
al., 2020). Only the 11 IDs listed in the scripts are used.

## grid/ (run_grid.py)

**`RQ1_{model}_ID_{ID}_{topo}_N{N}_{rule}_H5.pkl`**: one condition, all its runs.

| key | meaning |
|---|---|
| `total_attempts`, `ITERS` | number of runs |
| `conv_times_all` | list, convergence round of each run (-1 = not converged) |
| `final_states` | list, final majority word of each run (0/1) |
| `consensus` | `{0: n0, 1: n1}`, converged runs by final word |
| `time` | `{0: [...], 1: [...]}`, convergence rounds by final word |
| `non_consensus` | `{w: {"count", "freq"}}`, non-converged runs by majority word; `freq` is the share of word-1 outputs in the last round |
| `succ_t_sum`, `succ_t_sq` | float[T], sum and sum of squares over runs of the success rate in each round |
| `freq_t_sum`, `freq_t_sq` | float[T], same for the share of word-1 outputs |
| `snap_rounds` (`snap_fracs` is the same list) | rounds at which the degree counters were copied |
| `succ_snap_sum`, `tot_snap_sum` | int[len(snap_rounds), KMAX+1], summed over runs: successful and total interactions involving a node of degree k, cumulative up to each snapshot (an interaction counts once for each endpoint) |
| `deg_hist_sum` | int[KMAX+1], degree histogram summed over the graphs of all runs |
| `KMAX` | largest degree class (int(sqrt(N)) for UCM, 4 for RRN, 0 for FC) |
| `T` | t_max, rounds (1000) |
| `no_break` | if True the curves run to T even after convergence; otherwise they are padded with their last value |
| `pool_size` | number of distinct cached graphs cycled over the runs; 0 = a new graph per run |
| `base_seed` | seed of run `i` is `base_seed + i` |
| `rule`, `topo`, `N`, `H`, `model`, `ID` | condition |

Most files at N >= 1000 were written with the first 12 snapshot rounds (0.1 to
32). The UCM 2.5 files at N = 10000 (edge and annealed), the annealed files and
the runs at N = 100 ... 2000 have all 16 (up to 512). The runs at
N = 100 ... 2000 were made with `--no-pool`.

**`RQ2_{model}_ID_{ID}_{topo}_N10000_{placement}_dir{w}_edge_H5.pkl`**: critical
committed fraction for committed word `w`; the other agents start from full
memories of agreement on the other word. `placement` is `random`, `hub`
(highest-degree nodes) or `leaf` (lowest-degree nodes).

| key | meaning |
|---|---|
| `p_star` | critical committed fraction, `m_star / N`; 0 = flips without committed agents |
| `m_star` | number of committed agents at p* |
| `probes` | list of `(p, flip fraction, runs)` evaluated by the bisection |
| `R` | runs per probe (200) |
| `P_TOL` | bisection resolution in p (0.01) |
| `committed_word`, `placement`, `topo`, `rule`, `N`, `H`, `model`, `ID` | condition |

A run flips when it converges within 1000 rounds with the committed word in the
majority. Committed agents count in the success rate.

## tmax/ (run_tmax.py)

**`RQ1LONG_{model}_ID_{ID}_{topo}_N{N}_{rule}_T{T}_H5.pkl`**: replay, with a
longer t_max, of the runs of a `grid/` condition that did not converge in 1000
rounds.

| key | meaning |
|---|---|
| `T_long`, `T_orig` | new and original t_max (rounds) |
| `n_fail_orig` | runs that did not converge originally |
| `R_rerun` | runs replayed (at most 300 at N = 1000, 100 at 5000 and 10000, 30 at 100000) |
| `long_conv`, `long_state`, `long_freq` | per replayed run index: convergence round (-1 = still not converged), final word, last-round share of word 1 |
| `conv_times_merged` | list over all runs: original or replayed convergence round, -1 if still not converged, `None` if not replayed |
| `censored` | list over all runs, True where a replayed run did not converge |
| `n_escaped`, `n_censored` | replayed runs that did and did not converge |
| `iters_orig`, `base_seed`, `pool_size`, and the condition keys | as in `grid/` |

## density/ (run_density.py)

**`RQ1_{model}_ID_{ID}_rrn_k{K}_N10000_edge_H5.pkl`**: RRN of degree K, 100 runs,
t_max = 20000 rounds. Same keys as `grid/` RQ1 files, plus `K`, `kbar` and
`T_REC`: the per-round curves keep only the first `T_REC` = 1000 rounds. The
two Llama pairs 219 and 1161 are not part of the sweep; the four files with 20
runs for them at K = 192 and 256 are not used.

## metastable_K/ (run_density_metastable.py)

Llama-3.1-70B, IDs 219, 1161 and 92, N = 100 and 200, 200 runs per condition,
t_max = 1e5 (N = 100) and 1e6 (N = 200) rounds. One subfolder per arm:
`quenched`, `annealed`, `fc`, `quenched_swap`. Same keys as `density/`, plus
`arm` and `generators` (graph generator used in each run: `stub`, `swap` or
`complete`). For `fc`, `K` is N - 1.

## metastable_N/ (run_metastable_N.py)

**`RQ1_llama31_70B_ID_{ID}_{topo}_N{N}_edge_H5_Nsweep.pkl`**: 200 runs, t_max =
1e4 rounds, a new connected graph per run.

| key | meaning |
|---|---|
| `conv_times_all`, `final_states` | as in `grid/` |
| `graph_redraws` | per run, disconnected graphs drawn before a connected one |
| `T`, `base_seed`, `topo`, `N`, `ID`, `model`, `rule`, `H` | condition |

The folder also holds runs on RRN with K > 4 (`rrn_k{K}`), which are not used in
the paper.
