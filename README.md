# Network topology shapes consensus and vulnerability in LLM populations

Code for the paper by R. Passaro, A. Flint, L. M. Aiello, R. Pastor-Satorras and
A. Baronchelli (2026).

We simulate the naming game with agents whose policies were extracted from four
LLMs (Llama-3.1-70B, GPT-4o, QwQ-32B and Phi-4) by Flint et al. (2026), on a
complete graph, on uncorrelated scale-free networks (UCM, gamma = 2.5 and 5.0)
and on random regular networks, with populations of 10^3 to 10^5 agents. We
measure how fast a population reaches consensus, which convention wins, and
how large a committed minority must be to overturn an established convention.

## Setup

Python 3.12.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The agent policies (`data/q_dicts/`) and the word pairs are in the repository.
The simulation outputs used in the paper are on Zenodo
([doi:10.5281/zenodo.23497527](https://doi.org/10.5281/zenodo.23497527)). Download the archive and unpack it into `data/`,
so that you get `data/grid/`, `data/tmax/`, `data/density/`,
`data/metastable_K/` and `data/metastable_N/`. `data/README.md` describes every
file and field.

## Reproducing the tables and figures

Run the scripts from the repository root (they live in `scr/`), in this order:

```
python scr/analyze_friedman.py
python scr/analyze_friedman.py density
python scr/analyze_friedman_per_model.py
python scr/analyze_friedman_rq2.py
python scr/analyze_density_metastable.py
python scr/plot_slopegraph.py
python scr/plot_isosuccess.py
python scr/plot_pstar.py
python scr/plot_density.py
python scr/run_metastable_N.py plotlog
python scr/paper_numbers.py
```

Tables (CSV) are written to `tables/` and figures (PDF and PNG) to `figures/`.

| Paper | Script | Output |
|---|---|---|
| Table 1, degree bins | `paper_numbers.py` | printed |
| Section 3.1, winning word and consensus counts | `paper_numbers.py` | printed |
| Figure 1, Figure A1 | `plot_slopegraph.py` | `fig_slopegraph_{mode,mean}_edge_N10000` |
| Table A1 | `analyze_friedman_per_model.py` | `friedman_per_model_tests.csv` |
| Table A2, Figure A2 | `analyze_friedman.py density`, `plot_density.py` | `density_tests.csv`, `fig_density_rrn_mode` |
| Figure 2, Figures A5-A7, Table A5, tau statistics | `plot_isosuccess.py`, `paper_numbers.py` | `fig1e_isosuccess_*`, `isosuccess_*.csv` |
| Annealed vs quenched convergence time | `paper_numbers.py` | printed |
| Tables 2 and A3 | `paper_numbers.py` | printed |
| Figure A3 | `run_metastable_N.py plotlog` | `T_vs_N_log` |
| Table A4, Figure A4 | `analyze_density_metastable.py` | `metastable_K_fits.csv`, `T_vs_K` |
| Figure 3 | `plot_pstar.py` | `fig3a_topology`, `fig3b_placement` |
| Tables 3, 4 and A8, p* counts | `paper_numbers.py`, `analyze_friedman_rq2.py` | printed, `rq2_descriptives.csv` |
| Tables A6 and A7 | `analyze_friedman_rq2.py` | `rq2_tests.csv` |

## Running the simulations

The simulations are written in numba and are seeded. For the paper they were run on a Slurm cluster;
the runners take `--task-id` and `--num-tasks` to split the work over an array
job, and `--max-processes` for local parallelism. Use `--dry-run` to list the
conditions and get an upper bound on the CPU time.

```
python scr/run_grid.py pool                          # cache of 200 graphs per (topology, N)
python scr/run_grid.py rq1                           # edge rule, all topologies and N
python scr/run_grid.py rq1 --rule node
python scr/run_grid.py rq1 --rule annealed --topos ucm_2.5 --ns 10000
python scr/run_grid.py rq2                           # committed minorities, N = 10000

# Llama {young/old}, {more/less} and the control {African/White}, 300 runs
L="--models llama31_70B --ids 92 219 1161 --iters 300"
python scr/run_grid.py rq1 --rule annealed --topos ucm_5.0 rrn $L
python scr/run_grid.py rq1 --topos fc ucm_5.0 rrn --ns 100 200 300 500 2000 --no-pool $L
python scr/run_grid.py rq1 --rule annealed --topos ucm_5.0 rrn --ns 100 200 300 500 2000 --no-pool $L

python scr/run_tmax.py                               # longer runs for the non-converged ones
python scr/run_density.py pool && python scr/run_density.py run
python scr/run_density_metastable.py run
python scr/run_metastable_N.py run
```

## Files

| File | Content |
|---|---|
| `scr/emergence_dynamics.py` | agent state machine and well-mixed simulator of Flint et al. (2026), unchanged |
| `scr/networks.py` | UCM and random regular graph generators |
| `scr/run_grid.py` | consensus runs on every topology and rule; committed-minority bisection |
| `scr/run_tmax.py` | replays non-converged runs with t_max = 1e5 |
| `scr/run_density.py` | RRN degree sweep at N = 10000 |
| `scr/run_density_metastable.py` | RRN degree sweep at N = 100 and 200 for the non-converging Llama pairs |
| `scr/run_metastable_N.py` | convergence time against N for the same pairs |
| `scr/analyze_*.py`, `scr/plot_*.py`, `scr/paper_numbers.py` | statistics, tables and figures |
| `scr/paths.py`, `scr/figstyle.py` | data locations and plot style |

## Citation

If you use this code, please cite the paper (see `CITATION.cff`) and, for the
agent policies, Flint et al. (2026), Group size effects and collective
misalignment in LLM multi-agent systems, PNAS: https://www.pnas.org/doi/full/10.1073/pnas.2531697123.

The word pairs come from CrowS-Pairs.

## License

Code: MIT (see `LICENSE`). Data on Zenodo: CC BY 4.0.
