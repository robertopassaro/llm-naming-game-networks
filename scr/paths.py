"""Where inputs, simulation outputs, tables and figures live.

Everything is relative to the repository root, the parent of this folder.
Set CN_DATA to use a data directory somewhere else (for example on scratch
storage).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("CN_DATA", os.path.join(ROOT, "data"))

# inputs
Q_DICTS = os.path.join(DATA, "q_dicts")
WORD_PAIRS = os.path.join(DATA, "crows_pairs_stereo_sample.pkl")

# simulation outputs
GRID = os.path.join(DATA, "grid")                  # run_grid.py
TMAX = os.path.join(DATA, "tmax")                  # run_tmax.py
DENSITY = os.path.join(DATA, "density")            # run_density.py
METASTABLE_K = os.path.join(DATA, "metastable_K")  # run_density_metastable.py
METASTABLE_N = os.path.join(DATA, "metastable_N")  # run_metastable_N.py

# cached graphs, regenerated on demand from fixed seeds
NETWORKS = os.path.join(ROOT, "networks_cache")
NETWORKS_DENSITY = os.path.join(ROOT, "networks_cache_density")

TABLES = os.path.join(ROOT, "tables")
FIGURES = os.path.join(ROOT, "figures")


def q_dict_path(model, ID):
    return os.path.join(Q_DICTS, f"Q_dict_{model}_ID_{ID}_-50_100_0.5tmp_BLANK_False.pkl")
