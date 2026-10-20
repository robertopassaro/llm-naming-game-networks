"""Random graph generators: uncorrelated configuration model (UCM) and random
regular network (RRN).

UCM degrees are drawn from P(k) ~ k^-gamma with the structural cutoff
kmax = int(sqrt(N)); stubs are then matched at random, rejecting self-loops and
multi-edges, so the result is a simple uncorrelated graph. The RRN uses the same
wiring with a constant degree sequence.

    generate_ucm(N, gamma, kmin=4, seed=None) -> (edges, degrees)
    generate_rrn(N, k=4, seed=None)           -> (edges, degrees)

edges is int64[M, 2] with each undirected edge listed once; degrees is int64[N].
"""
import contextlib
import io
import os

import numpy as np
from numba import njit
from numba.typed import List


def degree_sequence_powerlaw(N, gamma, k_min=3):
    """
        Sample a degree sequence from P(k) ~ k^{-gamma}
        with structural cutoff k_max ~ sqrt(N) (UCM condition).
        Returns:
            degrees: A degree sequence of length N.
    """

    k_max = int(np.sqrt(N))

    while True:
        degrees = np.zeros(N, dtype=int)
        for i in range(N):
            while True:
                r = (k_min - 0.5)*np.power(np.random.rand(), -1/(gamma-1))
                k_test = int(r)
                if (r - k_test) > 0.5:
                    k_test += 1
                if k_test >= k_min and k_test <= k_max:
                    break
            degrees[i] = k_test

        if degrees.sum() % 2 == 1:
            print("Sum of degrees is odd, resampling...")
        else:
            break

    return degrees


@njit
def generate_network_numba(ki, max_tries=20000):
    """
        Generate a network from a degree sequence using the UCM algorithm
        Numba implementation for speed. The degree sequence is given by ki
        Returns:
            success: boolean indicating if the network was successfully generated
            neighbors: a list of lists containing the neighbors of each node
    """
    L = len(ki)

    total_stubs = 0
    for i in range(L):
        total_stubs += ki[i]

    if total_stubs % 2 != 0:
        empty = List()
        empty.append(List.empty_list(np.int64))
        empty.pop()
        return False, empty

    numlinks = total_stubs // 2

    stub_list = np.empty(total_stubs, dtype=np.int64)
    pos = 0
    for i in range(L):
        for _ in range(ki[i]):
            stub_list[pos] = i
            pos += 1

    nselect = total_stubs

    cki = ki.copy()

    neighbors = List()
    for i in range(L):
        neighbors.append(List.empty_list(np.int64))

    unbalanced = L

    for _ in range(numlinks):

        tries = 0

        while True:
            tries += 1
            if tries > max_tries:
                return False, neighbors

            n1 = np.random.randint(0, nselect)
            n2 = np.random.randint(0, nselect)

            v1 = stub_list[n1]
            v2 = stub_list[n2]

            if v1 == v2:
                continue

            ok = True
            neigh_v1 = neighbors[v1]
            for j in range(len(neigh_v1)):
                if neigh_v1[j] == v2:
                    ok = False
                    break

            if not ok:
                continue

            break

        neighbors[v1].append(v2)
        neighbors[v2].append(v1)

        if n1 > n2:
            tmp = n1
            n1 = n2
            n2 = tmp

        stub_list[n2] = stub_list[nselect - 1]
        nselect -= 1

        stub_list[n1] = stub_list[nselect - 1]
        nselect -= 1

        cki[v1] -= 1
        cki[v2] -= 1

        if cki[v1] == 0:
            unbalanced -= 1
        if cki[v2] == 0:
            unbalanced -= 1

    if unbalanced != 0:
        return False, neighbors

    return True, neighbors

def convert_neighbors(neighbors_typed):
    """
        Convert the neighbors list from numba typed list to a list of numpy arrays.
        Returns:
            out: a list of numpy arrays containing the neighbors of each node.
    """

    out = List()

    for i in range(len(neighbors_typed)):

        arr = np.array(neighbors_typed[i], dtype=np.int64)

        out.append(arr)

    return out

def build_edge_array(neighbors):
    """
        Convert the neighbors list to an edge array, where each edge is represented as a pair of node indices (i, j) with i < j to avoid duplicates.
        Returns:
            edges: a numpy array of shape (num_edges, 2) containing the edge list
    """
    edges = []
    for i in range(len(neighbors)):
        for j in neighbors[i]:
            if j > i:   # avoid duplicates
                edges.append((i, j))
    return np.array(edges, dtype=np.int64)

def generate_ucm_network(N, gamma, k_min=3):
    """
        Generate a UCM network of given size N and exponent gamma,
        Returns a tuple (neighbors, edges) where neighbors is a list of numpy
        arrays containing the neighbors of each node, and edges is a
        numpy array of shape (num_edges, 2) containing the edge list.
    """
    print(f"Generating UCM model network with N={N}, gamma={gamma} and k_min={k_min}...")
    while True:
        degrees = degree_sequence_powerlaw(N, gamma, k_min)
        success, neighbors = generate_network_numba(degrees)
        if success:
            print(f"Generated UCM model network with N={N}, gamma={gamma} and k_min={k_min}")
            return convert_neighbors(neighbors), build_edge_array(neighbors)

def generate_random_regular_network(N, k):
    """
        Generate a Random Regular network of given size N and degree k,
        Returns a tuple (neighbors, edges) where neighbors is a list of numpy
        arrays containing the neighbors of each node, and edges is a
        numpy array of shape (num_edges, 2) containing the edge list.
    """

    assert k < N, "Degree k must be less than the number of nodes N."
    assert k % 2 == 0, "Degree k must be even for a valid degree sequence."

    print(f"Generating random regular network with N={N} and k={k}...")
    degrees = np.full(N, k, dtype=int)

    while True:
        success, neighbors = generate_network_numba(degrees)
        if success:
            print(f"Generated random regular network with N={N} and k={k}")
            return convert_neighbors(neighbors), build_edge_array(neighbors)


@njit(cache=True)
def _numba_seed(s):
    np.random.seed(s)


def _seed_all(seed):
    """Seed numpy (degree sequence) and numba (wiring); they keep separate states."""
    if seed is None:
        return
    np.random.seed(seed % (2**32))
    _numba_seed(seed % (2**31))


# The generators print progress. Set NETWORKS_VERBOSE=1 to see it.
_VERBOSE = bool(int(os.environ.get("NETWORKS_VERBOSE", "0")))


def _quiet():
    return contextlib.nullcontext() if _VERBOSE else \
        contextlib.redirect_stdout(io.StringIO())


def generate_ucm(N, gamma, kmin=4, seed=None):
    """UCM graph with structural cutoff kmax = int(sqrt(N))."""
    _seed_all(seed)
    with _quiet():
        _, edges = generate_ucm_network(N, gamma, k_min=kmin)
    return edges, _degrees_from_edges(edges, N)


def generate_rrn(N, k=4, seed=None):
    """Random k-regular graph."""
    _seed_all(seed)
    with _quiet():
        _, edges = generate_random_regular_network(N, k)
    return edges, _degrees_from_edges(edges, N)


def _degrees_from_edges(edges, N):
    deg = np.zeros(N, dtype=np.int64)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    return deg
