#!/usr/bin/env python3
"""
Optimized isomorphism tester using graph_optimized (CSR + Numba).
================================================================

Same testing logic as iso_tester.py but uses optimized functions from
graph_optimized.py:
  - Numba JIT-compiled BFS network building (numpy-vectorized, single-thread)
  - CSR-based inverse network (transpose instead of dict-based inverse_network)
  - Combined network + BDP computation (single Numba pass)
  - CSR reuse for multiple vertices of the same graph

Supports two modes:
  1. Exhaustive pairwise testing within groups (graph6 files).
     Graphs in each file are all non-isomorphic to each other.
  2. Paired testing (DIMACS directories). Each pair of files
     (-1/-2 suffix) is a non-isomorphic pair.

Invariants (by priority):
  1. compute_bidirectional_degree_profiles
  2. find_loops_and_dead_end_branches
  3. get_loops_and_dead_end_branches_intersections

Usage (CLI):
  python iso_tester_optimized.py <file1.g6[.gz]> [file2 ...] [options]
  python iso_tester_optimized.py --pairs <dir1> [dir2 ...] [options]

Options:
  --pairs DIR       Test paired DIMACS files from directory (non-isomorphic pairs)
  --workers N       Number of parallel worker processes (default: 1)
  --fp-dir DIR      Directory for FP files (default: fp_results)
  --no-progress     No progress bar
  --range START END Test only pairs in range [START, END) (0-based, END exclusive)

Usage (Python / Spyder):
  from iso_tester_optimized import run_test

  # Sequential (1 core)
  run_test(['sr401224.g6'])

  # Parallel (6 cores)
  run_test(['sr401224.g6'], workers=6)

  # Test only pairs 0..100 from file with 378 pairs
  run_test(['sr401224.g6'], workers=6, pair_range=(0, 100))

  # Test second batch
  run_test(['sr401224.g6'], workers=6, pair_range=(100, 200))

  # DIMACS paired testing with 4 workers
  run_test(['/path/to/dimacs_dir'], workers=4)
"""

import gzip
import multiprocessing
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_optimized import (
    build_network_and_bdp,
    adj_dict_to_csr,
    EdgeSeenBuffer,
    find_loops_and_dead_end_branches,
    get_loops_and_dead_end_branches_intersections,
    _flat_tuple,
)


# --- Module-level shared data for multiprocessing workers (g6 mode) ----------
# Set by _init_g6_worker() in each child process via Pool initializer.
# This approach works reliably with ALL start methods (fork, spawn, forkserver).
_g6_adjs = None          # list of all adjacency dicts
_g6_pair_indices = None  # numpy array of shape (N, 2), dtype=int32


# --- graph6 parser -----------------------------------------------------------

def _resolve_path(filepath: str) -> str:
    """If file not found, tries adding/removing .gz / .tar.gz and searching
    relative to the script directory."""
    if os.path.exists(filepath):
        return filepath
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_rel = os.path.join(script_dir, filepath)
    if os.path.exists(script_rel):
        return script_rel

    # Try -tar.gz / .tar.gz extension
    for suffix in ['-tar.gz', '.tar.gz']:
        candidate_name = filepath + suffix if not filepath.endswith(suffix) else filepath
        for candidate in [candidate_name, os.path.join(script_dir, candidate_name)]:
            if os.path.exists(candidate):
                return candidate

    # Try .gz extension (but only if not already ending with tar.gz)
    if not filepath.endswith('-tar.gz') and not filepath.endswith('.tar.gz'):
        if filepath.endswith('.gz'):
            bare = filepath[:-3]
            for candidate in [bare, os.path.join(script_dir, bare)]:
                if os.path.exists(candidate):
                    return candidate
        else:
            gz = filepath + '.gz'
            for candidate in [gz, os.path.join(script_dir, gz)]:
                if os.path.exists(candidate):
                    return candidate

    return filepath


def load_graphs_g6(filepath: str) -> list[nx.Graph]:
    """
    Loads graphs from a graph6 format file (.g6 or .g6.gz).
    Each graph is simple, connected, and non-isomorphic to the others.
    """
    filepath = _resolve_path(filepath)
    opener = gzip.open if filepath.endswith('.gz') else open
    graphs = []
    with opener(filepath, 'rb') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                G = nx.from_graph6_bytes(line)
                if G.number_of_nodes() > 0:
                    graphs.append(G)
            except Exception as exc:
                print(f"  ERROR parsing line: {line[:40]}... - {exc}",
                      file=sys.stderr)
    return graphs


# --- DIMACS parser ------------------------------------------------------------

def load_graph_dimacs(filepath: str) -> nx.Graph:
    """
    Loads a single graph from DIMACS format file.
    Format:
      p edge <n> <m>    -- header
      e <u> <v>         -- edges (1-indexed)
    """
    G = nx.Graph()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('p '):
                parts = line.split()
                if len(parts) >= 3:
                    n = int(parts[2])
                    G.add_nodes_from(range(n))
            elif line.startswith('e '):
                parts = line.split()
                if len(parts) >= 3:
                    u, v = int(parts[1]) - 1, int(parts[2]) - 1
                    G.add_edge(u, v)
    return G


def discover_pairs(directory: str) -> list[tuple[str, str, str]]:
    """
    Discovers non-isomorphic pairs in a DIMACS benchmark directory.

    Looks for files with suffix -1 and -2 (same base name).
    Returns list of (base_name, file1_path, file2_path).
    """
    files = sorted(os.listdir(directory))
    pair_map = {}

    for fname in files:
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue
        m = re.match(r'^(.+)-([12])$', fname)
        if m:
            base = m.group(1)
            idx = int(m.group(2))
            if base not in pair_map:
                pair_map[base] = {}
            pair_map[base][idx] = fpath

    pairs = []
    for base in sorted(pair_map.keys()):
        mapping = pair_map[base]
        if 1 in mapping and 2 in mapping:
            pairs.append((base, mapping[1], mapping[2]))

    return pairs


# --- nx.Graph -> adjacency dict conversion ------------------------------------

def nx_to_adj(G: nx.Graph) -> dict:
    """Converts nx.Graph to adjacency dict {int: [int, ...]}."""
    adj = {}
    for v in G.nodes():
        adj[v] = sorted(list(G.neighbors(v)))
    return adj


# --- Grouping (graph6 mode) --------------------------------------------------

def degree_sequence_key(G: nx.Graph) -> tuple:
    """Grouping key: (n, sorted_degree_sequence)."""
    n = G.number_of_nodes()
    degs = sorted(d for _, d in G.degree())
    return (n, tuple(degs))


def group_graphs(graphs: list[nx.Graph]) -> dict:
    """
    Groups graphs by (n, degree_sequence).
    Graphs with different edge counts always end up in different groups,
    since sum(degree_sequence) = 2 * |E|.
    """
    groups = defaultdict(list)
    for G in graphs:
        groups[degree_sequence_key(G)].append(G)
    return dict(groups)


# --- test_isomorphism_exhaustive (optimized with CSR + Numba) -----------------

def test_isomorphism_exhaustive(g1_adj: dict, g2_adj: dict):
    """
    Iterates each vertex of g1 against vertices of g2 with the same degree.
    Invariants are checked by priority:
      1. compute_bidirectional_degree_profiles (CSR + Numba)
      2. find_loops_and_dead_end_branches (CSR-transpose inverse)
      3. get_loops_and_dead_end_branches_intersections
    The next invariant is computed only if the previous one matched.

    Uses CSR reuse: builds CSR once per graph, reuses for all vertices.
    Uses fast inverse network: CSR transpose via Numba instead of
    dict-based inverse_network.

    Returns:
        tuple: (fp_count, total_count, matches, max_depth)
        fp_count    -- vertices of g1 with at least one FP match
        total_count -- total number of vertices in g1
        matches     -- list of all (v1, v2) pairs that matched
        max_depth   -- maximum depth reached by any vertex pair (0..3):
                      0 -- no pair passed BDP
                      1 -- at least one pair passed BDP, but not inv2
                      2 -- at least one pair passed BDP+inv2, but not inv3
                      3 -- at least one pair passed all three (FP)
    """
    n1 = len(g1_adj)
    n2 = len(g2_adj)

    if n1 != n2:
        return (0, n1, [], 0)

    e1 = sum(len(vs) for vs in g1_adj.values())
    e2 = sum(len(vs) for vs in g2_adj.values())
    if e1 != e2:
        return (0, n1, [], 0)

    deg_seq_1 = sorted(len(g1_adj[v]) for v in g1_adj)
    deg_seq_2 = sorted(len(g2_adj[v]) for v in g2_adj)
    if deg_seq_1 != deg_seq_2:
        return (0, n1, [], 0)

    # Lazy computation of all three invariants for g2 vertices.
    # Each invariant is computed on-demand and cached, so that
    # each vertex's invariant is computed at most once.
    # This avoids V^2 redundant computations when iterating all
    # (v1, v2) pairs (e.g., for strongly regular graphs).
    csr2 = adj_dict_to_csr(g2_adj)
    esb2 = EdgeSeenBuffer(csr2[4])

    # Invariant 1 (BDP) cache for g2
    g2_data = {}

    def get_bdp(v2):
        if v2 not in g2_data:
            g2_data[v2] = build_network_and_bdp(g2_adj, v2, _csr_cache=csr2,
                                                 _edge_seen=esb2)
        return g2_data[v2]

    # Invariant 2 (loops/dead-end branches) cache for g2
    g2_ld2 = {}

    def get_ld2(v2):
        if v2 not in g2_ld2:
            d = get_bdp(v2)
            g2_ld2[v2] = find_loops_and_dead_end_branches(
                d.network, layer_degree_map=d.rdm,
                _flat=_flat_tuple(d))
        return g2_ld2[v2]

    # Invariant 3 (intersections) cache for g2
    # Uses the original graph.py implementation for correctness
    g2_ld3 = {}

    def get_ld3(v2):
        if v2 not in g2_ld3:
            _, ld2_res2 = get_ld2(v2)
            g2_ld3[v2] = get_loops_and_dead_end_branches_intersections(ld2_res2)
        return g2_ld3[v2]

    fp_count = 0
    matches = []
    max_depth = 0

    # CSR + EdgeSeenBuffer for g1 (reuse across all v1)
    csr1 = adj_dict_to_csr(g1_adj)
    esb1 = EdgeSeenBuffer(csr1[4])

    for v1 in g1_adj:
        d1_data = build_network_and_bdp(g1_adj, v1, _csr_cache=csr1,
                                             _edge_seen=esb1)

        ld2_1 = None
        ld3_1 = None

        d1 = len(g1_adj[v1])
        v1_matched = False

        for v2 in g2_adj:
            if len(g2_adj[v2]) != d1:
                continue

            # Inv1: BDP fingerprint comparison (np.array_equal — 100% correct)
            d2 = get_bdp(v2)
            if not np.array_equal(d1_data.fwd_fp, d2.fwd_fp):
                continue
            if not np.array_equal(d1_data.rev_fp, d2.rev_fp):
                continue

            if max_depth < 1:
                max_depth = 1

            if ld2_1 is None:
                ld2_1 = find_loops_and_dead_end_branches(
                    d1_data.network, layer_degree_map=d1_data.rdm,
                    _flat=_flat_tuple(d1_data))
            ld2_2 = get_ld2(v2)

            if ld2_1[0] != ld2_2[0]:
                continue

            if max_depth < 2:
                max_depth = 2

            if ld3_1 is None:
                ld3_1 = get_loops_and_dead_end_branches_intersections(ld2_1[1])
            ld3_2 = get_ld3(v2)

            # Compare Inv3: flat int64 fingerprint arrays via np.array_equal
            inv3_match = np.array_equal(ld3_1, ld3_2)

            if inv3_match:
                matches.append((v1, v2))
                v1_matched = True
                max_depth = 3

        if v1_matched:
            fp_count += 1

    return (fp_count, n1, matches, max_depth)


# --- Density zone determination -----------------------------------------------

def density_zone(density: float) -> str:
    if density < 50.0 - 1e-9:
        return 'low'
    elif density > 50.0 + 1e-9:
        return 'high'
    else:
        return 'mid'


# --- Progress bar helper ------------------------------------------------------

def _show_progress(tested: int, total: int, fp: int, t_start: float,
                   suffix: str = ""):
    """Print a single-line progress bar to stderr."""
    elapsed = time.perf_counter() - t_start
    rate = tested / elapsed if elapsed > 0 else 0
    pct = tested / total * 100 if total > 0 else 0
    eta = (total - tested) / rate if rate > 0 else 0
    sys.stderr.write(
        f"\r  Progress: {tested:>10,}/{total:,} "
        f"({pct:>5.1f}%)  |  "
        f"FP: {fp:>6}  |  "
        f"{rate:>8,.0f} pairs/s ETA: {eta:>5.0f}s{suffix}  "
    )
    sys.stderr.flush()


# --- Worker functions for multiprocessing -------------------------------------

def _init_g6_worker(adjs, pair_indices_np):
    """Initialize worker process with shared data.

    Called once per worker at pool creation time. The data is sent via
    Pool initargs (pickled once per worker), guaranteeing availability
    regardless of the start method (fork, spawn, or forkserver).
    """
    global _g6_adjs, _g6_pair_indices
    _g6_adjs = adjs
    _g6_pair_indices = pair_indices_np


def _process_g6_pair_by_idx(pair_idx):
    """Worker for parallel g6 pair processing using shared data index.

    _g6_adjs and _g6_pair_indices are set by _init_g6_worker(), which
    runs once per worker at pool creation time. Only the integer pair_idx
    is sent via IPC per task.
    Returns (pair_idx, result, error) where error is None on success.
    """
    try:
        row = _g6_pair_indices[pair_idx]
        idx_i, idx_j = int(row[0]), int(row[1])
        adj1 = _g6_adjs[idx_i]
        adj2 = _g6_adjs[idx_j]
        return pair_idx, test_isomorphism_exhaustive(adj1, adj2), None
    except Exception as exc:
        return pair_idx, None, str(exc)


def _process_dimacs_pair(args):
    """Worker for parallel DIMACS pair processing."""
    base, f1, f2 = args
    try:
        G1 = load_graph_dimacs(f1)
        G2 = load_graph_dimacs(f2)
        adj1 = nx_to_adj(G1)
        adj2 = nx_to_adj(G2)
        return (True, test_isomorphism_exhaustive(adj1, adj2))
    except Exception as exc:
        return (False, str(exc), base)


# --- Mode 1: Exhaustive pairwise testing (graph6) -----------------------------

def _build_g6_pair_lists(groups):
    """
    Pre-generate pair indices and metadata from grouped graphs.

    Instead of storing full adjacency dicts per pair (which causes massive
    memory and IPC overhead with millions of pairs), we store adjacency
    dicts once in a flat list and reference them by index. For parallel
    processing on Linux, workers inherit the flat list via fork (COW),
    so only integer indices need to be sent via IPC.

    Returns
    -------
    adjs : list of adjacency dicts (one per graph, in order)
    pair_indices : list of (i, j) index pairs into adjs
    pair_meta : list of (density, zone) per pair
    """
    adjs = []
    pair_indices = []
    pair_meta = []

    for key, gs in sorted(groups.items()):
        n, deg_seq = key
        m = len(gs)
        if m < 2:
            continue

        base_idx = len(adjs)
        for G in gs:
            adjs.append(nx_to_adj(G))

        e_k = sum(len(vs) for vs in adjs[base_idx].values()) // 2
        max_e = n * (n - 1) // 2
        density = e_k / max_e * 100 if max_e > 0 else 0.0
        zone = density_zone(density)

        for i in range(m):
            for j in range(i + 1, m):
                pair_indices.append((base_idx + i, base_idx + j))
                pair_meta.append((density, zone))

    return adjs, pair_indices, pair_meta


def run_test_g6(filepaths: list[str], fp_dir: str = "fp_results",
                show_progress: bool = True, workers: int = 1,
                pair_range: tuple[int, int] | None = None):
    """
    Main testing loop for graph6 files.

    Parameters
    ----------
    filepaths   : list of .g6 / .g6.gz file paths
    fp_dir      : directory for false positive output files
    show_progress : show progress bar on stderr
    workers     : number of parallel worker processes (1 = sequential)
    pair_range  : optional (start, end) to test only a subset of pairs.
                  Uses Python slicing: [start, end), 0-based.
                  Example: pair_range=(0, 100) tests pairs 0..99 inclusive.
    """
    t_total_start = time.perf_counter()

    all_graphs = []
    file_labels = []
    for fp in filepaths:
        label = os.path.basename(fp).replace('.g6', '').replace('.gz', '')
        print(f"Loading: {fp}")
        graphs = load_graphs_g6(fp)
        print(f"  Loaded: {len(graphs)} graphs")
        all_graphs.extend(graphs)
        file_labels.append(label)

    num_graphs = len(all_graphs)
    if num_graphs == 0:
        print("  No graphs found!")
        return

    max_n = max(G.number_of_nodes() for G in all_graphs)
    min_n = min(G.number_of_nodes() for G in all_graphs)
    print(f"\n  Total graphs loaded: {num_graphs}")
    print(f"  Vertices: min={min_n}, max={max_n}")

    groups = group_graphs(all_graphs)
    num_groups = len(groups)

    # Build pair lists (indices + metadata, no duplicate adjacency data)
    adjs, pair_indices, pair_meta = _build_g6_pair_lists(groups)
    total_pairs_full = len(pair_indices)

    # Apply pair_range
    if pair_range is not None:
        start, end = pair_range
        pair_indices = pair_indices[start:end]
        pair_meta = pair_meta[start:end]

    total_pairs = len(pair_indices)

    print(f"\n  Groups (by n + degree_seq): {num_groups}")
    print(f"  Total pairs: {total_pairs_full:,}")
    if pair_range is not None:
        print(f"  Range: [{pair_range[0]}, {pair_range[1]}) — testing {total_pairs:,} of {total_pairs_full:,} pairs")

    if total_pairs == 0:
        print("\nNo pairs to test.")
        return

    os.makedirs(fp_dir, exist_ok=True)
    result_label = "+".join(file_labels) if len(file_labels) > 1 else file_labels[0]
    fp_filename = f"fp_{result_label}.txt"
    fp_filepath = os.path.join(fp_dir, fp_filename)

    print(f"\n{'='*72}")
    print(f"  EXHAUSTIVE TEST: {total_pairs:,} pairs (OPTIMIZED)")
    if pair_range is not None:
        print(f"  Range: [{pair_range[0]}, {pair_range[1]}) of {total_pairs_full:,} total pairs")
    print(f"  Invariants: 1) BDP  2) Loops  3) Intersections")
    print(f"  Backend: CSR + Numba")
    if workers > 1:
        print(f"  Workers: {workers}")
    print(f"{'='*72}")

    total_fp = 0
    total_tested = 0
    fp_all_count = 0
    fp_details = []
    zone_stats = defaultdict(lambda: {'total': 0, 'fp': 0})
    depth_count = {0: 0, 1: 0, 2: 0, 3: 0}
    t_test_start = time.perf_counter()

    def _process_result(result, pidx):
        """Process a single test result and update accumulators."""
        nonlocal total_fp, fp_all_count
        fp_count, total_count, matches, max_depth = result
        density, zone = pair_meta[pidx]
        idx_i, idx_j = pair_indices[pidx]
        adj_i = adjs[idx_i]
        adj_j = adjs[idx_j]

        zone_stats[zone]['total'] += 1
        depth_count[max_depth] += 1

        if fp_count > 0:
            total_fp += 1
            is_all = (fp_count == total_count)
            if is_all:
                fp_all_count += 1
            zone_stats[zone]['fp'] += 1

            fp_details.append({
                'density': density,
                'zone': zone,
                'fp_count': fp_count,
                'total_count': total_count,
                'is_all': is_all,
                'matches': matches,
                'g1': adj_i,
                'g2': adj_j,
            })

    if workers <= 1:
        # ---- Sequential mode ----
        progress_step = max(1, total_pairs // 200)
        for pidx in range(total_pairs):
            total_tested += 1
            idx_i, idx_j = pair_indices[pidx]
            result = test_isomorphism_exhaustive(adjs[idx_i], adjs[idx_j])
            _process_result(result, pidx)

            if show_progress and total_tested % progress_step == 0:
                _show_progress(total_tested, total_pairs, total_fp,
                               t_test_start)
    else:
        # ---- Parallel mode with Pool + initializer ----
        # Data is sent to each worker ONCE via Pool initargs (pickled once per
        # worker), then only integer pair_idx is sent per task. This avoids
        # serializing adjacency dicts for each of the millions of tasks.
        # Works reliably with ALL start methods (fork, spawn, forkserver).

        # Convert pair_indices to numpy array for efficient pickling:
        # 5.3M Python tuples (~284 MB) → numpy int32 array (~42 MB)
        pair_indices_np = np.array(pair_indices, dtype=np.int32)

        # chunksize controls how many tasks are sent to a worker at once.
        # Too large = long delay before first result (appears to hang).
        # Too small = high IPC overhead. Target: ~1-5 seconds of work per chunk.
        # At ~50-100 pairs/s per worker, 200 pairs/chunk ≈ 2-4 seconds.
        chunksize = min(200, max(1, total_pairs // (workers * 500)))
        progress_step = max(1, total_pairs // 200)

        if show_progress:
            data_mb = pair_indices_np.nbytes / 1024 / 1024
            print(f"  Sending data to {workers} workers "
                  f"(indices: {data_mb:.0f} MB + adjs: {len(adjs)} dicts/worker)...")

        with multiprocessing.Pool(
            processes=workers,
            initializer=_init_g6_worker,
            initargs=(adjs, pair_indices_np),
        ) as pool:
            if show_progress:
                sys.stderr.write("  Workers started, processing...\n")
                sys.stderr.flush()

            for pair_idx, result, error in pool.imap_unordered(
                _process_g6_pair_by_idx, range(total_pairs),
                chunksize=chunksize
            ):
                total_tested += 1

                if error is not None:
                    print(f"\n  ERROR processing pair {pair_idx}: {error}",
                          file=sys.stderr)
                    depth_count[0] += 1
                else:
                    _process_result(result, pair_idx)

                if show_progress and total_tested % progress_step == 0:
                    _show_progress(total_tested, total_pairs, total_fp,
                                   t_test_start)

    t_test_end = time.perf_counter()
    test_time = t_test_end - t_test_start

    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    # Save FP details
    if fp_details:
        fp_details.sort(key=lambda x: x['density'])
        with open(fp_filepath, 'w') as f:
            for entry in fp_details:
                density_str = f"{entry['density']:.2f}%"
                count_str = 'all' if entry['is_all'] else f"{entry['fp_count']}/{entry['total_count']}"
                line = (f"{density_str}  {count_str}  {entry['matches']}  "
                        f"{entry['g1']}  {entry['g2']}\n")
                f.write(line)
        print(f"\n  FP saved: {fp_filepath} ({len(fp_details)} entries)")

    _print_statistics(result_label, total_tested, total_fp, fp_all_count,
                      test_time, depth_count, zone_stats, t_total_start)

    if total_fp > 0:
        _print_fp_details(total_fp, fp_all_count, total_tested)

    print(f"\n{'='*72}")


# --- Mode 2: Paired testing (DIMACS directories) ------------------------------

def run_test_pairs(directories: list[str], fp_dir: str = "fp_results",
                   show_progress: bool = True, workers: int = 1,
                   pair_range: tuple[int, int] | None = None):
    """
    Tests non-isomorphic pairs from DIMACS benchmark directories.
    Each pair (-1/-2 suffix) is tested individually.

    Parameters
    ----------
    directories : list of directory paths containing DIMACS paired files
    fp_dir      : directory for false positive output files
    show_progress : show progress bar on stderr
    workers     : number of parallel worker processes (1 = sequential)
    pair_range  : optional (start, end) to test only a subset of pairs.
                  Uses Python slicing: [start, end), 0-based.
    """
    t_total_start = time.perf_counter()

    all_pairs = []
    dir_labels = []
    for directory in directories:
        label = os.path.basename(directory.rstrip('/'))
        print(f"Scanning: {directory}")
        pairs = discover_pairs(directory)
        print(f"  Found {len(pairs)} pairs")
        all_pairs.extend(pairs)
        dir_labels.append(label)

    num_pairs_full = len(all_pairs)

    # Apply pair_range
    if pair_range is not None:
        start, end = pair_range
        all_pairs = all_pairs[start:end]

    num_pairs = len(all_pairs)
    if num_pairs == 0:
        print("  No pairs found!")
        return

    print(f"\n  Total pairs: {num_pairs_full}")
    if pair_range is not None:
        print(f"  Range: [{pair_range[0]}, {pair_range[1]}) — testing {num_pairs} of {num_pairs_full} pairs")
    print(f"\n  Size distribution:")

    size_counts = defaultdict(int)
    for base, f1, f2 in all_pairs:
        m = re.search(r'-(\d+)-\d+$', base)
        if m:
            size_counts[int(m.group(1))] += 1

    for sz in sorted(size_counts.keys()):
        print(f"    {sz:>6} vertices: {size_counts[sz]:>4} pairs")

    os.makedirs(fp_dir, exist_ok=True)
    result_label = "+".join(dir_labels) if len(dir_labels) > 1 else dir_labels[0]
    fp_filename = f"fp_pairs_{result_label}.txt"
    fp_filepath = os.path.join(fp_dir, fp_filename)

    print(f"\n{'='*72}")
    print(f"  PAIRED TEST: {num_pairs} pairs (non-isomorphic, OPTIMIZED)")
    if pair_range is not None:
        print(f"  Range: [{pair_range[0]}, {pair_range[1]}) of {num_pairs_full} total pairs")
    print(f"  Invariants: 1) BDP  2) Loops  3) Intersections")
    print(f"  Backend: CSR + Numba (vectorized)")
    if workers > 1:
        print(f"  Workers: {workers}")
    print(f"{'='*72}")

    total_fp = 0
    total_tested = 0
    fp_details = []
    depth_count = {0: 0, 1: 0, 2: 0, 3: 0}
    t_test_start = time.perf_counter()

    if workers <= 1:
        # ---- Sequential mode ----
        progress_step = max(1, num_pairs // 200)
        progress_counter = 0

        for pair_idx, (base, f1, f2) in enumerate(all_pairs):
            total_tested += 1
            progress_counter += 1

            m = re.search(r'-(\d+)-\d+$', base)
            n_verts = int(m.group(1)) if m else '?'

            try:
                G1 = load_graph_dimacs(f1)
                G2 = load_graph_dimacs(f2)
                adj1 = nx_to_adj(G1)
                adj2 = nx_to_adj(G2)
                fp_count, total_count, matches, max_depth = test_isomorphism_exhaustive(adj1, adj2)

                depth_count[max_depth] += 1

                if fp_count > 0:
                    total_fp += 1
                    is_all = (fp_count == total_count)
                    fp_details.append({
                        'n_verts': n_verts, 'base': base,
                        'fp_count': fp_count, 'total_count': total_count,
                        'is_all': is_all, 'matches': matches,
                        'max_depth': max_depth,
                    })

            except Exception as exc:
                print(f"\n  ERROR processing {base}: {exc}", file=sys.stderr)
                depth_count[0] += 1

            if show_progress and progress_counter >= progress_step:
                _show_progress(total_tested, num_pairs, total_fp, t_test_start)
                progress_counter = 0
    else:
        # ---- Parallel mode with incremental progress bar ----
        pair_args = [(base, f1, f2) for base, f1, f2 in all_pairs]

        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {}
            for idx, args in enumerate(pair_args):
                future = pool.submit(_process_dimacs_pair, args)
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                base, f1, f2 = all_pairs[idx]
                m = re.search(r'-(\d+)-\d+$', base)
                n_verts = int(m.group(1)) if m else '?'

                try:
                    result = future.result()
                except Exception as exc:
                    print(f"\n  ERROR processing {base}: {exc}",
                          file=sys.stderr)
                    depth_count[0] += 1
                    total_tested += 1
                    if show_progress:
                        _show_progress(total_tested, num_pairs, total_fp,
                                       t_test_start)
                    continue

                total_tested += 1

                if result[0]:
                    fp_count, total_count, matches, max_depth = result[1]
                    depth_count[max_depth] += 1
                    if fp_count > 0:
                        total_fp += 1
                        is_all = (fp_count == total_count)
                        fp_details.append({
                            'n_verts': n_verts, 'base': base,
                            'fp_count': fp_count, 'total_count': total_count,
                            'is_all': is_all, 'matches': matches,
                            'max_depth': max_depth,
                        })
                else:
                    print(f"\n  ERROR processing {base}: {result[2]}",
                          file=sys.stderr)
                    depth_count[0] += 1

                if show_progress:
                    _show_progress(total_tested, num_pairs, total_fp,
                                   t_test_start)

    t_test_end = time.perf_counter()
    test_time = t_test_end - t_test_start

    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    if fp_details:
        fp_details.sort(key=lambda x: x['n_verts'])
        with open(fp_filepath, 'w') as f:
            for entry in fp_details:
                count_str = 'all' if entry['is_all'] else f"{entry['fp_count']}/{entry['total_count']}"
                line = (f"n={entry['n_verts']}  {count_str}  depth={entry['max_depth']}  "
                        f"matches={entry['matches']}  {entry['base']}\n")
                f.write(line)
        print(f"\n  FP saved: {fp_filepath} ({len(fp_details)} entries)")

    t_total_end = time.perf_counter()
    total_time = t_total_end - t_total_start

    fp_rate = total_fp / total_tested * 100 if total_tested > 0 else 0
    tn_pct = (total_tested - total_fp) / total_tested * 100 if total_tested > 0 else 0

    print(f"\n{'='*72}")
    print(f"  FINAL STATISTICS: {result_label} (OPTIMIZED)")
    print(f"{'='*72}")

    inv_labels = {
        0: '#1 BDP cut completely',
        1: '#2 Loops cut (after BDP)',
        2: '#3 Intersections cut',
        3: 'FP (not cut)',
    }

    print(f"\n  Invariant statistics (by graph pairs):\n")
    print(f"  {'Group':>35}  {'Pairs':>10}  {'Share':>8}  {'Cumul.%':>10}")
    print("  " + "-" * 68)
    cumulative = 0
    for d in range(4):
        cnt = depth_count[d]
        pct = cnt / total_tested * 100 if total_tested > 0 else 0
        cumulative += cnt
        cum_pct = cumulative / total_tested * 100 if total_tested > 0 else 0
        marker = ''
        if d == 3 and cnt > 0:
            marker = '  <<< FP'
        print(f"  {inv_labels[d]:>35}  {cnt:>10}  {pct:>7.2f}%  {cum_pct:>9.2f}%{marker}")

    print(f"\n  Graph pairs requiring each invariant:")
    print(f"    Not required (BDP sufficient):    {depth_count[0]:>10}  ({depth_count[0]/total_tested*100:.2f}%)")
    print(f"    Required inv #2 (Loops):          {depth_count[1]:>10}  ({depth_count[1]/total_tested*100:.2f}%)")
    print(f"    Required inv #3 (Intersections): {depth_count[2]:>10}  ({depth_count[2]/total_tested*100:.2f}%)")
    print(f"    FP (all three passed):            {depth_count[3]:>10}  ({depth_count[3]/total_tested*100:.2f}%)")

    print(f"""
  Parameters:
    Directory:              {', '.join(dir_labels)}
    Pairs tested:           {total_tested:,}

  Accuracy:
    Correct (TN):           {total_tested - total_fp:,}  ({tn_pct:.4f}%)
    False positives:        {total_fp:,}  ({fp_rate:.4f}%)
    Test time:              {test_time:.1f} s
    Speed:                  {total_tested/test_time:,.2f} pairs/s
    Total time:             {total_time:.1f} s""")

    if total_fp > 0:
        fp_by_size = defaultdict(int)
        for entry in fp_details:
            fp_by_size[entry['n_verts']] += 1

        print(f"\n  FP by vertex count:")
        for sz in sorted(fp_by_size.keys()):
            print(f"    n={sz:>6}: {fp_by_size[sz]:>4} FP pairs")

    print(f"\n{'='*72}")


# --- Shared statistics printer (g6 mode) -------------------------------------

def _print_statistics(result_label, total_tested, total_fp, fp_all_count,
                      test_time, depth_count, zone_stats, t_total_start):
    t_total_end = time.perf_counter()
    total_time = t_total_end - t_total_start

    fp_rate = total_fp / total_tested * 100 if total_tested > 0 else 0
    tn_pct = (total_tested - total_fp) / total_tested * 100 if total_tested > 0 else 0
    all_fp_pct = fp_all_count / total_fp * 100 if total_fp > 0 else 0
    all_pairs_pct = fp_all_count / total_tested * 100 if total_tested > 0 else 0

    print(f"\n{'='*72}")
    print(f"  FINAL STATISTICS: {result_label} (OPTIMIZED)")
    print(f"{'='*72}")

    inv_labels = {
        0: '#1 BDP cut completely',
        1: '#2 Loops cut (after BDP)',
        2: '#3 Intersections cut',
        3: 'FP (not cut)',
    }

    print(f"\n  Invariant statistics (by graph pairs):\n")
    print(f"  {'Group':>35}  {'Pairs':>10}  {'Share':>8}  {'Cumul.%':>10}")
    print("  " + "-" * 68)
    cumulative = 0
    for d in range(4):
        cnt = depth_count[d]
        pct = cnt / total_tested * 100 if total_tested > 0 else 0
        cumulative += cnt
        cum_pct = cumulative / total_tested * 100 if total_tested > 0 else 0
        marker = ''
        if d == 3 and cnt > 0:
            marker = '  <<< FP'
        print(f"  {inv_labels[d]:>35}  {cnt:>10,}  {pct:>7.2f}%  {cum_pct:>9.2f}%{marker}")

    print(f"\n  Graph pairs requiring each invariant:")
    print(f"    Not required (BDP sufficient):    {depth_count[0]:>10,}  ({depth_count[0]/total_tested*100:.2f}%)")
    print(f"    Required inv #2 (Loops):          {depth_count[1]:>10,}  ({depth_count[1]/total_tested*100:.2f}%)")
    print(f"    Required inv #3 (Intersections): {depth_count[2]:>10,}  ({depth_count[2]/total_tested*100:.2f}%)")
    print(f"    FP (all three passed):            {depth_count[3]:>10,}  ({depth_count[3]/total_tested*100:.2f}%)")

    print(f"""
  Parameters:
    Pairs tested:          {total_tested:,}

  Accuracy:
    Correct (TN):          {total_tested - total_fp:,}  ({tn_pct:.4f}%)
    False positives:       {total_fp:,}  ({fp_rate:.4f}%)
    Of which 'all':        {fp_all_count:,}  ({all_fp_pct:.2f}% of FP)  ({all_pairs_pct:.6f}% of all pairs)
    Test time:             {test_time:.1f} s
    Speed:                 {total_tested/test_time:,.0f} pairs/s
    Total time:            {total_time:.1f} s""")

    print(f"\n  FP aggregation by density zones:\n")
    print(f"  {'Zone':>25}  {'Pairs':>10}  {'FP':>8}  {'TN':>10}  "
          f"{'FP%':>8}  {'FP share':>8}")
    print("  " + "-" * 76)

    zone_names = {
        'low': "Sparse       (<50%)",
        'mid': "Density      (=50%)",
        'high': "Dense        (>50%)",
    }
    zone_order = ['low', 'mid', 'high']

    for z in zone_order:
        zs = zone_stats[z]
        z_total = zs['total']
        z_fp = zs['fp']
        z_tn = z_total - z_fp
        z_fp_pct = z_fp / z_total * 100 if z_total > 0 else 0
        z_share = z_fp / total_fp * 100 if total_fp > 0 and z_fp > 0 else 0
        marker = " <<<" if z_fp_pct >= 10 else ""
        print(f"  {zone_names[z]:>25}  {z_total:>10,}  {z_fp:>8}  "
              f"{z_tn:>10,}  {z_fp_pct:>7.2f}%  {z_share:>7.1f}%{marker}")


def _print_fp_details(total_fp, fp_all_count, total_tested):
    non_all = total_fp - fp_all_count
    non_all_pct = non_all / total_fp * 100
    all_fp_pct = fp_all_count / total_fp * 100
    all_pairs_pct = fp_all_count / total_tested * 100
    print(f"""
  FP details:
    Regular FP (not 'all'):  {non_all:,}  ({non_all_pct:.2f}%)
    Full FP ('all'):         {fp_all_count:,}  ({all_fp_pct:.2f}%)
    'all' of all pairs:      {all_pairs_pct:.6f}%""")


# --- CLI entry point -----------------------------------------------------------

def _extract_tar_to_tmpdir(filepath: str) -> str:
    """
    Extracts a .tar.gz archive into a temporary directory.
    Returns the path to the extracted content directory.
    """
    tmpdir = tempfile.mkdtemp(prefix='lics_bench_')
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(tmpdir)
    entries = os.listdir(tmpdir)
    if len(entries) == 1 and os.path.isdir(os.path.join(tmpdir, entries[0])):
        return os.path.join(tmpdir, entries[0])
    return tmpdir


def run_test(paths: list[str], fp_dir: str = "fp_results",
              show_progress: bool = True, workers: int = 1,
              pair_range: tuple[int, int] | None = None):
    """
    Universal entry point: auto-detects input type and routes to the
    appropriate testing function.

    Accepts a mixed list of:
      - graph6 files  (.g6 / .g6.gz)  -> exhaustive pairwise within file(s)
      - directories                     -> DIMACS paired testing (--pairs logic)
      - .tar.gz archives                -> extract to tmp, DIMACS paired testing

    Parameters
    ----------
    paths         : list of file paths or directories
    fp_dir        : directory for FP output files
    show_progress : show progress bar
    workers       : number of parallel worker processes (1 = sequential)
    pair_range    : optional (start, end) to test only a subset of pairs.
                    0-based, end exclusive. Example: (0, 100) tests pairs 0..99.

    Examples
    --------
    >>> from iso_tester_optimized import run_test

    # Sequential (1 core)
    >>> run_test(['sr401224.g6'])

    # Parallel (6 cores)
    >>> run_test(['sr401224.g6'], workers=6)

    # Test only pairs 0..99
    >>> run_test(['sr401224.g6'], workers=6, pair_range=(0, 100))

    # Test second batch 100..199
    >>> run_test(['sr401224.g6'], workers=6, pair_range=(100, 200))

    # DIMACS paired testing
    >>> run_test(['/path/to/dimacs_dir'], workers=4)
    """
    dirs = []
    g6_files = []
    tmpdirs = []

    for p in paths:
        resolved = _resolve_path(p)
        if os.path.isdir(resolved):
            dirs.append(resolved)
        elif os.path.isfile(resolved) and (
            resolved.endswith('.tar.gz') or resolved.endswith('-tar.gz')
        ):
            print(f"Extracting: {resolved}")
            extracted = _extract_tar_to_tmpdir(resolved)
            tmpdirs.append(extracted)
            dirs.append(extracted)
        else:
            g6_files.append(p)

    try:
        if dirs and not g6_files:
            run_test_pairs(directories=dirs, fp_dir=fp_dir,
                           show_progress=show_progress, workers=workers,
                           pair_range=pair_range)
        elif g6_files and not dirs:
            run_test_g6(filepaths=g6_files, fp_dir=fp_dir,
                        show_progress=show_progress, workers=workers,
                        pair_range=pair_range)
        elif dirs and g6_files:
            run_test_g6(filepaths=g6_files, fp_dir=fp_dir,
                        show_progress=show_progress, workers=workers,
                        pair_range=pair_range)
            print()
            run_test_pairs(directories=dirs, fp_dir=fp_dir,
                           show_progress=show_progress, workers=workers,
                           pair_range=pair_range)
        else:
            print("No valid paths provided.")
    finally:
        for td in tmpdirs:
            try:
                shutil.rmtree(td)
            except OSError:
                pass


def main():
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Optimized isomorphism tester (CSR + Numba)"
    )
    parser.add_argument("files", nargs='*', help="File(s) .g6 / .g6.gz or directories")
    parser.add_argument("--pairs", nargs='+', metavar='DIR',
                        help="DIMACS benchmark directory with paired files (-1/-2)")
    parser.add_argument("--fp-dir", type=str, default="fp_results",
                        help="Directory for FP files (default: fp_results)")
    parser.add_argument("--no-progress", action="store_true",
                        help="No progress bar")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")
    parser.add_argument("--range", type=int, nargs=2, metavar=('START', 'END'),
                        help="Test only pairs in range [START, END) (0-based)")
    args = parser.parse_args()

    pair_range = tuple(args.range) if args.range else None

    if args.pairs:
        run_test_pairs(
            directories=args.pairs,
            fp_dir=args.fp_dir,
            show_progress=not args.no_progress,
            workers=args.workers,
            pair_range=pair_range,
        )
    elif args.files:
        run_test(
            paths=args.files,
            fp_dir=args.fp_dir,
            show_progress=not args.no_progress,
            workers=args.workers,
            pair_range=pair_range,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()