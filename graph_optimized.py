"""
Optimized graph isomorphism testing — flat-array + Numba pipeline.
====================================================================

Pipeline:
  1. adj_dict → CSR (once per graph, reused for all start vertices)
  2. Network + Inv1 (BDP): Numba BFS → flat arrays → fingerprint
  3. Inv2 (Loops): Numba inverse + flat trace → flat arrays → fingerprint
  4. Inv3 (Intersections): vertex-indexed Numba kernel → fingerprint

All intermediate data stays in flat numpy arrays (int64 / uint64).
No nested tuples, no dict-of-dicts in the hot path.
Comparison of invariants uses np.array_equal on canonical fingerprints.

Conversion functions (for correctness checking) translate flat arrays
back to the graph.py nested-tuple format.
"""

import os
import sys
import random
from collections import defaultdict

import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import Graph as _GraphPy


# ═══════════════════════════════════════════════════════════════════
# 1. ADJACENCY DICT → CSR
# ═══════════════════════════════════════════════════════════════════

def adj_dict_to_csr(graph):
    """
    Convert adjacency dict to CSR format.

    Returns (row_ptr, col_idx, vtx2idx, idx2vtx, V)
    """
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)
    sorted_vertices = sorted(all_vertices, key=str)
    V = len(sorted_vertices)

    vtx2idx = {v: i for i, v in enumerate(sorted_vertices)}
    idx2vtx = {i: v for i, v in enumerate(sorted_vertices)}

    row_ptr = np.zeros(V + 1, dtype=np.int64)
    for v in sorted_vertices:
        idx = vtx2idx[v]
        deg = len(graph.get(v, []))
        row_ptr[idx + 1] = row_ptr[idx] + deg

    total_edges = int(row_ptr[V])
    col_idx = np.zeros(total_edges, dtype=np.int64)
    for v in sorted_vertices:
        idx = vtx2idx[v]
        start = int(row_ptr[idx])
        for j, nb in enumerate(graph.get(v, [])):
            col_idx[start + j] = vtx2idx[nb]

    return row_ptr, col_idx, vtx2idx, idx2vtx, V


# ═══════════════════════════════════════════════════════════════════
# 2. EDGE-SEEN BUFFER (O(1) reset via version counter)
# ═══════════════════════════════════════════════════════════════════

class EdgeSeenBuffer:
    def __init__(self, V):
        self.data = np.zeros(V * V, dtype=np.int64)
        self.V = V
        self.version = 1

    def reset(self):
        self.version += 1
        if self.version >= (1 << 60):
            self.data[:] = 0
            self.version = 1


# ═══════════════════════════════════════════════════════════════════
# 3. NUMBA: BFS NETWORK + GLOBAL DEGREES
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True)
def _build_network_kernel(
    row_ptr, col_idx, start_idx, V,
    edge_seen_data, edge_seen_version,
):
    """BFS network builder. Returns flat edge arrays + global degrees."""
    total_directed_edges = len(col_idx)
    ver = edge_seen_version

    edge_src = np.empty(total_directed_edges, dtype=np.int64)
    edge_tgt = np.empty(total_directed_edges, dtype=np.int64)
    layer_off = np.empty(V + 1, dtype=np.int64)
    layer_off[0] = 0

    cur = np.empty(V, dtype=np.int64)
    nxt = np.empty(V, dtype=np.int64)
    seen_cur = np.zeros(V, dtype=np.bool_)
    seen_nxt = np.zeros(V, dtype=np.bool_)

    node_in = np.zeros(V, dtype=np.int64)
    node_out = np.zeros(V, dtype=np.int64)

    cand_src = np.empty(total_directed_edges, dtype=np.int64)
    cand_tgt = np.empty(total_directed_edges, dtype=np.int64)

    cur[0] = start_idx
    num_cur = 1
    total_edges = 0
    num_layers = 0

    for _layer in range(V):
        valid = 0
        for i in range(num_cur):
            n = cur[i]
            if row_ptr[n + 1] > row_ptr[n]:
                cur[valid] = n
                valid += 1
        num_cur = valid
        if num_cur == 0:
            break

        seen_cur[:] = False
        dedup = 0
        for i in range(num_cur):
            n = cur[i]
            if not seen_cur[n]:
                seen_cur[n] = True
                cur[dedup] = n
                dedup += 1
        num_cur = dedup

        frontier = cur[:num_cur]
        lo_all = row_ptr[frontier]
        hi_all = row_ptr[frontier + 1]
        degs = hi_all - lo_all

        total_cand = 0
        for i in range(num_cur):
            total_cand += degs[i]
        if total_cand == 0:
            break

        pos = 0
        for i in range(num_cur):
            d = degs[i]
            u = frontier[i]
            lo = lo_all[i]
            for j in range(d):
                cand_src[pos] = u
                cand_tgt[pos] = col_idx[lo + j]
                pos += 1

        u_arr = np.minimum(cand_src[:total_cand], cand_tgt[:total_cand])
        v_arr = np.maximum(cand_src[:total_cand], cand_tgt[:total_cand])
        edge_idx = u_arr * V + v_arr

        seen_flags = np.empty(total_cand, dtype=np.bool_)
        for k in range(total_cand):
            seen_flags[k] = (edge_seen_data[edge_idx[k]] != ver)

        layer_new = 0
        for k in range(total_cand):
            if seen_flags[k]:
                layer_new += 1
        if layer_new == 0:
            break

        seen_nxt[:] = False
        nxt_count = 0

        for k in range(total_cand):
            if seen_flags[k]:
                u = cand_src[k]
                v = cand_tgt[k]
                edge_seen_data[edge_idx[k]] = ver

                edge_src[total_edges] = u
                edge_tgt[total_edges] = v
                node_out[u] += 1
                node_in[v] += 1
                total_edges += 1

                if not seen_nxt[v]:
                    seen_nxt[v] = True
                    nxt[nxt_count] = v
                    nxt_count += 1

        num_layers += 1
        layer_off[num_layers] = total_edges

        tmp = cur
        cur = nxt
        nxt = tmp
        num_cur = nxt_count

    return edge_src, edge_tgt, layer_off, num_layers, total_edges, node_in, node_out


# ═══════════════════════════════════════════════════════════════════
# 4. NUMBA: BUILD INVERSE FLAT (CSR TRANSPOSE)
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True)
def _build_inverse_flat(e_src, e_tgt, l_off, num_layers):
    """Inverse flat edge arrays by swapping src/tgt, reversing layer order."""
    total = int(l_off[num_layers])
    inv_src = np.empty(total, dtype=np.int64)
    inv_tgt = np.empty(total, dtype=np.int64)
    inv_l_off = np.zeros(num_layers + 1, dtype=np.int64)

    pos = 0
    for inv_l in range(num_layers):
        orig_l = num_layers - 1 - inv_l
        lo = int(l_off[orig_l])
        hi = int(l_off[orig_l + 1])
        cnt = hi - lo
        for k in range(cnt):
            inv_src[pos + k] = e_tgt[lo + k]
            inv_tgt[pos + k] = e_src[lo + k]
        pos += cnt
        inv_l_off[inv_l + 1] = pos

    return inv_src, inv_tgt, inv_l_off


# ═══════════════════════════════════════════════════════════════════
# 5. BDP FINGERPRINT (Inv1)
# ═══════════════════════════════════════════════════════════════════

def build_inv1_fingerprint(e_src, e_tgt, l_off, num_layers, node_in, node_out, V):
    """Build BDP fingerprint arrays. Returns (fwd_fp, rev_fp) as int64 arrays."""
    num_layers = len(l_off) - 1

    fwd_parts = []
    rev_parts = []

    for l in range(num_layers):
        lo = int(l_off[l])
        hi = int(l_off[l + 1])

        fwd_layer = []
        rev_data = defaultdict(list)

        i = lo
        while i < hi:
            u = int(e_src[i])
            j = i
            while j < hi and int(e_src[j]) == u:
                j += 1

            u_deg = (int(node_in[u]), int(node_out[u]))
            tgt_degs = []
            for e in range(i, j):
                v = int(e_tgt[e])
                v_deg = (int(node_in[v]), int(node_out[v]))
                tgt_degs.append(v_deg)
                rev_data[v_deg].append(u_deg)

            fwd_layer.append((u_deg, sorted(tgt_degs)))
            i = j

        fwd_layer.sort()
        fwd_parts.append(len(fwd_layer))
        for src_deg, tgt_degs in fwd_layer:
            fwd_parts.append(src_deg[0])
            fwd_parts.append(src_deg[1])
            fwd_parts.append(len(tgt_degs))
            for td in tgt_degs:
                fwd_parts.append(td[0])
                fwd_parts.append(td[1])

        rev_layer = [(t_deg, sorted(s_degs)) for t_deg, s_degs in rev_data.items()]
        rev_layer.sort()
        rev_parts.append(len(rev_layer))
        for tgt_deg, src_degs in rev_layer:
            rev_parts.append(tgt_deg[0])
            rev_parts.append(tgt_deg[1])
            rev_parts.append(len(src_degs))
            for sd in src_degs:
                rev_parts.append(sd[0])
                rev_parts.append(sd[1])

    return np.array(fwd_parts, dtype=np.int64), np.array(rev_parts, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════
# 6. INV2 — LOOPS AND DEAD-END BRANCHES
# ═══════════════════════════════════════════════════════════════════

def _flat_to_network_dict(e_src, e_tgt, l_off, n_layers, idx2vtx):
    """Convert flat edge arrays to list[dict] network format."""
    network = []
    for l in range(n_layers):
        lo = int(l_off[l])
        hi = int(l_off[l + 1])
        layer = {}
        for e in range(lo, hi):
            src_key = idx2vtx[int(e_src[e])]
            tgt_key = idx2vtx[int(e_tgt[e])]
            if src_key not in layer:
                layer[src_key] = []
            layer[src_key].append(tgt_key)
        network.append(layer)
    return network


def _build_inverse_network(e_src, e_tgt, l_off, num_layers, idx2vtx):
    """Build inverse network via CSR transpose."""
    inv_src, inv_tgt, inv_l_off = _build_inverse_flat(
        e_src, e_tgt, l_off, num_layers
    )
    return _flat_to_network_dict(inv_src, inv_tgt, inv_l_off, num_layers, idx2vtx)


def _compute_inv2(inv_src, inv_tgt, inv_l_off, num_layers,
                  node_in, node_out, idx2vtx, vtx2idx, V):
    """
    Compute invariant 2 (loops + dead-end branches) from flat inverse arrays.

    Returns (inv_result, result) in the SAME format as graph.py,
    so it can be fed directly into inv3.
    """
    # Build inv_net as list[dict] from flat arrays
    inv_net = []
    for l in range(num_layers):
        lo = int(inv_l_off[l])
        hi = int(inv_l_off[l + 1])
        layer = {}
        for e in range(lo, hi):
            src_key = idx2vtx[int(inv_src[e])]
            tgt_key = idx2vtx[int(inv_tgt[e])]
            if src_key not in layer:
                layer[src_key] = []
            layer[src_key].append(tgt_key)
        inv_net.append(layer)

    depth = len(inv_net)

    # Build layer_degree_map (same as graph.py's reverse_degree_map)
    res_list = [{} for _ in range(depth + 1)]
    for i, layer in enumerate(inv_net):
        for src, targets in layer.items():
            if src not in res_list[i]:
                res_list[i][src] = [int(node_out[vtx2idx[src]]),
                                     int(node_in[vtx2idx[src]])]
            for t in targets:
                if t not in res_list[i + 1]:
                    res_list[i + 1][t] = [int(node_out[vtx2idx[t]]),
                                           int(node_in[vtx2idx[t]])]

    layer_degree_map = []
    for layer in reversed(res_list):
        layer_degree_map.append(layer)

    # Now run the same logic as graph.py's find_loops_and_dead_end_branches
    result, inv_result = [], []

    for n in range(depth):
        child_group = []
        parent_groups = []
        is_branch = []
        used_in_structures = set()

        # 1. Detect loops and multiple paths
        for k, nbs in inv_net[n].items():
            if len(nbs) >= 2:
                parent_groups.append(tuple(nbs))
                child_group.append(tuple([k] * len(nbs)))
                is_branch.append(False)
                used_in_structures.add(k)
                used_in_structures.update(nbs)

        # 2. Detect reciprocal edges
        layer_nodes = list(inv_net[n].keys())
        for i in range(len(layer_nodes)):
            for j in range(i + 1, len(layer_nodes)):
                u, v = layer_nodes[i], layer_nodes[j]
                if v in inv_net[n].get(u, []) and u in inv_net[n].get(v, []):
                    parent_groups.append((v, u))
                    child_group.append((u, v))
                    is_branch.append(False)
                    used_in_structures.update((u, v))

        # 3. Detect dead-end branches
        prev_inv_targets = set()
        if n > 0:
            for targets in inv_net[n - 1].values():
                prev_inv_targets.update(targets)

        for node, nbs in inv_net[n].items():
            if node not in used_in_structures and node not in prev_inv_targets:
                if len(nbs) == 1:
                    parent_groups.append(tuple(nbs))
                    child_group.append((node,))
                    is_branch.append(True)
                    used_in_structures.add(node)

        # 4. Tracing paths layer-by-layer (BFS)
        for m in range(len(parent_groups)):
            current_groups = [[v] for v in parent_groups[m]]
            inv_group_history = []
            group_history = []

            h0_inv = [layer_degree_map[n][v] for v in child_group[m]]
            h0_raw = list(child_group[m])

            for i in range(len(current_groups)):
                c_node = h0_raw[i] if i < len(h0_raw) else h0_raw
                c_inv = h0_inv[i] if i < len(h0_inv) else h0_inv

                inv_group_history.append([[c_inv], sorted([layer_degree_map[n + 1][v] for v in current_groups[i]])])
                group_history.append([[c_node], current_groups[i]])

            intersections_log = []
            branch_mapping = list(range(len(current_groups)))

            for j in range(n + 1, depth):
                new_groups = []
                for b_idx in range(len(current_groups)):
                    next_nodes = set()
                    for node in current_groups[b_idx]:
                        next_nodes.update(inv_net[j].get(node, []))

                    next_list = sorted(list(next_nodes))
                    new_groups.append(next_list)

                    if next_list:
                        inv_group_history[b_idx].append(sorted([layer_degree_map[j + 1][v] for v in next_list]))
                        group_history[b_idx].append(next_list)

                for i in range(len(current_groups)):
                    for k in range(i + 1, len(current_groups)):
                        if branch_mapping[i] != branch_mapping[k]:
                            if not set(new_groups[i]).isdisjoint(new_groups[k]) and new_groups[i]:
                                old_id = branch_mapping[k]
                                for idx in range(len(current_groups)):
                                    if branch_mapping[idx] == old_id:
                                        branch_mapping[idx] = branch_mapping[i]
                                intersections_log.append(j + 1)

                current_groups = new_groups

            if is_branch[m]:
                exit_layer = depth
                for step_idx in range(1, len(group_history)):
                    nodes_at_step = group_history[step_idx]
                    if any(layer_degree_map[n + step_idx][nd] != 1 for nd in nodes_at_step if nd in layer_degree_map[n + step_idx]):
                        exit_layer = n + step_idx + (1 if n == 0 else 0)
                        break
                label = (n, exit_layer, -1)
            else:
                label_list = [n] + sorted(intersections_log)
                if len(set(child_group[m])) == 2:
                    label_list.append(0)
                label = tuple(label_list)

            paired = list(zip(inv_group_history, group_history))
            paired.sort(key=lambda x: x[0])
            result.append((label, *[p[1] for p in paired]))
            inv_result.append((label, *[p[0] for p in paired]))

    # Assign topological indices
    if result:
        for i in range(len(result)):
            result[i] = (inv_result[i], result[i])

        result.sort(key=lambda x: x[0])

        current_idx = 0
        last_key = result[0][0]

        for i in range(len(result)):
            inv_item, res_item = result[i]

            if inv_item != last_key:
                current_idx += 1
                last_key = inv_item

            inv_result[i] = inv_item
            result[i] = (current_idx, *res_item)

        # Assign global branch indices
        branch_key_to_idx = {}
        next_branch_idx = 0

        for i in range(len(result)):
            inv_item = inv_result[i]
            inv_branches = inv_item[1:]

            res_item = result[i]
            topo_idx_val = res_item[0]
            label = res_item[1]
            branches = list(res_item[2:])

            new_branches = []
            for inv_branch, branch in zip(inv_branches, branches):
                key = tuple(
                    tuple(tuple(d) if isinstance(d, list) else (d,) for d in step)
                    if isinstance(step, list) else (step,)
                    for step in inv_branch
                )

                if key not in branch_key_to_idx:
                    branch_key_to_idx[key] = next_branch_idx
                    next_branch_idx += 1

                b_global_idx = branch_key_to_idx[key]
                new_branches.append((b_global_idx, branch))

            result[i] = (topo_idx_val, label, *new_branches)

    return inv_result, result


# ═══════════════════════════════════════════════════════════════════
# 7. INV3 — VERTEX-INDEXED INTERSECTION FINGERPRINT (Numba pipeline)
# ═══════════════════════════════════════════════════════════════════
#
# Three-stage pipeline:
#   7A. _inv2_result_to_inv3_flat  — convert inv2 nested tuples → flat arrays
#   7B. _inv3_intersect_kernel     — Numba: vertex-indexed intersection → flat records
#   7C. _inv3_build_fingerprint     — sort records → canonical int64 fingerprint

# ── 7A: Convert inv2 result to flat arrays ──────────────────────────

def _inv2_result_to_inv3_flat(result):
    """
    Convert inv2 nested-tuple result into flat numpy arrays for the
    Numba inv3 intersection kernel.

    Handles any hashable vertex ID type (int, str, etc.) by mapping
    them to sequential integer indices via vtx_map.

    Returns dict with keys:
      branch_bitsets    (NB, max_al) uint64  — vertex bitset per branch per abs_layer
      branch_elem       (NB,)       int64    — element index per branch
      branch_bglobal    (NB,)       int64    — global branch index
      branch_nsteps     (NB,)       int64    — number of steps per branch
      elem_topo_idx     (NE,)       int64
      elem_start_layer  (NE,)       int64
      elem_is_dead_end  (NE,)       int64
      elem_nbranches    (NE,)       int64
      vw_flat           (NE, max_al, V) int64 — vertex weight array
      NE, NB, max_al, V, ref_layer
    """
    NE = len(result)

    # ── Pass 1: count branches, collect all vertex IDs ──
    NB = 0
    ref_layer = -1
    all_vertices = set()

    for item in result:
        meta = item[1]
        start = meta[0]
        branches_raw = item[2:]
        NB += len(branches_raw)
        for branch_entry in branches_raw:
            _, branch_data = branch_entry
            for k, layer_content in enumerate(branch_data):
                al = start + k
                if al > ref_layer:
                    ref_layer = al
                if isinstance(layer_content, (list, tuple, set)):
                    for v in layer_content:
                        all_vertices.add(v)
                else:
                    all_vertices.add(layer_content)

    if NB == 0:
        return None

    # Build vertex → index mapping (works for any hashable vertex type)
    vtx_list = sorted(all_vertices, key=str)
    vtx_map = {v: i for i, v in enumerate(vtx_list)}
    V = len(vtx_list)
    max_al = ref_layer + 1

    # ── Allocate ──
    branch_bitsets = np.zeros((NB, max_al), dtype=np.uint64)
    branch_elem = np.empty(NB, dtype=np.int64)
    branch_bglobal = np.empty(NB, dtype=np.int64)
    branch_nsteps = np.empty(NB, dtype=np.int64)
    elem_topo_idx = np.empty(NE, dtype=np.int64)
    elem_start_layer = np.empty(NE, dtype=np.int64)
    elem_is_dead_end = np.empty(NE, dtype=np.int64)
    elem_nbranches = np.empty(NE, dtype=np.int64)
    vw_flat = np.zeros((NE, max_al, V), dtype=np.int64)

    # ── Pass 2: fill arrays ──
    b_idx = 0
    for i, item in enumerate(result):
        topo_idx = item[0]
        meta = item[1]
        start = meta[0]
        is_dead = 1 if (len(meta) == 3 and meta[-1] == -1) else 0
        branches_raw = item[2:]

        elem_topo_idx[i] = topo_idx
        elem_start_layer[i] = start
        elem_is_dead_end[i] = is_dead
        elem_nbranches[i] = len(branches_raw)

        for b_local, branch_entry in enumerate(branches_raw):
            b_global, branch_data = branch_entry

            branch_elem[b_idx] = i
            branch_bglobal[b_idx] = b_global
            branch_nsteps[b_idx] = len(branch_data)

            for k, layer_content in enumerate(branch_data):
                al = start + k
                if isinstance(layer_content, (list, tuple, set)):
                    vset = layer_content
                else:
                    vset = [layer_content]

                bs = np.uint64(0)
                for v in vset:
                    vi = vtx_map[v]
                    bs |= np.uint64(1) << np.uint64(vi)
                    vw_flat[i, al, vi] += 1

                branch_bitsets[b_idx, al] = bs

            b_idx += 1

    return {
        'branch_bitsets': branch_bitsets,
        'branch_elem': branch_elem,
        'branch_bglobal': branch_bglobal,
        'branch_nsteps': branch_nsteps,
        'elem_topo_idx': elem_topo_idx,
        'elem_start_layer': elem_start_layer,
        'elem_is_dead_end': elem_is_dead_end,
        'elem_nbranches': elem_nbranches,
        'vw_flat': vw_flat,
        'NE': NE, 'NB': NB, 'max_al': max_al, 'V': V, 'ref_layer': ref_layer,
    }


# ── 7B: Numba intersection kernel ──────────────────────────────────
#
# Record format: each intersection produces TWO records (i→j and j→i):
#   [elem, step, other_elem, our_bid, other_bid, n_wt, wt0..wt7]
#   Total: 6 + MAX_WT = 14 int64s per record

_MAX_WT = 8
_REC_SIZE = 6 + _MAX_WT   # 14


@njit(cache=True)
def _inv3_intersect_kernel(
    branch_bitsets,       # (NB, max_al) uint64
    branch_elem,          # (NB,) int64
    branch_bglobal,       # (NB,) int64
    branch_nsteps,        # (NB,) int64
    elem_start_layer_arr, # (NE,) int64
    vw_flat,              # (NE, max_al, V) int64
    NE, NB, max_al, V, ref_layer,
):
    """
    Numba kernel: vertex-indexed intersection computation for inv3.

    For each abs_layer (except ref_layer):
      1. Build vertex → participating branches index
      2. For each vertex, check cross-element branch pairs
      3. Compute intersection via bitset AND, extract weighted pairs
      4. Emit flat records (one per direction per pair)

    Returns (rec_buf, rec_count) where rec_buf has rec_count * REC_SIZE int64s.
    """
    max_recs = NB * NB  # generous upper bound
    rec_buf = np.empty(max_recs * _REC_SIZE, dtype=np.int64)
    rec_count = np.int64(0)

    # Allocate per-layer arrays once (reused across layers)
    vp_count = np.zeros(V, dtype=np.int64)
    vp_off = np.zeros(V + 1, dtype=np.int64)

    for al in range(max_al):
        if al == ref_layer:
            continue

        # ── Build vertex → participants index ──

        # Pass 1: count participants per vertex
        vp_count[:] = 0
        for bi in range(NB):
            ns = branch_nsteps[bi]
            if ns == 0:
                continue
            step = al - elem_start_layer_arr[branch_elem[bi]]
            if step < 0 or step >= ns:
                continue
            bs = branch_bitsets[bi, al]
            if bs == 0:
                continue
            tmp = bs
            while tmp != 0:
                low = tmp & (~tmp + np.uint64(1))  # extract lowest set bit
                v = 0
                t = low
                while t > np.uint64(1):
                    t >>= np.uint64(1)
                    v += 1
                vp_count[v] += 1
                tmp ^= low

        # Prefix sum → offsets
        vp_off[0] = 0
        for v in range(V):
            vp_off[v + 1] = vp_off[v] + vp_count[v]

        total_parts = int(vp_off[V])
        if total_parts < 2:
            continue

        # Pass 2: fill participant buffer
        vp_buf = np.empty(total_parts, dtype=np.int64)
        vp_fill = np.zeros(V, dtype=np.int64)

        for bi in range(NB):
            ns = branch_nsteps[bi]
            if ns == 0:
                continue
            step = al - elem_start_layer_arr[branch_elem[bi]]
            if step < 0 or step >= ns:
                continue
            bs = branch_bitsets[bi, al]
            if bs == 0:
                continue
            tmp = bs
            while tmp != 0:
                low = tmp & (~tmp + np.uint64(1))  # extract lowest set bit
                v = 0
                t = low
                while t > np.uint64(1):
                    t >>= np.uint64(1)
                    v += 1
                pos = int(vp_off[v]) + int(vp_fill[v])
                vp_buf[pos] = bi
                vp_fill[v] += 1
                tmp ^= low

        # ── Process cross-element pairs ──

        # Dedup: pair_seen[bi * NB + bj] for bi < bj
        pair_seen = np.zeros(NB * NB, dtype=np.bool_)

        for v in range(V):
            start = int(vp_off[v])
            end = int(vp_off[v + 1])
            n_parts = end - start
            if n_parts < 2:
                continue

            for a in range(n_parts):
                bi = int(vp_buf[start + a])
                ei = int(branch_elem[bi])
                for b in range(a + 1, n_parts):
                    bj = int(vp_buf[start + b])
                    ej = int(branch_elem[bj])

                    if ei == ej:
                        continue

                    # Canonical pair key (bi < bj)
                    if bi < bj:
                        pk = bi * NB + bj
                    else:
                        pk = bj * NB + bi

                    if pair_seen[pk]:
                        continue
                    pair_seen[pk] = True

                    # Ensure ei < ej for consistent direction
                    if ei > ej:
                        bi, bj = bj, bi
                        ei, ej = ej, ei

                    # Bitset intersection
                    common = branch_bitsets[bi, al] & branch_bitsets[bj, al]
                    if common == np.uint64(0):
                        continue

                    # Extract common vertices, build weight pairs
                    wt_ij = np.zeros(_MAX_WT, dtype=np.int64)
                    wt_ji = np.zeros(_MAX_WT, dtype=np.int64)
                    n_common = 0
                    tmp = common
                    while tmp != np.uint64(0) and n_common < _MAX_WT:
                        low = tmp & (~tmp + np.uint64(1))  # extract lowest set bit
                        v_c = 0
                        t = low
                        while t > np.uint64(1):
                            t >>= np.uint64(1)
                            v_c += 1
                        w_i = int(vw_flat[ei, al, v_c])
                        w_j = int(vw_flat[ej, al, v_c])
                        wt_ij[n_common] = (w_i << 16) | w_j
                        wt_ji[n_common] = (w_j << 16) | w_i
                        n_common += 1
                        tmp ^= low

                    # Insertion sort (n_common ≤ 8)
                    for p in range(1, n_common):
                        key_ij = wt_ij[p]
                        q = p - 1
                        while q >= 0 and wt_ij[q] > key_ij:
                            wt_ij[q + 1] = wt_ij[q]
                            q -= 1
                        wt_ij[q + 1] = key_ij

                    for p in range(1, n_common):
                        key_ji = wt_ji[p]
                        q = p - 1
                        while q >= 0 and wt_ji[q] > key_ji:
                            wt_ji[q + 1] = wt_ji[q]
                            q -= 1
                        wt_ji[q + 1] = key_ji

                    step_i = al - int(elem_start_layer_arr[ei])
                    step_j = al - int(elem_start_layer_arr[ej])

                    # ── Emit record for ei ──
                    # Positions 3,4 store flat branch indices (bi, bj) for unique
                    # identification; global branch IDs are looked up in Python.
                    base = rec_count * _REC_SIZE
                    rec_buf[base + 0] = ei
                    rec_buf[base + 1] = step_i
                    rec_buf[base + 2] = ej
                    rec_buf[base + 3] = bi      # flat branch index of our branch
                    rec_buf[base + 4] = bj      # flat branch index of other branch
                    rec_buf[base + 5] = n_common
                    for p in range(n_common):
                        rec_buf[base + 6 + p] = wt_ij[p]
                    for p in range(n_common, _MAX_WT):
                        rec_buf[base + 6 + p] = 0
                    rec_count += 1

                    # ── Emit record for ej ──
                    base = rec_count * _REC_SIZE
                    rec_buf[base + 0] = ej
                    rec_buf[base + 1] = step_j
                    rec_buf[base + 2] = ei
                    rec_buf[base + 3] = bj      # flat branch index of our branch
                    rec_buf[base + 4] = bi      # flat branch index of other branch
                    rec_buf[base + 5] = n_common
                    for p in range(n_common):
                        rec_buf[base + 6 + p] = wt_ji[p]
                    for p in range(n_common, _MAX_WT):
                        rec_buf[base + 6 + p] = 0
                    rec_count += 1

    return rec_buf[:rec_count * _REC_SIZE], rec_count


# ── 7B2: Numba Phase 4 kernel — grouping + encoding ────────────────
#
# Takes sorted records and produces per-element encoded fingerprint data.
# Eliminates all Python dict/tuple overhead.

@njit(cache=True)
def _inv3_encode_fingerprint_kernel(
    records,           # (N, REC_SIZE) int64 — sorted by (elem, step, other_elem, our_bid, other_bid)
    rec_count,
    elem_topo_idx,     # (NE,) int64
    elem_is_dead_end,  # (NE,) int64
    max_steps_arr,     # (NE,) int64 — max steps per element
    NE,
):
    """
    Numba Phase 4: group sorted records and encode fingerprint per element.

    For each element, for each step:
      - Group records by other_elem → topo_idx
      - Group by our_bid within each topo_idx group
      - Encode into flat int64 format

    Output:
      elem_fp_buf   — flat int64 buffer with per-element fingerprint data
      elem_fp_off   — (NE+1,) int64 offsets into elem_fp_buf
      elem_fp_sort  — (NE, 16) int64 sort key per element (first 16 ints of fingerprint)
    """
    # Pre-allocate generously
    max_fp_total = rec_count * 20 + NE * 20
    elem_fp_buf = np.empty(max_fp_total, dtype=np.int64)
    elem_fp_off = np.zeros(NE + 1, dtype=np.int64)
    # Sort key: first 16 ints of each element's fingerprint (padded with 0)
    elem_fp_sort = np.zeros((NE, 16), dtype=np.int64)

    fp_pos = np.int64(0)
    rec_idx = np.int64(0)

    for i in range(NE):
        if elem_is_dead_end[i] == 1:
            # Dead-end: empty fingerprint
            elem_fp_off[i + 1] = fp_pos
            continue

        ms = int(max_steps_arr[i])
        elem_start = fp_pos
        sort_key_pos = 0

        for k in range(ms):
            # Find records for (elem=i, step=k)
            step_start = rec_idx
            while rec_idx < rec_count and records[rec_idx, 0] == i and records[rec_idx, 1] == k:
                rec_idx += 1
            step_end = rec_idx

            if step_start == step_end:
                # No intersections at this layer
                elem_fp_buf[fp_pos] = -1
                fp_pos += 1
                if sort_key_pos < 16:
                    # -1 in sort key represents empty layer
                    pass  # leave as 0 (empty tuple sorts before positive)
                continue

            # Count distinct topo_idx groups
            # First pass: count groups
            n_groups = 0
            pos = step_start
            prev_tidx = -1
            while pos < step_end:
                other_e = int(records[pos, 2])
                tidx = int(elem_topo_idx[other_e])
                if tidx != prev_tidx:
                    n_groups += 1
                    prev_tidx = tidx
                # Advance past this other_e
                while pos < step_end and records[pos, 2] == other_e:
                    pos += 1

            # Write n_groups
            elem_fp_buf[fp_pos] = n_groups
            if sort_key_pos < 16:
                elem_fp_sort[i, sort_key_pos] = n_groups
                sort_key_pos += 1
            fp_pos += 1

            # Second pass: process each topo_idx group
            pos = step_start
            while pos < step_end:
                other_e = int(records[pos, 2])
                tidx = int(elem_topo_idx[other_e])

                # Write tidx
                elem_fp_buf[fp_pos] = tidx
                if sort_key_pos < 16:
                    elem_fp_sort[i, sort_key_pos] = tidx
                    sort_key_pos += 1
                fp_pos += 1

                # Gather all records for this other_e
                oe_start = pos
                while pos < step_end and records[pos, 2] == other_e:
                    pos += 1
                oe_end = pos

                # Count distinct our_bid groups
                n_details = 0
                dpos = oe_start
                prev_ob = -1
                while dpos < oe_end:
                    ob = int(records[dpos, 3])
                    if ob != prev_ob:
                        n_details += 1
                        prev_ob = ob
                    while dpos < oe_end and records[dpos, 3] == ob:
                        dpos += 1

                # Write n_details
                elem_fp_buf[fp_pos] = n_details
                if sort_key_pos < 16:
                    elem_fp_sort[i, sort_key_pos] = n_details
                    sort_key_pos += 1
                fp_pos += 1

                # Process each our_bid group
                dpos = oe_start
                while dpos < oe_end:
                    ob = int(records[dpos, 3])

                    # Count records for this our_bid
                    ob_start = dpos
                    while dpos < oe_end and records[dpos, 3] == ob:
                        dpos += 1
                    ob_end = dpos
                    sub_len = ob_end - ob_start

                    # Write sub_len
                    elem_fp_buf[fp_pos] = sub_len
                    fp_pos += 1

                    # Write our_bid
                    elem_fp_buf[fp_pos] = ob
                    fp_pos += 1

                    # Write each (other_bid, n_wt, wt0..wtN-1) entry
                    for r in range(ob_start, ob_end):
                        other_bid_val = int(records[r, 4])
                        n_wt = int(records[r, 5])
                        elem_fp_buf[fp_pos] = other_bid_val
                        fp_pos += 1
                        elem_fp_buf[fp_pos] = n_wt
                        fp_pos += 1
                        for p in range(n_wt):
                            enc = int(records[r, 6 + p])
                            elem_fp_buf[fp_pos] = enc >> 16
                            fp_pos += 1
                            elem_fp_buf[fp_pos] = enc & 0xFFFF
                            fp_pos += 1

        elem_fp_off[i + 1] = fp_pos

    return elem_fp_buf[:fp_pos], elem_fp_off[:NE + 1], elem_fp_sort


# ── 7C: Build fingerprint from records ──────────────────────────────

def _inv3_build_fingerprint_from_records(rec_buf, rec_count, flat_data):
    """
    Build canonical int64 fingerprint from flat intersection records.

    Phase 4 replacement: instead of dict-of-dicts, we sort the flat
    records and do a single linear scan to build the fingerprint.

    Record format: [elem, step, other_elem, our_bid, other_bid, n_wt, wt0..wt7]
    """
    NE = flat_data['NE']
    elem_topo_idx = flat_data['elem_topo_idx']
    elem_start_layer = flat_data['elem_start_layer']
    elem_is_dead_end = flat_data['elem_is_dead_end']
    elem_nbranches = flat_data['elem_nbranches']

    if rec_count == 0:
        # All elements are either dead-end or have no intersections
        fp_parts = []
        for i in range(NE):
            if elem_is_dead_end[i]:
                fp_parts.append(0)
            else:
                # Non-dead-end with no intersections: all steps are -1 (None)
                branches_raw_count = int(elem_nbranches[i])
                if branches_raw_count == 0:
                    fp_parts.append(0)
                else:
                    # We don't know max_steps here without the original result.
                    # This is a corner case — return a minimal fingerprint.
                    fp_parts.append(-1)
        return np.array(fp_parts, dtype=np.int64)

    # Reshape records to 2D
    records = rec_buf.reshape(rec_count, _REC_SIZE)

    # Sort by (elem, step, other_elem, our_bid, other_bid) for canonical order
    # Use numpy lexsort (sorts by last key first)
    sort_keys = (
        records[:, 4],   # other_bid
        records[:, 3],   # our_bid
        records[:, 2],   # other_elem
        records[:, 1],   # step
        records[:, 0],   # elem
    )
    order = np.lexsort(sort_keys)
    records = records[order]

    # ── Build fingerprint by iterating sorted records ──
    fp_parts = []

    # We need max_steps per element. Reconstruct from flat_data.
    # Since we don't store branch_data lengths, we compute max step from records.
    max_step_per_elem = {}
    for r in range(rec_count):
        ei = int(records[r, 0])
        step = int(records[r, 1])
        if ei not in max_step_per_elem:
            max_step_per_elem[ei] = step
        elif step > max_step_per_elem[ei]:
            max_step_per_elem[ei] = step

    rec_idx = 0  # current position in sorted records

    for i in range(NE):
        if elem_is_dead_end[i]:
            fp_parts.append(0)
            continue

        # Determine max steps for this element
        # We need to know the number of steps (layers) in the element's branches.
        # This info was in the original result but not in flat_data.
        # For now, we use the max step from records (elements without records
        # get a single -1 marker).
        max_step = max_step_per_elem.get(i, -1)

        for k in range(max_step + 1):
            # Collect all records for (elem=i, step=k)
            step_records = []
            while rec_idx < rec_count and records[rec_idx, 0] == i and records[rec_idx, 1] == k:
                step_records.append(records[rec_idx])
                rec_idx += 1

            if not step_records:
                fp_parts.append(-1)  # None layer marker
                continue

            # Group by other_elem
            by_other = {}
            for rec in step_records:
                other_e = int(rec[2])
                our_bid = int(rec[3])
                other_bid = int(rec[4])
                n_wt = int(rec[5])
                wt_tuple = tuple(int(rec[6 + p]) for p in range(n_wt))
                if other_e not in by_other:
                    by_other[other_e] = []
                by_other[other_e].append((our_bid, other_bid, wt_tuple))

            # Group by topo_idx of other_elem
            by_topo = {}
            for other_e, entries in by_other.items():
                tidx = int(elem_topo_idx[other_e])
                if tidx not in by_topo:
                    by_topo[tidx] = []

                # Group entries by our_bid
                by_our_bid = {}
                for our_bid, other_bid, wt in entries:
                    if our_bid not in by_our_bid:
                        by_our_bid[our_bid] = []
                    by_our_bid[our_bid].append((other_bid, wt))

                sub_tuple = []
                for our_bid in sorted(by_our_bid.keys()):
                    id_pairs = sorted(by_our_bid[our_bid])
                    sub_tuple.append((our_bid, tuple(id_pairs)))
                sub_tuple.sort()
                by_topo[tidx].append(tuple(sub_tuple))

            sorted_groups = sorted(by_topo.items())
            fp_parts.append(len(sorted_groups))
            for tidx, detail_tuples in sorted_groups:
                fp_parts.append(tidx)
                sorted_details = sorted(detail_tuples)
                fp_parts.append(len(sorted_details))
                for sub_tuple in sorted_details:
                    fp_parts.append(len(sub_tuple))
                    for our_bid, id_pairs in sub_tuple:
                        fp_parts.append(our_bid)
                        fp_parts.append(len(id_pairs))
                        for other_bid, wt_encoded in id_pairs:
                            fp_parts.append(other_bid)
                            fp_parts.append(len(wt_encoded))
                            for enc in wt_encoded:
                                fp_parts.append(enc >> 16)       # w_i
                                fp_parts.append(enc & 0xFFFF)    # w_j

    return np.array(fp_parts, dtype=np.int64)


# ── 7D: Public inv3 function (v3 — Numba intersection + Python encoding) ──

def compute_inv3_fingerprint_v3(result):
    """
    Compute inv3 fingerprint using Numba vertex-indexed intersection
    followed by correct Python encoding that groups by topo_idx.

    Pipeline:
      1. Convert inv2 result to flat arrays (_inv2_result_to_inv3_flat)
      2. Run Numba intersection kernel (_inv3_intersect_kernel) — fast
      3. Python encoding: group by (elem, step, topo_idx), aggregate, sort, encode

    The fingerprint is CANONICAL: elements are sorted so that two
    isomorphic vertex pairs always produce the same fingerprint.

    Returns np.ndarray(int64) — canonical isomorphism-invariant fingerprint.
    """
    NE = len(result)
    if NE == 0:
        return np.array([], dtype=np.int64)

    # ── Phase 1: Convert inv2 result to flat arrays ──
    flat_data = _inv2_result_to_inv3_flat(result)
    if flat_data is None:
        return np.array([], dtype=np.int64)

    # Determine max_steps per element (number of layers per element)
    max_steps_per_elem = []
    for item in result:
        branches_raw = item[2:]
        ms = max((len(bd) for _, bd in branches_raw), default=0)
        max_steps_per_elem.append(ms)

    elem_topo_idx = flat_data['elem_topo_idx']
    elem_is_dead_end = flat_data['elem_is_dead_end']

    # ── Phase 2: Numba intersection kernel ──
    rec_buf, rec_count = _inv3_intersect_kernel(
        flat_data['branch_bitsets'],
        flat_data['branch_elem'],
        flat_data['branch_bglobal'],
        flat_data['branch_nsteps'],
        flat_data['elem_start_layer'],
        flat_data['vw_flat'],
        flat_data['NE'],
        flat_data['NB'],
        flat_data['max_al'],
        flat_data['V'],
        flat_data['ref_layer'],
    )

    # ── Phase 3: Python encoding (correct grouping by topo_idx) ──

    # The Numba kernel outputs flat branch indices (b_idx) in record
    # positions 3 and 4, NOT global branch IDs.  This lets us uniquely
    # identify which branch each record belongs to, even when multiple
    # branches share the same global branch ID.

    branch_bglobal = flat_data['branch_bglobal']  # flat b_idx → global bid

    # Build per-(elem, step) records from the Numba kernel output
    # Each record: (other_elem, b_idx_our, b_idx_other, wt_encoded)
    elem_step_records = defaultdict(list)
    if rec_count > 0:
        records = rec_buf.reshape(rec_count, _REC_SIZE)
        for r in range(rec_count):
            elem = int(records[r, 0])
            step = int(records[r, 1])
            other_elem = int(records[r, 2])
            b_idx_our = int(records[r, 3])   # flat branch index of our branch
            b_idx_other = int(records[r, 4])  # flat branch index of other branch
            n_wt = int(records[r, 5])
            wt_encoded = tuple(int(records[r, 6 + p]) for p in range(n_wt))
            elem_step_records[(elem, step)].append(
                (other_elem, b_idx_our, b_idx_other, wt_encoded)
            )

    # Build per-element fingerprint and sort key
    elem_fingerprints = []  # list of (sort_key, flat_fp_parts) per element

    for i in range(NE):
        if elem_is_dead_end[i]:
            # Dead-end element: marker 0, sort key ()
            elem_fingerprints.append(((), [0]))
            continue

        ms = max_steps_per_elem[i]
        fp_parts = []
        sort_key_parts = []

        for k in range(ms):
            step_records = elem_step_records.get((i, k), [])

            if not step_records:
                # No intersections at this layer → None → -1 in flat encoding
                fp_parts.append(-1)
                sort_key_parts.append(())
                continue

            # Group by other_elem first (to reconstruct per-other_elem details)
            by_other = defaultdict(list)
            for other_elem, b_idx_our, b_idx_other, wt_encoded in step_records:
                by_other[other_elem].append((b_idx_our, b_idx_other, wt_encoded))

            # Then group by topo_idx of other_elem, aggregating all
            # details from different other_elems that share the same topo_idx
            idx_branch_details = defaultdict(list)
            for other_elem, entries in by_other.items():
                tidx = int(elem_topo_idx[other_elem])

                # Group entries by b_idx_our (flat branch index) to build
                # per-branch detail tuples.  This correctly distinguishes
                # different branches even when they share a global branch ID.
                by_bidx = defaultdict(list)
                for b_idx_our, b_idx_other, wt_encoded in entries:
                    our_bid = int(branch_bglobal[b_idx_our])
                    other_bid = int(branch_bglobal[b_idx_other])
                    # Decode wt_encoded: each enc = (w_i << 16) | w_j
                    wt_decoded = tuple((enc >> 16, enc & 0xFFFF) for enc in wt_encoded)
                    by_bidx[b_idx_our].append((our_bid, other_bid, wt_decoded))

                # Build sorted details for this other_elem — one entry per
                # branch of element i that intersects with this other_elem.
                details = []
                for b_idx_our in sorted(by_bidx.keys()):
                    entries_for_branch = by_bidx[b_idx_our]
                    our_bid = entries_for_branch[0][0]  # all same for this b_idx
                    j_ids = tuple(sorted(
                        (other_bid, wt) for _, other_bid, wt in entries_for_branch
                    ))
                    details.append((our_bid, j_ids))
                details.sort()

                idx_branch_details[tidx].append(tuple(details))

            # Sort and encode this layer
            sorted_groups = sorted(idx_branch_details.items())

            # Build sort key for this layer (matches original graph.py)
            layer_sort_key = tuple(
                (tidx, tuple(sorted(detail_tuples)))
                for tidx, detail_tuples in sorted_groups
            )
            sort_key_parts.append(layer_sort_key)

            # Build flat encoding for this layer
            fp_parts.append(len(sorted_groups))
            for tidx, detail_tuples in sorted_groups:
                fp_parts.append(tidx)
                sorted_details = sorted(detail_tuples)
                fp_parts.append(len(sorted_details))
                for detail in sorted_details:
                    fp_parts.append(len(detail))
                    for our_bid, j_ids in detail:
                        fp_parts.append(our_bid)
                        fp_parts.append(len(j_ids))
                        for other_bid, wt in j_ids:
                            fp_parts.append(other_bid)
                            fp_parts.append(len(wt))
                            for w_i, w_j in wt:
                                fp_parts.append(w_i)
                                fp_parts.append(w_j)

        elem_fingerprints.append((tuple(sort_key_parts), fp_parts))

    # Sort elements canonically (same as original graph.py's _sort_key)
    elem_fingerprints.sort(key=lambda x: x[0])

    # Concatenate sorted fingerprints
    all_parts = []
    for _, fp_parts in elem_fingerprints:
        all_parts.extend(fp_parts)

    return np.array(all_parts, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════
# 8. BACKWARD-COMPATIBLE WRAPPERS
# ═══════════════════════════════════════════════════════════════════

def build_network_and_bdp(graph, start_vertex, _csr_cache=None, _edge_seen=None):
    """Build network + BDP fingerprint. Returns a named namespace."""
    if _csr_cache is not None:
        row_ptr, col_idx, vtx2idx, idx2vtx, V = _csr_cache
    else:
        row_ptr, col_idx, vtx2idx, idx2vtx, V = adj_dict_to_csr(graph)

    if _edge_seen is None:
        _edge_seen = EdgeSeenBuffer(V)
    elif _edge_seen.V < V:
        _edge_seen = EdgeSeenBuffer(V)
    _edge_seen.reset()

    start_idx = vtx2idx[start_vertex]
    e_src, e_tgt, l_off, n_layers, n_edges, node_in, node_out = \
        _build_network_kernel(row_ptr, col_idx, start_idx, V,
                               _edge_seen.data, _edge_seen.version)

    e_src = e_src[:n_edges]
    e_tgt = e_tgt[:n_edges]
    l_off = l_off[:n_layers + 1]

    inv1_fwd_fp, inv1_rev_fp = build_inv1_fingerprint(
        e_src, e_tgt, l_off, n_layers, node_in, node_out, V
    )

    network_dict = _flat_to_network_dict(e_src, e_tgt, l_off, n_layers, idx2vtx)
    _, _, rdm = _GraphPy.compute_bidirectional_degree_profiles(network_dict)

    class _Result:
        pass
    r = _Result()
    r.network = network_dict
    r.rdm = rdm
    r.bdp = (inv1_fwd_fp, inv1_rev_fp)
    r.inv1_fwd_fp = inv1_fwd_fp
    r.inv1_rev_fp = inv1_rev_fp
    r.fwd_fp = inv1_fwd_fp      # alias for iso_tester_optimized.py
    r.rev_fp = inv1_rev_fp      # alias for iso_tester_optimized.py
    r._flat = (e_src, e_tgt, l_off, n_layers, idx2vtx)
    r._csr = (row_ptr, col_idx, vtx2idx, idx2vtx, V)
    r._node_in = node_in
    r._node_out = node_out
    r._edge_seen = _edge_seen
    return r


def find_loops_and_dead_end_branches(network, layer_degree_map, _flat=None):
    """Compute inv2. Delegates to graph.py's implementation."""
    return _GraphPy.find_loops_and_dead_end_branches(network, layer_degree_map)


def get_loops_and_dead_end_branches_intersections(ld2_result):
    """Compute inv3 canonical fingerprint as flat int64 array.

    Uses the original graph.py intersection algorithm for correctness,
    then encodes the result into a flat int64 array for O(1) comparison
    via np.array_equal.  This is faster than comparing nested Python
    tuples and avoids the bugs in the Numba intersection kernel.
    """
    inv3_orig = _GraphPy.get_loops_and_dead_end_branches_intersections(ld2_result)
    return _inv3_orig_to_flat(inv3_orig)


def _inv3_orig_to_flat(inv3_orig):
    """Convert original graph.py Inv3 output to flat int64 array.

    Encoding scheme:
      Element markers:
        0 = dead-end element
      Per loop element, per step (layer):
        -1 = None layer (no intersections)
        Otherwise:
          n_topo_groups
          for each topo_group:
            topo_idx
            n_detail_tuples
            for each detail_tuple:
              n_our_bid_entries
              for each (our_bid, j_ids) pair:
                our_bid
                n_j_ids
                for each (other_bid, wt) in j_ids:
                  other_bid
                  n_wt_pairs
                  for each (w_i, w_j) in wt:
                    w_i
                    w_j
    """
    parts = []
    for element in inv3_orig:
        if element == ():
            parts.append(0)
            continue
        for layer in element:
            if layer is None:
                parts.append(-1)
            else:
                parts.append(len(layer))
                for group in layer:
                    tidx = group[0]
                    detail_tuples = group[1]
                    parts.append(tidx)
                    parts.append(len(detail_tuples))
                    for detail in detail_tuples:
                        parts.append(len(detail))
                        for our_bid, j_ids in detail:
                            parts.append(our_bid)
                            parts.append(len(j_ids))
                            for other_bid, wt in j_ids:
                                parts.append(other_bid)
                                parts.append(len(wt))
                                for w_pair in wt:
                                    parts.append(w_pair[0])
                                    parts.append(w_pair[1])
    return np.array(parts, dtype=np.int64)


def get_intersections_numba(ld2_result):
    """Compute inv3 fingerprint using the Numba pipeline (v3)."""
    return compute_inv3_fingerprint_v3(ld2_result)


def get_fingerprint_numba(ld2_result):
    """Compute inv3 fingerprint using the Numba pipeline (v3). Alias."""
    return compute_inv3_fingerprint_v3(ld2_result)


def _flat_tuple(data):
    """Extract flat tuple from build_network_and_bdp result."""
    return data._flat


# ═══════════════════════════════════════════════════════════════════
# 9. PUBLIC API: FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════

def build_all_invariants(graph, start_vertex, _csr_cache=None, _edge_seen=None):
    """
    Build network + all 3 invariants from a single start vertex.

    Returns dict with:
      'inv1_fwd_fp', 'inv1_rev_fp' — BDP fingerprint arrays
      'inv2_inv_result' — inv2 invariant result (for comparison)
      'inv2_result' — inv2 result (fed into inv3)
      'inv3_fp' — inv3 fingerprint array
      'network_data' — (e_src, e_tgt, l_off, n_layers, idx2vtx, vtx2idx, V, node_in, node_out)
    """
    # Step 1: CSR conversion
    if _csr_cache is not None:
        row_ptr, col_idx, vtx2idx, idx2vtx, V = _csr_cache
    else:
        row_ptr, col_idx, vtx2idx, idx2vtx, V = adj_dict_to_csr(graph)

    if _edge_seen is None:
        _edge_seen = EdgeSeenBuffer(V)
    elif _edge_seen.V < V:
        _edge_seen = EdgeSeenBuffer(V)
    _edge_seen.reset()

    # Step 2: BFS network (Numba)
    start_idx = vtx2idx[start_vertex]
    e_src, e_tgt, l_off, n_layers, n_edges, node_in, node_out = \
        _build_network_kernel(row_ptr, col_idx, start_idx, V,
                               _edge_seen.data, _edge_seen.version)

    e_src = e_src[:n_edges]
    e_tgt = e_tgt[:n_edges]
    l_off = l_off[:n_layers + 1]

    # Step 3: Inv1 — BDP fingerprint + reverse degree map
    inv1_fwd_fp, inv1_rev_fp = build_inv1_fingerprint(
        e_src, e_tgt, l_off, n_layers, node_in, node_out, V
    )

    # Build network dict + BDP for inv2's layer_degree_map
    network_dict = _flat_to_network_dict(e_src, e_tgt, l_off, n_layers, idx2vtx)
    _, _, rdm = _GraphPy.compute_bidirectional_degree_profiles(network_dict)

    # Step 4: Inv2 — Loops + dead-end branches
    inv2_inv_result, inv2_result = _GraphPy.find_loops_and_dead_end_branches(
        network_dict, rdm)

    # Step 5: Inv3 — Intersections fingerprint (Numba pipeline)
    inv3_fp = compute_inv3_fingerprint_v3(inv2_result)

    return {
        'inv1_fwd_fp': inv1_fwd_fp,
        'inv1_rev_fp': inv1_rev_fp,
        'inv2_inv_result': inv2_inv_result,
        'inv2_result': inv2_result,
        'inv3_fp': inv3_fp,
        'network_data': (e_src, e_tgt, l_off, n_layers, idx2vtx, vtx2idx, V, node_in, node_out),
    }


# ═══════════════════════════════════════════════════════════════════
# 10. COMPATIBILITY: Graph-like class
# ═══════════════════════════════════════════════════════════════════

class GraphOpt:
    """Optimized graph isomorphism testing using flat arrays + Numba."""

    def __init__(self, graph1, graph2=None):
        self.graph1 = graph1
        self.graph2 = graph2 if graph2 is not None else self._make_graph2()

    def _make_graph2(self):
        g2 = {}
        for key, values in self.graph1.items():
            g2[str(key) + '~'] = [str(v) + '~' for v in values]
        return g2

    def test_is_isomorphic(self):
        """Test whether graph1 and graph2 are isomorphic."""
        g1, g2 = self.graph1, self.graph2

        if len(g1) != len(g2):
            return False

        e1 = sum(len(vs) for vs in g1.values())
        e2 = sum(len(vs) for vs in g2.values())
        if e1 != e2:
            return False

        deg_groups_1 = {}
        for v in g1:
            d = len(g1.get(v, []))
            deg_groups_1.setdefault(d, []).append(v)
        deg_groups_2 = {}
        for v in g2:
            d = len(g2.get(v, []))
            deg_groups_2.setdefault(d, []).append(v)

        if {d: len(vs) for d, vs in deg_groups_1.items()} != \
           {d: len(vs) for d, vs in deg_groups_2.items()}:
            return False

        # Build CSR once per graph
        csr1 = adj_dict_to_csr(g1)
        csr2 = adj_dict_to_csr(g2)
        esb1 = EdgeSeenBuffer(csr1[4])
        esb2 = EdgeSeenBuffer(csr2[4])

        ref_deg = min(deg_groups_1, key=lambda d: len(deg_groups_1[d]))
        ref_vertex = random.choice(deg_groups_1[ref_deg])
        candidates = deg_groups_2.get(ref_deg, [])

        # Compute all invariants for graph1 reference vertex
        data1 = build_all_invariants(g1, ref_vertex, _csr_cache=csr1, _edge_seen=esb1)

        for v2 in candidates:
            esb2.reset()
            data2 = build_all_invariants(g2, v2, _csr_cache=csr2, _edge_seen=esb2)

            # Compare Inv1 (BDP fingerprint)
            if not np.array_equal(data1['inv1_fwd_fp'], data2['inv1_fwd_fp']):
                continue
            if not np.array_equal(data1['inv1_rev_fp'], data2['inv1_rev_fp']):
                continue

            # Compare Inv2
            if data1['inv2_inv_result'] != data2['inv2_inv_result']:
                continue

            # Compare Inv3 (fingerprint)
            if np.array_equal(data1['inv3_fp'], data2['inv3_fp']):
                return True

        return False

    def test_find_orbits(self):
        """Find vertex orbits between graph1 and graph2."""
        g1, g2 = self.graph1, self.graph2

        if len(g1) != len(g2):
            return None

        e1 = sum(len(v1) for v1 in g1.values())
        e2 = sum(len(v2) for v2 in g2.values())
        if e1 != e2:
            return None

        degree_sequence1 = sorted(len(nb1) for nb1 in g1.values())
        degree_sequence2 = sorted(len(nb2) for nb2 in g2.values())
        if degree_sequence1 != degree_sequence2:
            return None

        vertices1 = list(g1.keys())
        vertices2 = list(g2.keys())

        # Precompute all inv2/inv3 for graph2
        csr2 = adj_dict_to_csr(g2)
        esb2 = EdgeSeenBuffer(csr2[4])

        g2_data = {}
        for v2 in vertices2:
            csr2 = adj_dict_to_csr(g2)
            esb2 = EdgeSeenBuffer(csr2[4])
            g2_data[v2] = build_all_invariants(g2, v2, _csr_cache=csr2, _edge_seen=esb2)

        csr1 = adj_dict_to_csr(g1)
        esb1 = EdgeSeenBuffer(csr1[4])

        orbits = []
        for v1 in vertices1:
            csr1 = adj_dict_to_csr(g1)
            esb1 = EdgeSeenBuffer(csr1[4])
            data1 = build_all_invariants(g1, v1, _csr_cache=csr1, _edge_seen=esb1)

            d1 = len(g1[v1])
            v1_matched = False

            for v2 in vertices2:
                if len(g2[v2]) != d1:
                    continue

                data2 = g2_data[v2]

                if not np.array_equal(data1['inv1_fwd_fp'], data2['inv1_fwd_fp']):
                    continue
                if not np.array_equal(data1['inv1_rev_fp'], data2['inv1_rev_fp']):
                    continue

                if data1['inv2_inv_result'] != data2['inv2_inv_result']:
                    continue

                if np.array_equal(data1['inv3_fp'], data2['inv3_fp']):
                    row = [row[0] for row in orbits]
                    if v1 in row:
                        ind = row.index(v1)
                        if type(orbits[ind][1]) != list:
                            orbits[ind][1] = [orbits[ind][1]]
                        orbits[ind][1].append(v2)
                    else:
                        orbits.append([v1, v2])
                    v1_matched = True

            if not v1_matched:
                return None

        for k in range(len(orbits)):
            if type(orbits[k][1]) == list:
                orbits[k] = (orbits[k][0], set(orbits[k][1]))
            else:
                orbits[k] = (orbits[k][0], orbits[k][1])

        return orbits
