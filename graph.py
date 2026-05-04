__LICENSE__ = '''

The MIT License

Copyright (c) 2024 Korneev Anton Aleksandrovich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''
from collections import Counter, defaultdict, deque
from random import choice, shuffle, random

class Graph:
    
    def __init__(self, graph1, graph2 = None):
         
        self.graph1 = graph1
        self.graph2 = graph2 if graph2 is not None else self.get_graph2()
        
    def get_graph2(self):
    
        graph2 = {}
        for key,values in self.graph1.items():
            graph2[str(key)+'~'] = [str(value)+'~' for value in values]
    
        return graph2
    
    @staticmethod
    def oneway_network(graph, start_vertex, depth=None, end_vertex=None):
        
        if depth is None:
            depth = len(graph) - 1

        network = []
        neurons = [start_vertex]
        exclude = {start_vertex, end_vertex} if end_vertex is not None else {start_vertex}

        for _ in range(depth):
            valid_neurons = [n for n in neurons if n in graph]
            if not valid_neurons:
                break

            layer = {}
            next_neurons = []

            for key in valid_neurons:
                neighbors = []
                for v in graph[key]:
                    if v not in exclude:
                        neighbors.append(v)
                        next_neurons.append(v)
                
                if neighbors:
                    layer[key] = neighbors

            if not layer:
                break

            network.append(layer)
            neurons = next_neurons

        return network

    @staticmethod
    def twoway_network(graph, first_vertex, end_vertex):
        
        max_depth = len(graph) - 1
        half_depth = (max_depth + 1) // 2

        forward = Graph.oneway_network(graph, first_vertex, half_depth, end_vertex)

        reverse_depth = half_depth if max_depth % 2 == 0 else half_depth - 1
        backward = Graph.oneway_network(graph, end_vertex, reverse_depth, first_vertex)

        inverted_backward = []
        for layer in reversed(backward):
            inverted = {}
            for src, targets in layer.items():
                for tgt in targets:
                    inverted.setdefault(tgt, []).append(src)
            inverted_backward.append(inverted)

        return forward + inverted_backward

    @staticmethod
    def minimal_oneway_network(graph, start_vertex, depth=None):
        
        if depth is None:
            depth = len(graph) - 1
            
        edges = set()
        network = []
        neurons = [start_vertex]
        
        for _ in range(depth):
            if not neurons:
                break
            layer = {}
            next_neurons = []
            new_edges = set()
            
            for key in neurons:
                if key not in graph:
                    continue
                values = []
                
                for value in graph[key]:
                    edge = (key, value) if key < value else (value, key)
                    if edge not in edges:
                        values.append(value)
                        next_neurons.append(value)
                        new_edges.add(edge)
                        
                if values:
                    layer[key] = values
                    
            if not layer:
                break
            
            edges.update(new_edges)
            network.append(layer)
            neurons = next_neurons
            
        return network

    @staticmethod
    def get_degree_matrix(network):
        
        depth = len(network)
        in_degrees = {}
        out_degrees = {}
    
        for n, layer in enumerate(network):
            for src, dsts in layer.items():
                out_degrees.setdefault(src, {})[n] = len(dsts)
                
                next_n = n + 1
                for dst in dsts:
                    in_dict = in_degrees.setdefault(dst, {})
                    in_dict[next_n] = in_dict.get(next_n, 0) + 1
    
        degree_matrix = {}
        for v in in_degrees.keys() | out_degrees.keys():
            in_list = [0] * (depth + 1)
            out_list = [0] * (depth + 1)
            
            for idx, cnt in in_degrees.get(v, {}).items():
                in_list[idx] = cnt
            for idx, cnt in out_degrees.get(v, {}).items():
                out_list[idx] = cnt
                
            degree_matrix[v] = (in_list, out_list)
    
        return degree_matrix

    @staticmethod
    def get_degree_dict(network):
        degree_dict = {}
        for layer_idx, layer in enumerate(network):
            for src, dsts in layer.items():
                out_d = len(dsts)
                
                entry_src = degree_dict.setdefault(src, {})
                entry_src.setdefault(layer_idx, [0, 0])[1] = out_d
                
                next_idx = layer_idx + 1
                for dst in dsts:
                    entry_dst = degree_dict.setdefault(dst, {})
                    entry_dst.setdefault(next_idx, [0, 0])[0] += 1
                    
        return degree_dict

    @staticmethod
    def get_degree_dict_hashes(network, inv=False):
        """
        Static Method: Computes structural invariant signatures for vertices in a multilayer network.
        
        This function analyzes the in-degree and out-degree of each vertex across all layers 
        to create a label-independent signature. To ensure mathematical certainty for 
        isomorphism proofs, it uses full sorted tuples as signatures instead of hash values, 
        thereby eliminating any possibility of hash collisions.
        
        Args:
            network (list[dict]): A list of layers, where each layer is a 
                dictionary {source_node: [target_nodes]}.
            inv (bool): If False (default), returns a mapping from vertex 
                labels to their structural signatures. 
                If True, returns a mapping from signatures to lists of vertex 
                labels (grouping symmetric vertices).
                
        Returns:
            dict: 
                - If inv=False: {vertex: signature_tuple}
                - If inv=True:  {signature_tuple: [vertex1, vertex2, ...]}
                
        Note:
            The signature is a sorted tuple of (layer_index, (in_degree, out_degree)).
            Using tuples directly as keys is computationally safe in Python and 
            guarantees that distinct structures will never be treated as identical.
            
        Example:
            >>> net = [{4: [9, 10]}]
            >>> A = Graph.get_degree_dict_hashes(net, inv=False)
            >>> B = Graph.get_degree_dict_hashes(net, inv=True)
            >>> set(A.values()) == set(B.keys())
            True
        """
        degree_dict = {}
        # 1. Collect in/out degrees for each vertex across layers
        for layer_idx, layer in enumerate(network):
            for src, dsts in layer.items():
                # Set out-degree for source
                degree_dict.setdefault(src, {}).setdefault(layer_idx, [0, 0])[1] = len(dsts)
                # Increment in-degree for each target
                for dst in dsts:
                    degree_dict.setdefault(dst, {}).setdefault(layer_idx + 1, [0, 0])[0] += 1
    
        # 2. Generate (vertex, structural_signature) pairs
        v_signature_pairs = (
            (v, tuple(sorted((l, tuple(d)) for l, d in layers.items())))
            for v, layers in degree_dict.items()
        )
    
        # 3. Format output based on 'inv' flag
        if not inv:
            return dict(v_signature_pairs)
        
        result = defaultdict(list)
        for v, signature in v_signature_pairs:
            result[signature].append(v)
        return dict(result)
    
    @staticmethod
    def network_derivative(network):

        depth = len(network) 
        network_der = []
        for k in range(depth-1):  

            current_keys = list(network[k].keys())

            layer = []
            for m in current_keys:

                current_values = network[k][m]
        
                num = []
                for j in current_values:
                    if j in network[k+1]:
                        next_values = network[k+1][j]
                        num.append(len(next_values))
            
                if num != []:
                    layer.append((len(current_values),sorted(num)))
                    
            network_der.append(sorted(layer))
        
        return network_der

    @staticmethod
    def compute_bidirectional_degree_profiles(network):
        """
        Computes global degree-based structural invariants and mapping for path tracing.
    
        This method performs two passes over the network to calculate global node degrees 
        and constructs bidirectional profiles that serve as topological invariants. 
        It also generates a specialized global degree map for identifying loops 
        and dead-end branches.
    
        Args:
            network (list[dict]): A sequence of layers where each dictionary maps 
                source node IDs to lists of target node IDs.
    
        Returns:
            tuple: A triple containing:
                1. forward_profiles (list[list[tuple]]): Forward structural invariants.
                   Format: [(global_node_degree, sorted([global_target_degrees]))]. 
                   Sorted within each layer to provide a canonical representation.
                
                2. reverse_profiles (list[list[tuple]]): Reverse structural invariants.
                   Format: [(global_target_degree, sorted([global_source_degrees]))]. 
                   Sorted within each layer to provide a canonical representation.
    
                3. reverse_degree_map (list[dict]): A layer-by-layer mapping of global 
                   node degrees, indexed in reversed order with a length of len(network) + 1. 
                   Contains global degrees formatted as [out_degree, in_degree] to 
                   support sink-to-root tracing in `find_loops_and_dead_end_branches`.
    
        Note:
            - Global degrees are (in_degree, out_degree) calculated across the entire network.
            - In reverse_degree_map, degrees are swapped to [out, in] to reflect 
              the reversed tracing logic (where a source in the original network 
              becomes a target in the reversed path).
        """
        node_in_g = defaultdict(int)
        node_out_g = defaultdict(int)
    
        for layer in network:
            for src, targets in layer.items():
                u_targets = set(targets)
                node_out_g[src] += len(u_targets)
                for t in u_targets:
                    node_in_g[t] += 1
    
        num_layers = len(network)
        forward_profiles = []
        reverse_profiles = []
        
        res_list_global = [{} for _ in range(num_layers + 1)]
    
        for i, layer in enumerate(network):
            fwd_layer = []
            rev_layer_data = defaultdict(list)
            
            curr_res = res_list_global[i]
            next_res = res_list_global[i + 1]
    
            for src, targets in layer.items():
                u_targets = set(targets)
                src_deg_tuple = (node_in_g[src], node_out_g[src])
                
                if src not in curr_res:
                    curr_res[src] = [node_in_g[src], node_out_g[src]]
    
                target_degs_g = []
                for t in u_targets:
                    t_deg_tuple = (node_in_g[t], node_out_g[t])
                    target_degs_g.append(t_deg_tuple)
                    rev_layer_data[t].append(src_deg_tuple)
                    
                    if t not in next_res:
                        next_res[t] = [node_in_g[t], node_out_g[t]]
    
                fwd_layer.append((src_deg_tuple, sorted(target_degs_g)))
    
            fwd_layer.sort()
            forward_profiles.append(fwd_layer)
    
            rev_layer_list = []
            for t, parents_degs in rev_layer_data.items():
                t_deg_g = (node_in_g[t], node_out_g[t])
                rev_layer_list.append((t_deg_g, sorted(parents_degs)))
            rev_layer_list.sort()
            reverse_profiles.append(rev_layer_list)
    
        reverse_degree_map = []
        for layer in reversed(res_list_global):
            reverse_degree_map.append({node: [d[1], d[0]] for node, d in layer.items()})
    
        return forward_profiles, reverse_profiles, reverse_degree_map

    @staticmethod
    def inverse_network(network):

        inverted_network = []
        for layer in reversed(network):
            inv_layer = {}
            for key, values in layer.items():
                for val in values:
                    if val not in inv_layer:
                        inv_layer[val] = []
                    inv_layer[val].append(key)
            inverted_network.append(inv_layer)

        return inverted_network

    def find_loops_and_dead_end_branches(net, layer_degree_map):
        """
        Identifies structural loops and dead-end branches using a Breadth-First Search (BFS)
        traversal from sinks back to the origin.

        The function performs a layer-by-layer backward traversal. Structural features
        are identified based on local node degrees within each layer:

        1. Multiple-path loops: Detected at convergence points where the local
           reversed out-degree (originally local in-degree) is >= 2.
        2. Reciprocal edges: Detected as mutual connections between nodes within the same layer.
        3. Dead-end branches: Detected at nodes with local reversed in-degree 0
           (originally local out-degree 0).

        Args:
            net (list[dict]): A sequence of layers representing the forward network.
            layer_degree_map (list[dict]): A mapping of global node degrees indexed
                in reversed order (length = len(net) + 1). Each entry contains
                global [out_degree, in_degree] relative to the reversed logic.

        Returns:
            tuple: A pair (sorted_inv_result, result) containing:
                1. inv_result (list): Sorted tracing history using local node degree profiles,
                   serving as a structural invariant.
                2. result (list): Tracing history using raw node IDs.

        Note:
            - The BFS approach ensures that all paths of a multi-branch structure are
              traced simultaneously layer-by-layer back to the origin node.
            - The process identifies where loops "close" (on the sink side) and
              moves backward to find where they "open" (closer to the origin).
            - Tracing persists until the origin node is reached for all branches.
        """
        # Pass 1: Invert the forward network to trace paths from sinks back to origin
        inv_net = Graph.inverse_network(net)
        depth = len(inv_net)
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
    
            # 2. Detect reciprocal edges (mutual connections)
            layer_nodes = list(inv_net[n].keys())
            for i in range(len(layer_nodes)):
                for j in range(i + 1, len(layer_nodes)):
                    u, v = layer_nodes[i], layer_nodes[j]
                    if v in inv_net[n].get(u, []) and u in inv_net[n].get(v, []):
                        pair = (u, v)
                        parent_groups.append((v, u))
                        child_group.append((u, v))
                        is_branch.append(False)
                        used_in_structures.update(pair)
    
            # 3. Detect dead-end branches (in=0, out=1 strictly inside inv_net)
            # Check if the node was targeted in the previous layer of inv_net
            prev_inv_targets = set()
            if n > 0:
                for targets in inv_net[n - 1].values():
                    prev_inv_targets.update(targets)
    
            for node, nbs in inv_net[n].items():
                # A node is a dead-end if it has exactly 1 parent in inv_net 
                # and no incoming edges in the reversed direction
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
    
                    inv_group_history.append([[c_inv], [layer_degree_map[n + 1][v] for v in current_groups[i]]])
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
                            inv_group_history[b_idx].append([layer_degree_map[j + 1][v] for v in next_list])
                            group_history[b_idx].append(next_list)
    
                    # Track intersection layers for loops
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
    
                # Finalize the Label based on structural type
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
    
                result.append((label, *sorted(group_history)))
                inv_result.append((label, *sorted(inv_group_history)))
    
        return sorted(inv_result), result

    def get_loops_and_dead_end_branches_intersections(loops_and_dead_end_branches):
        
        indexed = []
        for item in loops_and_dead_end_branches:
            meta = item[0]
            start = meta[0]
            branches = item[1:]
            
            element_branches = []
            for branch in branches:
                branch_layers = {}
                for i, layer_content in enumerate(branch):
                    abs_layer = i + start
                    if isinstance(layer_content, (list, tuple, set)):
                        branch_layers[abs_layer] = set(layer_content)
                    else:
                        branch_layers[abs_layer] = {layer_content}
                element_branches.append(branch_layers)
            indexed.append(element_branches)
        
        temp_output = []
        for item in loops_and_dead_end_branches:
            branches = item[1:]
            temp_output.append([{} for _ in branches])
            
        n = len(loops_and_dead_end_branches)
        
        for i in range(n):
            for j in range(i + 1, n):
                layers_i = set(l for b in indexed[i] for l in b)
                layers_j = set(l for b in indexed[j] for l in b)
                common_layers = layers_i & layers_j
                
                for layer in common_layers:
                    counts_i_to_j = []
                    for b_i in indexed[i]:
                        if layer in b_i:
                            row = []
                            for b_j in indexed[j]:
                                if layer in b_j:
                                    row.append(len(b_i[layer] & b_j[layer]))
                                else:
                                    row.append(0)
                            counts_i_to_j.append(row)
                        else:
                            counts_i_to_j.append([])
                    
                    counts_j_to_i = []
                    for b_j in indexed[j]:
                        if layer in b_j:
                            row = []
                            for b_i in indexed[i]:
                                if layer in b_i:
                                    row.append(len(b_j[layer] & b_i[layer]))
                                else:
                                    row.append(0)
                            counts_j_to_i.append(row)
                        else:
                            counts_j_to_i.append([])
    
                    for b_i_idx, branch_counts in enumerate(counts_i_to_j):
                        if not branch_counts:
                            continue
                        filtered = sorted([c for c in branch_counts if c > 0])
                        if filtered:
                            if layer not in temp_output[i][b_i_idx]:
                                temp_output[i][b_i_idx][layer] = {}
                            temp_output[i][b_i_idx][layer][j] = filtered
    
                    for b_j_idx, branch_counts in enumerate(counts_j_to_i):
                        if not branch_counts:
                            continue
                        filtered = sorted([c for c in branch_counts if c > 0])
                        if filtered:
                            if layer not in temp_output[j][b_j_idx]:
                                temp_output[j][b_j_idx][layer] = {}
                            temp_output[j][b_j_idx][layer][i] = filtered
    
        final_output = []
        for i, item in enumerate(loops_and_dead_end_branches):
            meta = item[0]
            start_i = meta[0]
            branches = item[1:]
            
            element_branches_set = set()
            for b_idx, branch in enumerate(branches):
                branch_list = []
                for k in range(len(branch)):
                    abs_layer = start_i + k
                    layer_data = temp_output[i][b_idx].get(abs_layer)
                    if layer_data:
                        branch_list.append(frozenset(
                            (j, tuple(sorted(counts))) for j, counts in layer_data.items()
                        ))
                    else:
                        branch_list.append(None)
                
                element_branches_set.add(tuple(branch_list))
                
            final_output.append(element_branches_set)
    
        return sorted(final_output)

    def test_is_isomorphic(self):

        if len(self.graph1) != len(self.graph2):
            return False

        vertices1 = list(self.graph1.keys())
        vertices2 = list(self.graph2.keys())

        if not vertices1 or not vertices2:
            return False

        e1 = sum(len(vs) for vs in self.graph1.values())
        e2 = sum(len(vs) for vs in self.graph2.values())
        if e1 != e2:
            return False

        deg_groups_1 = {}
        for v in self.graph1:
            d = len(self.graph1.get(v, []))
            deg_groups_1.setdefault(d, []).append(v)

        deg_groups_2 = {}
        for v in self.graph2:
            d = len(self.graph2.get(v, []))
            deg_groups_2.setdefault(d, []).append(v)

        deg_dist_1 = {d: len(vs) for d, vs in deg_groups_1.items()}
        deg_dist_2 = {d: len(vs) for d, vs in deg_groups_2.items()}

        if deg_dist_1 != deg_dist_2:
            return False

        ref_deg = min(deg_groups_1, key=lambda d: len(deg_groups_1[d]))
        ref_vertex = random.choice(deg_groups_1[ref_deg])

        candidates = deg_groups_2.get(ref_deg, [])

        net1 = Graph.minimal_oneway_network(self.graph1, ref_vertex)
        bdp1_fwd, bdp1_rev, rdm1 = Graph.compute_bidirectional_degree_profiles(net1)
        bdp1 = (bdp1_fwd, bdp1_rev)

        ld2_1 = None
        ld3_1 = None

        for v2 in candidates:
            net2 = Graph.minimal_oneway_network(self.graph2, v2)

            bdp2_fwd, bdp2_rev, rdm2 = Graph.compute_bidirectional_degree_profiles(net2)
            if bdp1 != (bdp2_fwd, bdp2_rev):
                continue

            if ld2_1 is None:
                ld2_1 = Graph.find_loops_and_dead_end_branches(
                    net1, layer_degree_map=rdm1)
            ld2_inv2, ld2_res2 = Graph.find_loops_and_dead_end_branches(
                net2, layer_degree_map=rdm2)
            if ld2_1[0] != ld2_inv2:
                continue

            if ld3_1 is None:
                ld3_1 = Graph.get_loops_and_dead_end_branches_intersections(
                    ld2_1[1])
            ld3_2 = Graph.get_loops_and_dead_end_branches_intersections(
                ld2_res2)
            if ld3_1 == ld3_2:
                return True

        return False

    @staticmethod
    def _has_perfect_matching(orbits, vertices2):

        n = len(orbits)
        if n != len(vertices2):
            return False

        v2_to_idx = {v: i for i, v in enumerate(vertices2)}

        adj = []
        for v1, s2 in orbits:
            targets = s2 if isinstance(s2, set) else {s2}
            neighbors = [v2_to_idx[v] for v in targets if v in v2_to_idx]
            if not neighbors:
                return False
            adj.append(neighbors)

        match_r = [-1] * n

        def bpm(u, seen):
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    if match_r[v] == -1 or bpm(match_r[v], seen):
                        match_r[v] = u
                        return True
            return False

        for u in range(n):
            seen = [False] * n
            if not bpm(u, seen):
                return False
        return True

    def test_find_orbits(self):

        nodes_count1 = len(self.graph1)
        nodes_count2 = len(self.graph2)
        nodes_count_match = nodes_count1 == nodes_count2

        edges_count1 = sum(len(v1) for v1 in self.graph1.values())
        edges_count2 = sum(len(v2) for v2 in self.graph2.values())
        edges_count_match = edges_count1 == edges_count2

        degree_sequence1 = sorted(len(nb1) for nb1 in self.graph1.values())
        degree_sequence2 = sorted(len(nb2) for nb2 in self.graph2.values())
        degrees_match = degree_sequence1 == degree_sequence2

        if not (nodes_count_match and edges_count_match and degrees_match):
            return None

        vertices1 = list(self.graph1.keys())
        vertices2 = list(self.graph2.keys())

        g2_bdp = {}
        g2_rdm = {}
        g2_net = {}
        for v2 in vertices2:
            net2 = Graph.minimal_oneway_network(self.graph2, v2)
            g2_net[v2] = net2
            fwd, rev, rdm = Graph.compute_bidirectional_degree_profiles(net2)
            g2_bdp[v2] = (fwd, rev)
            g2_rdm[v2] = rdm

        g2_ld2 = {}

        def get_ld2(v2):
            if v2 not in g2_ld2:
                g2_ld2[v2] = Graph.find_loops_and_dead_end_branches(
                    g2_net[v2], layer_degree_map=g2_rdm[v2])
            return g2_ld2[v2]

        g2_ld3 = {}

        def get_ld3(v2):
            if v2 not in g2_ld3:
                _, ld2_res2 = get_ld2(v2)
                g2_ld3[v2] = \
                    Graph.get_loops_and_dead_end_branches_intersections(
                        ld2_res2)
            return g2_ld3[v2]

        orbits = []
        for v1 in vertices1:

            net1 = Graph.minimal_oneway_network(self.graph1, v1)
            bdp1_fwd, bdp1_rev, rdm1 = Graph.compute_bidirectional_degree_profiles(
                net1)
            ld2_inv1, ld2_res1 = Graph.find_loops_and_dead_end_branches(
                net1, layer_degree_map=rdm1)
            ld3_1 = Graph.get_loops_and_dead_end_branches_intersections(
                ld2_res1)

            d1 = len(self.graph1[v1])
            v1_matched = False

            for v2 in vertices2:

                if len(self.graph2[v2]) != d1:
                    continue

                bdp2 = g2_bdp[v2]
                if (bdp1_fwd, bdp1_rev) != bdp2:
                    continue

                ld2_inv2, ld2_res2 = get_ld2(v2)
                if ld2_inv1 != ld2_inv2:
                    continue

                ld3_2 = get_ld3(v2)
                if ld3_1 == ld3_2:

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

        if not Graph._has_perfect_matching(orbits, vertices2):
            return None

        return orbits

      def find_automorphism(self,orbits):
        
        vertices = tuple(orbit[0] for orbit in orbits)
        num_vertices = len(vertices)
        
        mapping = [None]*num_vertices
        in_processing = [0]*num_vertices
        for n in range(num_vertices):
            if type(orbits[n][1]) == str:
                mapping[n] = orbits[n][1]
                in_processing[n] = 1 
        
        if None not in mapping:
            return [vertices,tuple(mapping)]
        
        for n in range(num_vertices):
            if type(orbits[n][1]) != str and len(list(orbits[n][1])) == 2:
                break
        
        mapping[n] = choice(list(orbits[n][1]))
        in_processing[n] = 1
        
        for n in range(num_vertices):
            i = in_processing.index(1)
            neighbors1 = self.graph1[vertices[i]]
            neighbors2 = self.graph2[mapping[i]]
            shuffle(neighbors1)
            shuffle(neighbors2)
        
            for nb1 in neighbors1:
                ind = vertices.index(nb1)
            
                for nb2 in neighbors2:
                    if mapping[ind] == None and nb2 not in mapping and nb2 in orbits[ind][1]:
                        mapping[ind] = nb2
                        in_processing[ind] = 1
            
            in_processing[i] = 0
        
        return [vertices,tuple(mapping)]
