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

        node_in = defaultdict(int)
        node_out = defaultdict(int)

        for layer in network:
            for src, targets in layer.items():
                unique_targets = set(targets)
                node_out[src] += len(unique_targets)
                for t in unique_targets:
                    node_in[t] += 1

        deg = lambda n: (node_in[n], node_out[n])

        forward_profiles = []
        reverse_profiles = []

        for layer in network:
            target_to_sources = defaultdict(set)
            for src, targets in layer.items():
                for t in set(targets):
                    target_to_sources[t].add(src)

            fwd_layer = []
            for src, targets in layer.items():
                unique_targets = set(targets)
                fwd_layer.append(
                    (deg(src), sorted(deg(t) for t in unique_targets)))
            fwd_layer.sort()
            forward_profiles.append(fwd_layer)

            rev_layer = []
            for t, sources in target_to_sources.items():
                rev_layer.append(
                    (deg(t), sorted(deg(s) for s in sources)))
            rev_layer.sort()
            reverse_profiles.append(rev_layer)

        return forward_profiles, reverse_profiles
    
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

    @staticmethod
    def compute_layer_degree_map(network):

        res_list = [{} for _ in range(len(network) + 1)]

        for layer_idx, layer in enumerate(network):
            curr_res = res_list[layer_idx]
            next_res = res_list[layer_idx + 1]

            for src, dsts in layer.items():
                if src in curr_res:
                    curr_res[src][1] = len(dsts)
                else:
                    curr_res[src] = [0, len(dsts)]

                for dst in dsts:
                    if dst in next_res:
                        next_res[dst][0] += 1
                    else:
                        next_res[dst] = [1, 0]

        return res_list

    @staticmethod
    def find_loops_and_dead_end_branches(net):

        inv_net = Graph.inverse_network(net)
        layer_degree_map = Graph.compute_layer_degree_map(inv_net)
        depth = len(inv_net)
        result, inv_result = [], []
        
        for n in range(depth):
            child_groups_data = [] 
            used_in_structures = set()
    
            for k, nbs in inv_net[n].items():
                if len(nbs) >= 2:
                    child_groups_data.append((tuple(nbs), tuple([k]*len(nbs)), False))
                    used_in_structures.add(k)
                    used_in_structures.update(nbs)
    
            layer_nodes = list(inv_net[n].keys())
            for i in range(len(layer_nodes)):
                for j in range(i + 1, len(layer_nodes)):
                    u, v = layer_nodes[i], layer_nodes[j]
                    if v in inv_net[n][u] and u in inv_net[n][v]:
                        pair = (u, v)
                        child_groups_data.append(((v, u), (u, v), False))
                        used_in_structures.update(pair)
    
            for node, degrees in layer_degree_map[n].items():
                if node not in used_in_structures and degrees[0] == 0:
                    if node in inv_net[n]:
                        child_groups_data.append((tuple(inv_net[n][node]), (node,), True))
    
            for parent_group, child_group, is_branch in child_groups_data:
                num_branches = len(parent_group)
                group_hist = [[[child_group[i] if i < len(child_group) else child_group[0]], [parent_group[i]]] for i in range(num_branches)]
                inv_hist = [[[layer_degree_map[n][child_group[i] if i < len(child_group) else child_group[0]]], [layer_degree_map[n+1][parent_group[i]]]] for i in range(num_branches)]
    
                intersections_log = []
                branch_mapping = list(range(num_branches)) 
                
                for j in range(n + 1, depth):
                    current_step_nodes = []
                    for b_idx in range(num_branches):
                        last_nodes = group_hist[b_idx][-1]
                        next_nodes = set()
                        for node in last_nodes:
                            next_nodes.update(inv_net[j].get(node, []))
                        
                        next_list = sorted(list(next_nodes))
                        group_hist[b_idx].append(next_list)
                        inv_hist[b_idx].append([layer_degree_map[j+1][v] for v in next_list])
                        current_step_nodes.append(next_nodes)
    
                    for i in range(num_branches):
                        for k in range(i + 1, num_branches):
                            if branch_mapping[i] != branch_mapping[k]:
                                if not current_step_nodes[i].isdisjoint(current_step_nodes[k]) and current_step_nodes[i]:
                                    old_id = branch_mapping[k]
                                    for idx in range(num_branches):
                                        if branch_mapping[idx] == old_id:
                                            branch_mapping[idx] = branch_mapping[i]
                                    intersections_log.append(j + 1)
                                    
                if is_branch:
                    exit_l = depth
                    for step_idx in range(1, len(group_hist[0])):
                        node = group_hist[0][step_idx][0] if group_hist[0][step_idx] else None
                        if node and layer_degree_map[n+step_idx][node][1] != 1:
                            exit_l = n + step_idx + (1 if n == 0 else 0)
                            break
                    label = (n, exit_l, -1)
                else:
                    label = tuple([n] + sorted(intersections_log))
                
                result.append((label, *[h for h in group_hist]))
                inv_result.append((label, *[h for h in inv_hist]))
    
        return sorted(inv_result), result

    @staticmethod
    def get_loops_and_dead_end_branches_intersections(loops_and_dead_end_branches):

        loops_and_dead_end_branches = sorted(
            loops_and_dead_end_branches)

        intersections = defaultdict(lambda: defaultdict(int))
        n = len(loops_and_dead_end_branches)

        indexed = []
        for item in loops_and_dead_end_branches:
            meta = item[0]
            offset = meta[0]
            branches = item[1:]

            layer_map = defaultdict(set)
            for branch in branches:
                for i, layer_content in enumerate(branch):
                    abs_layer = i + offset
                    if isinstance(layer_content, list):
                        layer_map[abs_layer].update(layer_content)
                    else:
                        layer_map[abs_layer].add(layer_content)
            indexed.append(layer_map)

        for i in range(n):
            for j in range(i + 1, n):
                layers_i = indexed[i]
                layers_j = indexed[j]

                common_layers = set(layers_i.keys()) & set(layers_j.keys())

                for layer in common_layers:
                    common_nodes = layers_i[layer] & layers_j[layer]
                    if common_nodes:
                        intersections[(i, j)][layer] = len(common_nodes)

        return {k: dict(v) for k, v in intersections.items()}

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

        return orbits
