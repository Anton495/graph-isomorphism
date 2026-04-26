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
    def find_dead_end_branches(network):

        out_deg = {}
        in_deg = {}
        layer_of = {}
        rev_graph = defaultdict(list)

        for i, layer_dict in enumerate(network):
            for src, dsts in layer_dict.items():
                out_deg[src] = len(dsts)
                layer_of[src] = i
                for dst in dsts:
                    rev_graph[dst].append(src)

        all_vertices = set(out_deg.keys())
        for dst, srcs in rev_graph.items():
            all_vertices.add(dst)

        for v in all_vertices:
            if v not in out_deg:
                out_deg[v] = 0

        for v in all_vertices:
            srcs = rev_graph.get(v, [])
            v_layer = layer_of.get(v)
            in_deg[v] = len([s for s in srcs
                             if layer_of.get(s) != v_layer])

        leaves = [v for v in all_vertices
                  if in_deg[v] == 1 and out_deg[v] == 0]

        result = {}
        for leaf in leaves:
            branch = {}
            curr = leaf

            while True:
                curr_layer = layer_of.get(curr)
                parents = [p for p in rev_graph.get(curr, [])
                           if layer_of.get(p) != curr_layer]

                if not parents:
                    break

                parent = parents[0]
                parent_in = in_deg[parent]
                parent_out = out_deg[parent]
                layer = layer_of[parent]

                branch[layer] = (parent_in, parent_out)

                if parent_in != 1:
                    break

                curr = parent

            result[leaf] = branch

        return result

    @staticmethod
    def normalize_dead_end_branches(dead_ends):

        return sorted(
            tuple(sorted(branch.items()))
            for branch in dead_ends.values()
        )

    @staticmethod
    def find_loops_old(network):

        out_edges = {}
        in_edges = defaultdict(list)
        layer_of = {}

        for layer_idx, layer_dict in enumerate(network):
            for src, dsts in layer_dict.items():
                out_edges[src] = dsts
                layer_of[src] = layer_idx
                for dst in dsts:
                    in_edges[dst].append(src)

        loops_found = []
        for a, dsts_a in out_edges.items():
            for b in dsts_a:
                if b in out_edges and a in out_edges[b]:
                    if a < b:
                        layer_close = layer_of[a]

                        visited_a = {a}
                        visited_b = {b}
                        queue_a = deque([a])
                        queue_b = deque([b])
                        common = None

                        while queue_a or queue_b:
                            if queue_a and not common:
                                next_a = []
                                for node in queue_a:
                                    for parent in in_edges.get(node, []):
                                        if node == a and parent == b:
                                            continue
                                        if parent in visited_a:
                                            continue
                                        visited_a.add(parent)
                                        next_a.append(parent)
                                        if parent in visited_b:
                                            common = parent
                                            break
                                    if common:
                                        break
                                queue_a = deque(next_a)

                            if common:
                                break

                            if queue_b and not common:
                                next_b = []
                                for node in queue_b:
                                    for parent in in_edges.get(node, []):
                                        if node == b and parent == a:
                                            continue
                                        if parent in visited_b:
                                            continue
                                        visited_b.add(parent)
                                        next_b.append(parent)
                                        if parent in visited_a:
                                            common = parent
                                            break
                                    if common:
                                        break
                                queue_b = deque(next_b)

                        if common is not None:
                            layer_start = layer_of[common]
                            loops_found.append(f"{layer_close}{layer_start}")

        loops_found.sort()
        return loops_found

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
    def find_loops(network):
        
        inv_net = Graph.inverse_network(network)
        depth = len(inv_net)
        
        result = []
        for n in range(depth):
        
            parent_groups = []
            for k, nbs in inv_net[n].items():
                if len(nbs) >= 2:
                    parent_groups.append(tuple(nbs))
            
                for nb in nbs:
                    if nb in inv_net[n] and k in inv_net[n][nb]:
                        pair = tuple(sorted((k,nb)))
                        if pair not in parent_groups:
                            parent_groups.append(pair)
            
            for m in range(len(parent_groups)):
            
                start_groups = [[v] for v in parent_groups[m]]
                for j in range(n + 1, depth):
                    loop = str(n)
                    new_group = []
                    
                    for group in start_groups:
                        current_union = set()
                        for key in group:
                            val = inv_net[j].get(key, [])
                            current_union.update(val)
                        new_group.append(list(current_union))
                    
                    seen_elements = set()
                    intersections_count = 0
                    
                    for g in new_group:
                        current_set = set(g)
                        if not current_set.isdisjoint(seen_elements):
                            intersections_count += 1
                        
                        seen_elements.update(current_set)
                    
                    if intersections_count > 0:
                        for _ in range(intersections_count):
                            loop += str(j)
                        
                        result.append(loop)
                        break  
                    else:
                        start_groups = new_group

        return sorted(result)
    
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
        bdp1 = Graph.compute_bidirectional_degree_profiles(net1)
        deb1 = Graph.normalize_dead_end_branches(
            Graph.find_dead_end_branches(net1))
        loops1 = Graph.find_loops(net1)

        for v2 in candidates:
            net2 = Graph.minimal_oneway_network(self.graph2, v2)

            bdp2 = Graph.compute_bidirectional_degree_profiles(net2)
            if bdp1 != bdp2:
                continue

            deb2 = Graph.normalize_dead_end_branches(
                Graph.find_dead_end_branches(net2))
            if deb1 != deb2:
                continue

            loops2 = Graph.find_loops(net2)
            if loops1 == loops2:
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
        g2_net = {}
        for v2 in vertices2:
            net2 = Graph.minimal_oneway_network(self.graph2, v2)
            g2_net[v2] = net2
            g2_bdp[v2] = Graph.compute_bidirectional_degree_profiles(net2)

        g2_deb = {}

        def get_deb(v2):
            if v2 not in g2_deb:
                g2_deb[v2] = Graph.normalize_dead_end_branches(
                    Graph.find_dead_end_branches(g2_net[v2]))
            return g2_deb[v2]

        g2_loops = {}

        def get_loops(v2):
            if v2 not in g2_loops:
                g2_loops[v2] = Graph.find_loops(g2_net[v2])
            return g2_loops[v2]

        orbits = []
        for v1 in vertices1:

            net1 = Graph.minimal_oneway_network(self.graph1, v1)
            bdp1 = Graph.compute_bidirectional_degree_profiles(net1)
            deb1 = Graph.normalize_dead_end_branches(
                Graph.find_dead_end_branches(net1))
            loops1 = Graph.find_loops(net1)

            d1 = len(self.graph1[v1])
            v1_matched = False

            for v2 in vertices2:

                if len(self.graph2[v2]) != d1:
                    continue

                bdp2 = g2_bdp[v2]
                if bdp1 != bdp2:
                    continue

                deb2 = get_deb(v2)
                if deb1 != deb2:
                    continue

                loops2 = get_loops(v2)
                if loops1 == loops2:

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
