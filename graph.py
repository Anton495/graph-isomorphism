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

from random import choice, shuffle

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
    
            for key in neurons:
                if key not in graph:
                    continue
                    
                values = []
                for value in graph[key]:
                    edge = (key, value) if key < value else (value, key)
                    if edge not in edges:
                        values.append(value)
                        next_neurons.append(value)
                        edges.add(edge)
    
                if values:
                    layer[key] = values
    
            if not layer:
                break
    
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
    
    def part_test_isomorphism(self,v1,depth,minimal=True):
        
        vertices2 = list(self.graph2.keys())
        
        if minimal == True:
            network_1 = Graph.minimal_oneway_network(self.graph1, v1, depth)
        else:
            network_1 = Graph.oneway_network(self.graph1, v1, depth)
        
        der_network_1 = Graph.network_derivative(network_1)
        
        N = len(self.graph1[v1])
        new_vertices2 = []
        for v2 in vertices2:
            if N == len(self.graph2[v2]):
                
                if minimal == True:
                    network_2 = Graph.minimal_oneway_network(self.graph2, v2, depth)
                else:
                    network_2 = Graph.oneway_network(self.graph2, v2, depth)
                    
                der_network_2 = Graph.network_derivative(network_2)
        
                if der_network_1==der_network_2:
                    new_vertices2.append(v2)
    
        return new_vertices2
    
    def test_isomorphism(self,minimal=True,depth=2):

        if len(self.graph1) != len(self.graph2):
            return False        

        vertices1 = list(self.graph1.keys())
        
        if depth == 0:
            vertices2 = list(self.graph2.keys())
        else:
            vertices2 = self.part_test_isomorphism(vertices1[0],depth,minimal)
        
        if vertices2 == []:
            return False
        
        if minimal == True:
            network_1 = Graph.minimal_oneway_network(self.graph1, vertices1[0])
        else:
            network_1 = Graph.oneway_network(self.graph1, vertices1[0])
        
        der_network_1 = Graph.network_derivative(network_1)
        
        for v2 in vertices2:
                
            if minimal == True:
                network_2 = Graph.minimal_oneway_network(self.graph2, v2)
            else:
                network_2 = Graph.oneway_network(self.graph2, v2)
                    
            der_network_2 = Graph.network_derivative(network_2)
        
            if der_network_1==der_network_2:
                return True
    
        return False
    
    def find_orbits(self,minimal=True,depth=2):
        
        if len(self.graph1) != len(self.graph2):
            return None
        
        vertices1 = list(self.graph1.keys())
        
        if depth == 0:
            vertices2 = list(self.graph2.keys())
        
        orb = []
        for v1 in vertices1:
            
            if depth != 0:
                vertices2 = self.part_test_isomorphism(v1,depth,minimal)
            
            if vertices2 == []:
                return None
            
            if minimal == True:    
                network_1 = Graph.minimal_oneway_network(self.graph1, v1)
            else:
                network_1 = Graph.oneway_network(self.graph1, v1)
            
            der_network_1 = Graph.network_derivative(network_1)
            
            for v2 in vertices2:
                    
                if minimal == True:
                    network_2 = Graph.minimal_oneway_network(self.graph2, v2)
                else:
                    network_2 = Graph.oneway_network(self.graph2, v2)
                        
                der_network_2 = Graph.network_derivative(network_2)
        
                if der_network_1==der_network_2:
                        
                    row = [row[0] for row in orb]
                    if v1 in row:
                        ind = row.index(v1)
                        if type(orb[ind][1]) != list:
                            orb[ind][1] = [orb[ind][1]]
                        
                        orb[ind][1].append(v2)
                    else:
                        orb.append([v1,v2])  
            
            if orb == []:
                return None
        
        for k in range(len(orb)):
            if type(orb[k][1]) == list:
                orb[k] = (orb[k][0],set(orb[k][1]))
            else:
                orb[k] = (orb[k][0],orb[k][1])
                
        return orb
    
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
