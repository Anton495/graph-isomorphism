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
    
        if depth == None: 
            depth = len(graph)-1

        network = []
        neurons = [start_vertex]

        if start_vertex in graph[start_vertex]:
            start_vertex = None
        
        for k in range(depth):
            neurons = [neuron for neuron in neurons if neuron in graph]
            network.append({key:[v for v in graph[key] if v != start_vertex and v != end_vertex] for key in neurons})
        
            neurons = list(network[k].values())
            neurons = sum(neurons, [])
        
        return network
    
    @staticmethod
    def twoway_network(graph, first_vertex, end_vertex):
    
        depth = len(graph)-1
        n = depth//2 if depth//2 == depth/2 else depth//2+1
    
        virtual_network_1 = Graph.oneway_network(graph, first_vertex, n, end_vertex)
    
        if n//2 == n/2:
            network_2 = Graph.oneway_network(graph, end_vertex, n, first_vertex)
        else:
            network_2 = Graph.oneway_network(graph, end_vertex, n-1, first_vertex)
        
        network_2 = list(reversed(network_2))
    
        virtual_network_2 = []

        for k in range(len(network_2)):
            new_layer = {}
            for key,values in network_2[k].items():
                for value in values:   
                    new_layer.setdefault(value,[]).append(key)

            virtual_network_2.append(new_layer)

        return virtual_network_1 + virtual_network_2
    
    @staticmethod
    def minimal_oneway_network(graph, start_vertex, depth=None):

        if depth == None: 
            depth = len(graph)-1

        edges = set()
        network = []
        neurons = [start_vertex]

        for k in range(depth):
            neurons = [neuron for neuron in neurons if neuron in graph]
    
            layer = {}
            new_edges = set()
            for key in neurons:
                values = []
                for value in graph[key]:
                    edge = tuple(sorted([key,value]))
                    if edge not in edges:
                        values.append(value)
                        new_edges.add(edge)
        
                if values != []:
                    layer[key] = values

            n = len(edges)
            edges = edges.union(new_edges)
            
            if len(edges) == n:
                break

            network.append(layer)

            neurons = list(network[k].values())
            neurons = sum(neurons, [])
            
        return network
    
    @staticmethod
    def get_degree_matrix(graph,network):
        
        keys = set(graph.keys())
        values = set(sum(graph.values(),[]))
        vertices = sorted(list(keys|values))

        N = len(vertices)  
        depth = len(network)

        degree_matrix = []
        for n in range(N):
            degree_matrix.append([[0]*(depth+1),[0]*(depth+1)])
                
        for n in range(depth):
            values = sum(network[n].values(),[])
            for m in range(N):
                if vertices[m] in values:
                    degree_matrix[m][0][n+1] = values.count(vertices[m])
                if vertices[m] in network[n]:
                    degree_matrix[m][1][n] = len(network[n][vertices[m]])

        for n in range(N):
            degree_matrix[n] = tuple(degree_matrix[n])
            
        degree_matrix_dict = {}
        for n in range(N):
            degree_matrix_dict[vertices[n]] = degree_matrix[n]
        
        return degree_matrix_dict
    
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
    
    def part_test_isomophism(self,v1,depth,minimal=True):
        
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
    
    def test_isomophism(self,minimal=True,depth=2):

        if len(self.graph1) != len(self.graph2):
            return False        

        vertices1 = list(self.graph1.keys())
        
        if depth == 0:
            vertices2 = list(self.graph2.keys())
        else:
            vertices2 = self.part_test_isomophism(vertices1[0],depth,minimal)
        
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
                vertices2 = self.part_test_isomophism(v1,depth,minimal)
            
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
    
    def find_automorfism(self,orb):
        
        G = tuple(k[0] for k in orb)
        N = len(G)
        
        A = [None]*N
        A_test = [0]*N
        for n in range(N):
            if type(orb[n][1]) == str:
                A[n] = orb[n][1]
                A_test[n] = 1 
        
        if None not in A:
            return [G,tuple(A)]
        
        r = 0
        for m in range(len(orb)):
            if type(orb[m][1]) != str and len(list(orb[m][1])) == 2:
                r = m
                break
        
        A[r] = choice(list(orb[r][1]))
        A_test[r] = 1
        
        for m in range(N):
            i = A_test.index(1)
            vertices1 = self.graph1[G[i]]
            vertices2 = self.graph2[A[i]]
            shuffle(vertices1)
            shuffle(vertices2)
        
            for v1 in vertices1:
                ind = G.index(v1)
            
                for v2 in vertices2:
                    if A[ind] == None and v2 not in A and v2 in orb[ind][1]:
                        A[ind] = v2
                        A_test[ind] = 1
                
            A_test[i] = 0
        
        return [G,tuple(A)]
