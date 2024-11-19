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
    def oneway_network(graph, start_vertex, end_vertex=None, N=None):
    
        if N == None: 
            N = len(graph)-1

        network = []
        neurons = [start_vertex]

        if start_vertex in graph[start_vertex]:
            start_vertex = None
        
        for k in range(N):
            neurons = [m for m in neurons if m in graph]
            network.append({key:[k for k in graph[key] if k != start_vertex and k != end_vertex] for key in neurons})
        
            neurons = list(network[k].values())
            neurons = sum(neurons, [])
        
        return network
    
    @staticmethod
    def twoway_network(graph, first_vertex, end_vertex=None):
    
        n = len(graph) if end_vertex==None else len(graph)-1
        N = n//2 if n//2 == n/2 else n//2+1
    
        virtual_network_1 = Graph.oneway_network(graph, first_vertex, end_vertex, N)
    
        if N//2 == N/2:
            if end_vertex==None:        
                network_2 = list(reversed(virtual_network_1))
            else:
                network_2 = Graph.oneway_network(graph, end_vertex, first_vertex, N)
                network_2 = list(reversed(network_2))
        else:
            if end_vertex==None:  
                network_2 = list(reversed(virtual_network_1[:-1]))
            else:
                network_2 = Graph.oneway_network(graph, end_vertex, first_vertex, N-1)
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
    def inverse_network(network,first_vertex,only_keys=True):
    
        rev_network = list(reversed(network))

        all_keys = set()
        inv_network = []
        for k in range(len(rev_network)):
    
            if k == 0:
                true_keys = [first_vertex]
            else:    
                true_keys = sum(inv_network[k-1].values(),[])

            new_layer = {}
            for key,values in rev_network[k].items():
                for value in values:
                    if value in true_keys:    
                        new_layer.setdefault(value,[]).append(key)
                        all_keys.add(key)

            inv_network.append(new_layer)
    
        if only_keys == True:
            return all_keys
        else:
            return inv_network

    @staticmethod
    def network_with_reversed(graph, start_vertex):
        '''This function does not support directed graphs'''
        
        network_rev = [{start_vertex : graph[start_vertex]}]

        for k in range(len(graph)):
            layer = {}
            all_values = list(set(sum(network_rev[k].values(),[])))

            for value in all_values:
                test_values = graph[value]
                new_values = []
                for v in test_values:
                    if v not in Graph.inverse_network(network_rev,value):
                        new_values.append(v)
            
                if new_values != []:
                    layer[value] = new_values

            if layer != {}:
                network_rev.append(layer)
            else:
                return network_rev

    @staticmethod
    def network_derivative(network):

        N = len(network) 
        network_der = []
        for k in range(N-1):  

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

    def test_isomophism(self, rev=False):

        vertices1 = list(self.graph1.keys())
        vertices2 = list(self.graph2.keys())
        
        if rev == True:
            network_1 = Graph.network_with_reversed(self.graph1, vertices1[0])
        else:
            network_1 = Graph.oneway_network(self.graph1, vertices1[0])
        
        der_network_1 = Graph.network_derivative(network_1)
        
        N = len(self.graph1[vertices1[0]])
        for v2 in vertices2:
            if N == len(self.graph2[v2]):
                
                if rev == True:
                    network_2 = Graph.network_with_reversed(self.graph2, v2)
                else:
                    network_2 = Graph.oneway_network(self.graph2, v2)
                
                der_network_2 = Graph.network_derivative(network_2)
        
                if der_network_1==der_network_2:
                    return True
    
        return False

    def find_orbits(self, rev=False):
    
        vertices1 = list(self.graph1.keys())
        vertices2 = list(self.graph2.keys())
    
        isom = []
        for v1 in vertices1:
            
            if rev == True:
                network_1 = Graph.network_with_reversed(self.graph1, v1)
            else:
                network_1 = Graph.oneway_network(self.graph1, v1)
                
            der_network_1 = Graph.network_derivative(network_1)
    
            for v2 in vertices2:
                if len(self.graph1[v1]) == len(self.graph2[v2]):
                    
                    if rev == True:
                        network_2 = Graph.network_with_reversed(self.graph2, v2)
                    else:
                        network_2 = Graph.oneway_network(self.graph2, v2)
                        
                    der_network_2 = Graph.network_derivative(network_2)
        
                    if der_network_1==der_network_2:
                        
                        row = [row[0] for row in isom]
                        if v1 in row:
                            ind = row.index(v1)
                            if type(isom[ind][1]) != list:
                                isom[ind][1] = [isom[ind][1]]
                                
                            isom[ind][1].append(v2)
                        else:
                            isom.append([v1,v2])  
            
            if isom == []:
                return False
        
        for k in range(len(isom)):
            if type(isom[k][1]) == list:
                isom[k] = (isom[k][0],set(isom[k][1]))
            else:
                isom[k] = (isom[k][0],isom[k][1])
                
        return isom

    def find_automorfism(self,iso):

        G = tuple(k[0] for k in iso)
        A = [None]*len(G)
        A_test = [0]*len(G)
        
        r = None
        for m in range(len(iso)):
            if len(iso[m][1]) == 2:
                r = m
                break
        
        if r == None:
            A[0] = choice(list(iso[0][1]))
            A_test[0] = 1
        else:
            A[r] = choice(list(iso[r][1]))
            A_test[r] = 1
        
        for m in range(len(A)):    
            i = A_test.index(1)
            v1 = self.graph1[G[i]]
            v2 = self.graph2[A[i]]
            shuffle(v1)
            shuffle(v2)
        
            for k1 in range(len(v1)):
                ind = G.index(v1[k1])
            
                for k2 in range(len(v2)):
                    if A[ind] == None and v2[k2] not in A and v2[k2] in iso[ind][1]:
                        A[ind] = v2[k2]
                        A_test[ind] = 1
                
            A_test[i] = 0
        
        return [G,tuple(A)]
