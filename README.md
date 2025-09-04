# Graph isomorphism

This package have two basic functions:

1. The `test_isomophism(minimal=True,depth=2)` function can be used to test graphs for isomorphism. Outputs `True` or `False`.

2. The `find_orbits(minimal=True,depth=2)` function can be used to find set of isomorphic vertices for each vertex of the graph. For non-symmetric graphs, outputs isomorphism substitution.

Runtime depends polynomially on the number of vertices and edges. In the first case the complexity of the algorithm is $O(|V|^{c}\cdot(|V|+|E|))$, in the second case it is $O(|V|^{1+c}\cdot(|V|+|E|))$, where $0<c<1$, $|V|$ is the number of vertices and $|E|$ is the number of edges in the graph. With the correct choice of `depth`, the parameter $c$ is close to zero. In most cases, it is sufficient to choose a relatively small `depth`. Here `depth>=2`.

Directed graphs and pseudomultigraphs are supported.

The module has been tested to work on Python 3.12.7.

# Usage

Run examples.py for usage in an interactive session.

```python
>>> g1 = {'1': ['2'],'2': ['1','3'], '3': ['2']}
>>> g2 = {'a': ['b','c'],'b': ['a','c'], 'c': ['a','b']}
>>> example = Graph(g1,g2)
>>> example.test_isomophism()
False
>>> example1().graph1
{'1': ['2', '3'],
 '2': ['1', '3', '4', '5'],
 '3': ['1', '2', '4', '5'],
 '4': ['2', '3', '5'],
 '5': ['2', '3', '4']}
>>> example1().graph2
{'a': ['b','c'],
 'b': ['a','c','d','e'],
 'c': ['a','b','d','e'],
 'd': ['b','c','e'],
 'e': ['b','c','d']}
>>> example1().test_isomophism()
True
>>> example1().find_orbits()
[('1', 'a'),
 ('2', {'b', 'c'}),
 ('3', {'b', 'c'}),
 ('4', {'d', 'e'}),
 ('5', {'d', 'e'})]
>>>
```

# Symmetry in graphs and automorfism

Symmetry in graphs can be divided into three types:
1. Simple symmetry. The graph has one axes of symmetry. The order of the automorphism group is $\ge2$.
2. k-fold symmetry. The graph has k axes of symmetry. The order of the automorphism group is $\ge2*k$.
3. Multiple symmetry. All vertices of the graph lie on the axes of symmetry. The order of the automorphism group is $\ge2*n$, where $n$ is the number of vertices of the graph.

![Symmetry types](./figure/Symmetry_types.png)

The `find_automorfism()` function can be used to find arbitary automorphism substitution for any symmetry type.

```python
>>> orb = example2().find_orbits()
>>> example2().find_automorfism(orb)
[('a', 'b', 'c', 'd', 'e', 'f'),
 ('f~', 'd~', 'e~', 'b~', 'c~', 'a~')]
>>>
```

# Basic concepts
## Graph virtual neural network

Graph virtual neural network (GVN) is a network focused on working with graphs. In networks of this type, the vertices of the graph are called neurons and the edges of the graph are called synaptic connections. This type of network can be one-way or two-way.

We illustrate the main idea of this design with the following example. 
Let's build two-way virtual neural network for grid 3x3. 
We choose (0,0) and (2,0) as the initial and final coordinates, respectively.
Such the network contains information about all possible paths from point (0,0) to point (2,0) in eight unit steps.
The construction is carried out on both sides, which eliminates the formation of dead-end paths.
For the grid we will have the complexity $O(|V|^{2.1})$.
In general, the complexity will not exceed $O(|V|^3)$.

```python
>>> Graph.twoway_network(example6().graph1, (0,0), (2,0))
[{(0, 0): [(1, 0), (0, 1)]},
 {(1, 0): [(1, 1)], (0, 1): [(1, 1), (0, 2)]},
 {(1, 1): [(2, 1), (0, 1), (1, 2), (1, 0)], (0, 2): [(1, 2), (0, 1)]},
 {(2, 1): [(1, 1), (2, 2)],
  (0, 1): [(1, 1), (0, 2)],
  (1, 2): [(2, 2), (0, 2), (1, 1)],
  (1, 0): [(1, 1)]},
 {(1, 1): [(2, 1), (0, 1), (1, 2), (1, 0)],
  (2, 2): [(2, 1), (1, 2)],
  (0, 2): [(0, 1), (1, 2)]},
 {(2, 1): [(1, 1), (2, 2)],
  (0, 1): [(1, 1)],
  (1, 2): [(1, 1), (2, 2)],
  (1, 0): [(1, 1)]},
 {(1, 1): [(1, 0), (2, 1)], (2, 2): [(2, 1)]},
 {(1, 0): [(2, 0)], (2, 1): [(2, 0)]}]
>>>
```

![Virtual neural network](./figure/Network.png)

One of the key concepts of GVN is the matrix indegrees and outdegrees. Each row in this matrix corresponds to its own vertex. Furthermore, its rows do not have a fixed association with indices.

```python
>>>
network = Graph.twoway_network(example6().graph1, (0,0), (2,0))
Graph.get_degree_matrix(example6().graph1,network)
{(0, 0): ([0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0]),
 (0, 1): ([0, 1, 0, 2, 0, 2, 0, 0, 0], [0, 2, 0, 2, 0, 1, 0, 0, 0]),
 (0, 2): ([0, 0, 1, 0, 2, 0, 0, 0, 0], [0, 0, 2, 0, 2, 0, 0, 0, 0]),
 (1, 0): ([0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0]),
 (1, 1): ([0, 0, 2, 0, 4, 0, 4, 0, 0], [0, 0, 4, 0, 4, 0, 2, 0, 0]),
 (1, 2): ([0, 0, 0, 2, 0, 3, 0, 0, 0], [0, 0, 0, 3, 0, 2, 0, 0, 0]),
 (2, 0): ([0, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
 (2, 1): ([0, 0, 0, 1, 0, 2, 0, 2, 0], [0, 0, 0, 2, 0, 2, 0, 1, 0]),
 (2, 2): ([0, 0, 0, 0, 2, 0, 2, 0, 0], [0, 0, 0, 0, 2, 0, 1, 0, 0])}
>>>
```

## Graph virtual neural network derivative

When constructing the derivative of GVN, non-zero elements of the outdegree matrix are used. This is done as follows:

1. For each layer, a list is created, consisting of the outdegrees of each vertex. Thus, a certain sequence of lists is obtained.
   
2. Each outdegree from the current list is associated with the outdegrees from the next list. For example, the entry $(2,[1,2])$ means that the vertex with the outdegree of 2 generates two vertices with outdegrees of 1 and 2.

The resulting network stores information about how the adjacent edges are connected to each other in the original network.

Two graphs are isomorphic if and only if the derivatives of their GVNs constructed from the same vertex coincide.

As an example, consider two one-way virtual neural networks and its derivative.

```python
>>> network1 = Graph.oneway_network(example7().graph1,'a',3)
>>> Graph.network_derivative(network1)
[[(2, [1, 2])],[(1, [2]), (2, [1, 2])]]
>>> network2 = Graph.oneway_network(example7().graph2,'i',3)
>>> Graph.network_derivative(network2)
[[(2, [1, 2])],[(1, [1]), (2, [2, 2])]]
>>>
```

![Virtual neural network derivative](./figure/Derivative.png)

## Minimal one-way graph virtual neural network

When constructing each layer in a network of this type, the edges traversed in all previous layers are not used. However, the same edge can be traversed in both directions in each layer. This is necessary in order to preserve information about the symmetry of the graph. It allows us to significantly speed up the testing of graphs for isomorphism and finding graph vertices orbits. The time complexity is $O(|V|+|E|)$.

```python
>>> Graph.minimal_oneway_network(cube().graph1,'a')
[{'a': ['b', 'd', 'e']},
 {'b': ['c', 'f'], 'd': ['c', 'h'], 'e': ['f', 'h']},
 {'c': ['g'], 'f': ['g'], 'h': ['g']}]
>>> cube().test_isomophism()
True
>>> iso = cube().find_orbits()
>>> cube().find_automorfism(iso)
[('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'),
('e~', 'f~', 'g~', 'h~', 'a~', 'b~', 'c~', 'd~')]
```

![Minimal network](./figure/Minimal_network.png)

# References

[1] Ronald C. Read, Derek G. Corneil, The Graph Isomorphism Disease, J. Graph Theory, vol. 1, 1977, pp. 339-363.

[2] Johannes Köbler, Uwe Schöning, Jacobo Torán, The Graph Isomorphism Problem: Its Structural Complexity, Springer Science+Business Media, LLC, 1993, 167 p.

[3] László Babai, Graph Isomorphism in Quasipolynomial Time, Preliminary verson, 2015, 84 p., arXiv: 1512.03547

[4] László Babai, Graph Isomorphism in Quasipolynomial Time, Version 2.5, 2018, 109 p., https://people.cs.uchicago.edu/~laci/quasi25.pdf

[5] Steven S. Skiena, The Algorithm Design Manual, Springer, 3nd ed, 2020, 800 p.

[6] Maxime Labonne, Hands-On Graph Neural Networks Using Python, 2023, 331 p., https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python
