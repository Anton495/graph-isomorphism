import argparse
from collections import defaultdict
import random

from graph import Graph


def _create_random_graph(n, p):
    graph = defaultdict(list)
    for i in range(n):
        vi = f"v{i}"
        for j in range(i+1, n):
            vj = f"v{j}"
            if random.random() < p:
                graph[vi].append(vj)
                graph[vj].append(vi)
    return {v: n for v, n in graph.items()}



def _get_rename_from_isomorphism(iso):
    rename = {}
    seen = set()
    for v, w in iso:
        if isinstance(w, set) or isinstance(w, list):
            not_used = [u for u in sorted(w) if not u in seen]
            w = not_used[0]
        rename[v] = w
        seen.add(w)
    return rename


def _rename_vertices(graph, rename):
    return {rename[v]: [rename[w] for w in neighbours] for v, neighbours in graph.items()}


def _permute_graph(graph):
    vertices = sorted(list(graph.keys()))
    permuted = random.sample(vertices, len(vertices))
    rename = {v: p for v, p in zip(vertices, permuted)}
    return _rename_vertices(graph, rename)


def test_isomorphic_random_graphs(n, p):
    graph = _create_random_graph(n, p)
    permuted = _permute_graph(graph)
    pair = Graph(graph, permuted)
    iso = pair.find_isomophism(rev=True)
    if iso is False:
        raise ValueError("test_isomorphic_random_graphs failed: not found isomorphism")
    rename = _get_rename_from_isomorphism(iso)
    if _rename_vertices(graph, rename) != permuted:
        raise ValueError("test_isomorphic_random_graphs failed: incorrect isomorphism")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test isomorphism search for random graphs")
    parser.add_argument("-n", "--vertices", help="number of vertices", type=int, default=30)
    parser.add_argument("-p", "--probability", help="edge probability", type=float, default=0.5)
    parser.add_argument("-s", "--seed", help="random seed", type=int)
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    test_isomorphic_random_graphs(args.vertices, args.probability)
    print("ok!")
