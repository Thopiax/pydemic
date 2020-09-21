import networkx as nx


def get_compartment(graph: nx.DiGraph, key: str):
    return graph.nodes[key]["val"]