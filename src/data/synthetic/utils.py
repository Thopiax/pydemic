import networkx as nx

from enums import Outcome


def infectious_compartment_label(stream_index: int, outcome: Outcome, k: int):
    return f"I_({stream_index},{outcome.value},{k})"


def get_compartment(graph: nx.DiGraph, key: str):
    return graph.nodes[key]["val"]