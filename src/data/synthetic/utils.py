import pandas as pd
import networkx as nx

from typing import Collection

from enums import Outcome


def infectious_compartment_label(stream_index: int, outcome: Outcome, k: int):
    return f"I_({stream_index},{outcome.value},{k})"


def average_dfs(dfs: Collection[pd.DataFrame]):
    assert len(dfs) > 1
    result = dfs[0]

    for sim in dfs[1:]:
        result.add(sim, axis=1)

    result.div(float(len(dfs)))

    return result


def get_compartment(graph: nx.DiGraph, key: str):
    return graph.nodes[key]["val"]