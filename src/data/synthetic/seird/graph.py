from functools import cached_property
from typing import List, Dict, Tuple

import networkx as nx
import numpy as np

from data.synthetic.seird.utils import get_compartment


class SEIRDGraph:
    def __init__(self, graph: nx.DiGraph, N: int, **kwargs):
        self._graph = graph

        self.N = N
        self.set_initial_state(**kwargs)

    def __getitem__(self, item):
        if item in self.compartments:
            return get_compartment(self._graph, item)

        return getattr(self, item)

    def set_initial_state(self, E_0: float = 100.0):
        self._graph.add_nodes_from([
            ("S", dict(val=self.N - E_0)),
            ("E", dict(val=E_0)),
            *self.infected_compartments,
            "R",
            "D",
        ], val=0.0)

    @cached_property
    def infected_compartments(self):
        return list(filter(lambda s: s.startswith("I"), self.compartments))

    @cached_property
    def compartments(self) -> List[str]:
        return [node for node in self._graph.nodes]

    @property
    def transition_rates(self) -> Dict[str, List[Tuple[str, float]]]:
        rates = {}

        for (src, dest, data) in self._graph.edges.data():
            payload = (dest, data["rate_func"](self._graph))

            if src in rates:
                rates[src].append(payload)
            else:
                rates[src] = [payload]

        return rates

    @property
    def state(self) -> Dict[str, float]:
        return {comp: get_compartment(self._graph, comp) for comp in self._graph.nodes}

    def update_state(self, state: Dict[str, float]):
        for node, data in self._graph.nodes.data():
            data["val"] += state[node]

    def set_state_from_vector(self, state_vector: np.array):
        for index, (_, data) in enumerate(self._graph.nodes.data()):
            # replace value with state vector value
            data["val"] = state_vector[index]