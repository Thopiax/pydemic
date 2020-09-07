import networkx as nx
from typing import Optional, Dict, List, Tuple

from functools import partial

import numpy as np

from data.synthetic.sei4rd.parameters import OutcomeParameters, InfectionParameters
from outcome import Outcome


class SEI4RD:
    def __init__(self, N: int, alpha: float,
                 infection_parameters: Optional[InfectionParameters] = None,
                 death_parameters: Optional[OutcomeParameters] = None,
                 recovery_parameters: Optional[OutcomeParameters] = None):
        self.N = N

        assert 0 < alpha < 1
        self.alpha = alpha

        self._parameters = {
            "infection": infection_parameters or InfectionParameters(),
            Outcome.DEATH: death_parameters or OutcomeParameters(),
            Outcome.RECOVERY: recovery_parameters or OutcomeParameters()
        }

        self.infected_compartments = [
            "IA_D",
            "IA_R",
            *self._all_IS_labels(Outcome.DEATH),
            *self._all_IS_labels(Outcome.RECOVERY)
        ]

        self._outcome_probabilities = {
            Outcome.DEATH: self.alpha,
            Outcome.RECOVERY: 1 - self.alpha
        }

        self._graph = None

    def _E_in_rate(self):
        I = sum(self.get_compartment(comp) for comp in self.infected_compartments)

        return (I * self._parameters["infection"].c * self._parameters["infection"].beta) / self.N

    def _IA_label(self, outcome: Outcome):
        return f"IA_{outcome.value}"

    def _IA_in_rate(self, outcome: Outcome):
        p = self._outcome_probabilities[outcome]

        return (p * (1 - self._parameters[outcome].gamma)) / self._parameters["infection"].D_E

    def _IA_out_rate(self, outcome: Outcome):
        return self._parameters[outcome].lambdaA

    def _IS_label(self, outcome: Outcome, k: int):
        return f"IS_{outcome.value}{k}"

    def _all_IS_labels(self, outcome: Outcome):
        return [self._IS_label(outcome, k) for k in range(1, self._parameters[outcome].K + 1)]

    def _IS_in_rate(self, outcome: Outcome):
        p = self._outcome_probabilities[outcome]

        return (p * self._parameters[outcome].gamma) / self._parameters["infection"].D_E

    def _IS_out_rate(self, outcome: str):
        return self._parameters[outcome].lambdaS

    def init_graph(self, **kwargs):
        self._graph = nx.DiGraph()

        self._build_graph_nodes(self._graph, **kwargs)
        self._build_graph_edges(self._graph)

    def _build_graph_nodes(self, g: nx.DiGraph, E_0: float = 10.0):
        g.add_nodes_from([
            ("S", dict(val=self.N - E_0)),
            ("E", dict(val=E_0)),
            *self.infected_compartments,
            "R",
            "D",
        ], val=0.0)

    def _build_graph_edges(self, g: nx.DiGraph):
        g.add_edge("S", "E", rate_func=self._E_in_rate)

        for outcome in Outcome:
            # Add IA edges
            g.add_edge("E", self._IA_label(outcome), rate_func=partial(self._IA_in_rate, outcome))
            g.add_edge(self._IA_label(outcome), outcome.value, rate_func=partial(self._IA_out_rate, outcome))

            # Add IS edges
            g.add_edge("E", self._IS_label(outcome, 1), rate_func=partial(self._IS_in_rate, outcome))

            K = self._parameters[outcome].K
            for k in range(1, K):
                g.add_edge(
                    self._IS_label(outcome, k),
                    self._IS_label(outcome, k + 1),
                    rate_func=partial(self._IS_out_rate, outcome)
                )

            # Add final IS edge to outcome
            g.add_edge(self._IS_label(outcome, K), outcome.value, rate_func=partial(self._IS_out_rate, outcome))

    @property
    def compartments(self) -> List[str]:
        assert self._graph is not None

        return [node for node in self._graph.nodes]

    def get_compartment(self, key: str):
        assert self._graph is not None

        return self._graph.nodes[key]["val"]

    @property
    def transition_rates(self) -> Dict[str, List[Tuple[str, float]]]:
        assert self._graph is not None

        rates = {}

        for (src, dest, data) in self._graph.edges.data():
            payload = (dest, data["rate_func"]())

            if src in rates:
                rates[src].append(payload)
            else:
                rates[src] = [payload]

        return rates

    @property
    def state(self) -> Dict[str, float]:
        assert self._graph is not None

        return {comp: self.get_compartment(comp) for comp in self._graph.nodes}

    def update_state(self, state: Dict[str, float]):
        assert self._graph is not None

        for node, data in self._graph.nodes.data():
            data["val"] += state[node]

    def set_state_from_vector(self, state_vector: np.array):
        assert self._graph is not None

        for index, (_, data) in enumerate(self._graph.nodes.data()):
            # replace value with state vector value
            data["val"] = state_vector[index]

