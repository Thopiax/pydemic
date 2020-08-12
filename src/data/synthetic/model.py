from collections import namedtuple
import networkx as nx
from typing import Optional, Dict

import numpy as np
from scipy.integrate import solve_ivp

OutcomeParameters = namedtuple("OutcomeParameters", ["gamma", "K", "lambdaS", "lambdaA"],
                               defaults=[1.0, 3, 1 / 5.0, 1 / 5.0])

SEParameters = namedtuple("SEParameters", ["c_I", "beta_I", "c_E", "beta_E", "D_E"],
                          defaults=[1.0, 0.5, 1.0, 0.5, 5])


def build_IS_labels(O, K):
    return [f"IS_{O}{k}" for k in range(1, K + 1)]


def get_compartment_value(g: nx.DiGraph, key: str):
    return g.nodes[key]["val"]


class SEI4RD:
    def __init__(self, N: int, alpha: float,
                 se_parameters: Optional[SEParameters] = None,
                 death_parameters: Optional[OutcomeParameters] = None,
                 recovery_parameters: Optional[OutcomeParameters] = None):
        self.N = N

        assert 0 < alpha < 1
        self.alpha = alpha

        self._parameters = {
            "SE": se_parameters or SEParameters(),
            "D": death_parameters or OutcomeParameters(),
            "R": recovery_parameters or OutcomeParameters()
        }

        self.infected_compartments = [
            "IA_D",
            "IA_R",
            *build_IS_labels("D", self._parameters["D"].K),
            *build_IS_labels("R", self._parameters["R"].K)
        ]

        self._outcome_probabilities = {
            "D": self.alpha,
            "R": 1 - self.alpha
        }

        self.graph = None

    def _E_in_rate(self, g: nx.DiGraph):
        E = get_compartment_value(g, "E")
        I = sum(get_compartment_value(g, comp) for comp in self.infected_compartments)

        I_rate = (I * self._parameters["SE"].c_I * self._parameters["SE"].beta_I) / self.N
        E_rate = (E * self._parameters["SE"].c_E * self._parameters["SE"].beta_E) / self.N

        return I_rate + E_rate

    def _IA_in_rate(self, outcome: str):
        p = self._outcome_probabilities[outcome]

        return (p * (1 - self._parameters[outcome].gamma)) / self._parameters["SE"].D_E

    def _IA_out_rate(self, outcome: str):
        return self._parameters[outcome].lambdaA

    def _IS_in_rate(self, outcome: str):
        p = self._outcome_probabilities[outcome]

        return (p * self._parameters[outcome].gamma) / self._parameters["SE"].D_E

    def _IS_out_rate(self, outcome: str):
        return self._parameters[outcome].lambdaS

    def build_graph(self, **kwargs):
        g = nx.DiGraph()

        self._build_graph_nodes(g, **kwargs)
        self._build_graph_edges(g)

        return g

    def _build_graph_nodes(self, g: nx.DiGraph, E_0: int = 10):
        g.add_nodes_from([
            ("S", dict(val=self.N - E_0)),
            ("E", dict(val=E_0)),
            *self.infected_compartments,
            "R",
            "D",
        ], val=0.0)

    def _build_graph_edges(self, g: nx.DiGraph):
        g.add_edge("S", "E", weight_func=self._E_in_rate)

        for outcome in ["R", "D"]:
            # Add IA edges
            g.add_edge("E", f"IA_{outcome}", weight=self._IA_in_rate(outcome))
            g.add_edge(f"IA_{outcome}", outcome, weight=self._IA_out_rate(outcome))

            # Add IS edges
            g.add_edge("E", f"IS_{outcome}1", weight=self._IS_in_rate(outcome))
            for k in range(1, self._parameters[outcome].K):
                g.add_edge(f"IS_{outcome}{k}", f"IS_{outcome}{k + 1}", weight=self._IS_out_rate(outcome))

            g.add_edge(f"IS_{outcome}{self._parameters[outcome].K}", outcome, weight=self._IS_out_rate(outcome))
