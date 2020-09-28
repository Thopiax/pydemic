from functools import cached_property, partial
from typing import Union, Collection

import networkx as nx
import numpy as np

from .model import SEIRDModel
from data.synthetic.utils import get_compartment, infectious_compartment_label
from src.enums import Outcome


class InfectiousStream:
    def __init__(self, index: int, p: float, alpha: float, c: float = 1.0, beta: float = 0.5,
                 K_D: int = 1, rate_D: float = 1 / 5.0, K_R: int = 1, rate_R: float = 1 / 5.0):
        self.index = index

        assert 0 <= p <= 1
        self.p = p
        self.alpha = alpha

        self.c = c
        self.beta = beta

        assert K_D >= 1
        self.K_D = K_D
        self.rate_D = rate_D

        assert K_R >= 1
        self.K_R = K_R
        self.rate_R = rate_R

    def update_parameters(self, **params):
        for key, value in params.items():
            if getattr(self, key, None) is None:
                print(f"Parameter {key} does not exist.")
            else:
                setattr(self, key, value)

    @property
    def parameters(self):
        return dict(p=self.p, alpha=self.alpha, c=self.c, beta=self.beta, chain_parameters=self.chain_parameters)

    @property
    def chain_parameters(self):
        return {
            Outcome.DEATH: {
                "p": self.alpha * self.p,
                "K": self.K_D,
                "rate": self.rate_D
            }, Outcome.RECOVERY: {
                "p": (1 - self.alpha) * self.p,
                "K": self.K_R,
                "rate": self.rate_R
            }
        }

    @property
    def compartments(self):
        return self.outcome_compartments(Outcome.DEATH) + self.outcome_compartments(Outcome.RECOVERY)

    def _compartment_label(self, outcome: Outcome, k: int):
        return infectious_compartment_label(self.index, outcome, k)

    def outcome_compartments(self, outcome: Outcome):
        return [self._compartment_label(outcome, k + 1) for k in range(self.chain_parameters[outcome]["K"])]


class SEIRDBuilder:
    def __init__(self, alpha: float, D_E: float = 0.2, N: int = 1_000_000):
        # # if only one number is passed,
        # if type(stream_probabilities) is not list:
        #     # we assume that the system has binary infection streams
        #     stream_probabilities = [stream_probabilities, 1 - stream_probabilities]
        #
        # stream_probabilities = np.array(stream_probabilities)
        #
        # assert sum(abs(stream_probabilities)) == 1

        self.D_E = D_E
        self._alpha = alpha
        self.N = N

        self._stream_probabilities = 0
        self._streams = []

    def with_stream(self, probability: float = 1.0, **kwargs):
        assert probability + self._stream_probabilities <= 1

        index = len(self._streams)
        alpha = kwargs.get("alpha", self._alpha)

        stream = InfectiousStream(index, probability, alpha, **kwargs)

        self._streams.append(stream)
        self._stream_probabilities += probability

        return self

    def build(self, **kwargs):
        assert self._stream_probabilities == 1

        graph = nx.DiGraph()

        self._build_graph_edges(graph)

        return SEIRDModel(graph, parameters=self.parameters, **kwargs)

    @cached_property
    def parameters(self):
        return dict(
            N=self.N,
            alpha=self._alpha,
            D_E=self.D_E,
            streams=[s.parameters for s in self._streams]
        )

    def _exposure_rate(self, N: int, graph: nx.DiGraph):
        result = 0

        for stream in self._streams:
            population = sum(get_compartment(graph, comp) for comp in stream.compartments)

            result += (population * stream.c * stream.beta)

        return result / N

    def _infectiousness_rate(self, stream: InfectiousStream, outcome: Outcome, _graph: nx.DiGraph):
        return stream.chain_parameters[outcome]["p"] / self.D_E

    @staticmethod
    def _outcome_rate(stream: InfectiousStream, outcome: Outcome, _graph: nx.DiGraph):
        return stream.chain_parameters[outcome]["rate"]

    def _build_graph_edges(self, g: nx.DiGraph):
        g.add_edge("S", "E", rate_func=partial(self._exposure_rate, self.N))

        for stream in self._streams:
            for outcome in Outcome:
                compartment_labels = stream.outcome_compartments(outcome)

                # add connection from E to stream
                g.add_edge(
                    "E",
                    compartment_labels[0],
                    rate_func=partial(self._infectiousness_rate, stream, outcome)
                )

                # add middle compartments
                for k in range(1, stream.chain_parameters[outcome]["K"]):
                    g.add_edge(
                        compartment_labels[k - 1],
                        compartment_labels[k],
                        rate_func=partial(self._outcome_rate, stream, outcome)
                    )

                # add connection from stream to outcome
                g.add_edge(
                    compartment_labels[-1],
                    outcome.value,
                    rate_func=partial(self._outcome_rate, stream, outcome)
                )


