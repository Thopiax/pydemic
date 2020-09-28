from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt

from data.synthetic.seird import SEIRDModel


class Simulator(ABC):
    STABILITY_THRESHOLD = 1e-8

    def __init__(self, graph_model: SEIRDModel):
        self.graph_model = graph_model

        self._previous_simulations = []

    @staticmethod
    def aggregate_infection_compartments(simulation: pd.DataFrame):
        df = simulation.copy()

        infected_compartments = df.columns[df.columns.str.startswith("I")]

        if len(infected_compartments) == 1:
            print("Already aggregated.")
            return

        df["I"] = simulation[infected_compartments].sum(axis=1)
        df.drop(columns=infected_compartments, inplace=True)

        # rearange the columns
        df = df[["S", "E", "I", "R", "D"]]

        return df

    def _verify_stability(self, simulation: pd.DataFrame):
        # numerical stability: sum of all compartments should not deviate from N
        assert all(abs(simulation.sum(axis=1) - self.graph_model.N) < Simulator.STABILITY_THRESHOLD)

    @abstractmethod
    def simulate(self, T: int, dt: float = 0.05, aggregate_I: bool = False):
        pass
