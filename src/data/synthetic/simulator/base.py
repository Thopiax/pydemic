from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt

from data.synthetic.seird import SEIRDGraph


class Simulator(ABC):
    STABILITY_THRESHOLD = 1e-8

    def __init__(self, model: SEIRDGraph):
        self.model = model

        self._previous_simulations = []

    def aggregate_infection_compartments(self, simulation: pd.DataFrame):
        df = simulation.copy()

        df["I"] = simulation[self.model.infected_compartments].sum(axis=1)
        df.drop(columns=self.model.infected_compartments, inplace=True)

        # rearange the columns
        df = df[["S", "E", "I", "R", "D"]]

        return df

    def plot(self, simulation: Optional[pd.DataFrame] = None):
        if simulation is None:
            simulation = self._previous_simulations[-1]

        ax = plt.gca()

        simulation.plot(ax=ax)

        plt.show()

    def _verify_stability(self, simulation: pd.DataFrame):
        # numerical stability: sum of all compartments should not deviate from N
        assert all(abs(simulation.sum(axis=1) - self.model.N) < Simulator.STABILITY_THRESHOLD)

    @abstractmethod
    def simulate(self, T: int, dt: float = 0.05):
        pass