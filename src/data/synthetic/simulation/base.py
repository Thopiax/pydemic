from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt

from data.synthetic.seird import SEIRDModel


class Simulation(ABC):
    STABILITY_THRESHOLD = 1e-8

    def __init__(self, model: SEIRDModel):
        self.model = model

        self._observer = None
        self._previous_simulations = []

    def attach_observer(self, observer):
        self._observer = observer

    @staticmethod
    def aggregate_infection_compartments(simulation: pd.DataFrame):
        df = simulation.copy()

        infectious_compartments = df.columns[df.columns.str.startswith("I")]

        if len(infectious_compartments) == 1:
            print("Already aggregated.")
            return

        df["I"] = simulation[infectious_compartments].sum(axis=1)
        df.drop(columns=infectious_compartments, inplace=True)

        # rearange the columns
        df = df[["S", "E", "I", "R", "D"]]

        return df

    def _verify_stability(self, simulation: pd.DataFrame):
        # numerical stability: sum of all compartments should not deviate from N
        assert all(abs(simulation.sum(axis=1) - self.model.parameters["N"]) < Simulation.STABILITY_THRESHOLD)

    @abstractmethod
    def run(self, T: int, dt: float = 0.05, aggregate_I: bool = False):
        pass
