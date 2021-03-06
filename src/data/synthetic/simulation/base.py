from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt

from data.synthetic.seird import SEIRDModel
from .observer import PerfectObserver


class Simulation(ABC):
    STABILITY_THRESHOLD = 1e-8

    def __init__(self, model: SEIRDModel, with_observer: bool = True):
        self.model = model

        self._observer = PerfectObserver(simulation=self) if with_observer else None
        self._previous_simulations = []

    def attach_observer(self, observer):
        self._observer = observer

    @property
    def observations(self):
        observation_keys = [f"obs_{i}" for i in range(len(self._observer._observations))]

        return pd.concat(self._observer._observations, axis=1, keys=observation_keys)

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
