from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd

from data.synthetic.simulation.base import Simulation
from data.synthetic.utils import infectious_compartment_label
from enums import Outcome


class SimulationObserver(ABC):
    columns = []

    def __init__(self, simulation: Simulation, dt: float = 1.0):
        self.simulation = simulation
        self.simulation.attach_observer(self)

        self.dt = dt

        self._t: float = 0

        self._observations = []

        self._state = None
        self._state_history = None

        self._reset_state()
        self._reset_observation()

        pass

    def _reset_observation(self):
        self._t = 0
        self._state_history = pd.DataFrame(columns=self.__class__.columns)

    def _reset_state(self):
        self._state = pd.Series(data=np.zeros(len(self.__class__.columns)), index=self.__class__.columns)

    def observe(self, t: int, state_diff: Dict[str, float]):
        # include right boundary
        if t > (self._t + self.dt):
            self._t += self.dt

            self._state_history.loc[self._t, :] = self._state
            self._reset_state()

        self._observe_state(state_diff)

    @property
    def latest_observation(self):
        if len(self._observations) == 0:
            return None

        return self._observations[-1]

    @abstractmethod
    def _observe_state(self, state_diff: Dict[str, float]):
        pass

    @abstractmethod
    def flush(self, T: int):
        pass


class PerfectObserver(SimulationObserver):
    columns = ["cases", "deaths", "recoveries", "ECR"]

    @staticmethod
    def _observe_cases(state_diff: Dict[str, float]):
        return max(- (state_diff["E"] + state_diff["S"]), 0)

    def _observe_state(self, state_diff: Dict[str, float]):
        cases = PerfectObserver._observe_cases(state_diff)

        self._state["cases"] += cases
        self._state["deaths"] += state_diff["D"]
        self._state["recoveries"] += state_diff["R"]

    def flush(self, T: int):
        self._observations.append(self._state_history.copy(deep=True))
        print("flushed state_history", self._observations)

        self._state_history = None

        self._reset_observation()
        self._reset_state()

