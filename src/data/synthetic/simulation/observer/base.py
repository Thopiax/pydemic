from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd


class SimulationObserver(ABC):
    columns = []

    def __init__(self, simulation, dt: float = 1.0):
        self.simulation = simulation
        self.simulation.attach_observer(self)

        self.dt = dt

        self._t: float = 0

        self._observations = []

        self._state = None
        self._state_history = None

        self._reset_observation()

        pass

    def _reset_observation(self):
        self._t = 0

        self._state_history = pd.DataFrame(columns=self.__class__.columns)
        self._reset_state()
        # # set initial state
        # self._state_history.loc[0, :] = np.zeros(len(self.__class__.columns))

    def _reset_state(self):
        self._state = pd.Series(data=np.zeros(len(self.__class__.columns)), index=self.__class__.columns)

    def _append_state(self):
        self._state_history.loc[self._t, :] = self._state
        self._reset_state()

    def observe(self, t: int, state_diff: Dict[str, float]):
        # include right boundary
        if t > (self._t + self.dt):
            self._append_state()

            self._t += self.dt

        self._observe_state(state_diff)

    @property
    def latest_observation(self):
        if len(self._observations) == 0:
            return None

        return self._observations[-1]

    @abstractmethod
    def _observe_state(self, state_diff: Dict[str, float]):
        pass

    def flush(self, T: int):
        if self._t < T:
            self._append_state()

        self._observations.append(self._state_history.copy(deep=True))

        self._reset_observation()
