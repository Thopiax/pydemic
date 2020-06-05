from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from epydemic.outbreak import Outbreak, OutbreakTimeWindow


class AbstractCFREstimator(ABC):
    def __init__(self, outbreak: Outbreak):
        self.outbreak = outbreak

    def window_estimates(self, **kwargs):
        estimates = pd.Series()

        outbreak = self.outbreak

        for otw in self.outbreak.expanding_windows(**kwargs):
            t = otw.end
            self.outbreak = otw

            if len(otw) == 0:
                # if the time window is empty we shouldn't consider it.
                continue

            estimates.loc[t] = self.estimate()

        self.outbreak = outbreak

        return estimates

    @abstractmethod
    def estimate(self, t: int = -1) -> float:
        # if period of analysis is in the future, return NaN.
        if t > len(self.outbreak):
            return np.nan()

        pass