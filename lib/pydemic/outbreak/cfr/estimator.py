
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class AbstractCFREstimator(ABC):
    def __init__(self, outbreak):
        self.outbreak = outbreak

    def range_estimate(self, **kwargs):
        estimates = pd.Series()

        for t in self.outbreak.range(**kwargs):
            estimates.loc[t] = self.estimate(t)

        return estimates

    @abstractmethod
    def estimate(self, t : int = -1) -> float:
        # if period of analysis is in the future, return NaN.
        if t > self.outbreak.duration:
            return np.nan()

        pass


class NaiveCFREstimator(AbstractCFREstimator):
    def __repr__(self):
        return "nCFR"

    def estimate(self, t : int = -1) -> float:
        super().estimate(t)

        return self.outbreak.cumulative_deaths.iloc[t] / self.outbreak.cumulative_cases.iloc[t]


class ResolvedCFREstimator(AbstractCFREstimator):
    def __repr__(self):
        return "rCFR"

    def estimate(self, t : int = -1) -> float:
        super().estimate(t)

        assert self.outbreak.recoveries is not None

        resolved_cases = (self.outbreak.cumulative_deaths.iloc[t] + self.outbreak.cumulative_recoveries.iloc[t])

        # avoid divide by zero error
        if resolved_cases == 0:
            return np.nan

        return self.outbreak.cumulative_deaths.iloc[t] / resolved_cases


class WeightedResolvedCFREstimator(AbstractCFREstimator):
    def __repr__(self):
        return "wrCFR"

    def estimate(self, t : int = -1) -> float:
        nCFR_estimate = NaiveCFREstimator(self.outbreak).estimate(t)
        rCFR_estimate = ResolvedCFREstimator(self.outbreak).estimate(t)

        resolved_rate = self.outbreak.resolved_case_rate.iloc[t]

        return nCFR_estimate * (1 - resolved_rate) + rCFR_estimate * resolved_rate

class FatalityDensityCFREstimator(AbstractCFREstimator):
    # source: Early Epidemiological Assessment of the Virulence of Emerging Infectious Diseases
    def __repr__(self):
        return "fdCFR"

    def __init__(self, outbreak, fatality_density):
        super().__init__(outbreak)
        self.fatality_density = fatality_density

    def _underestimation_coefficient(self, t):
        return [self.outbreak.cases[i - j] * self.fatality_density[j] for i in range(t) for j in range(i + 1)]

    def estimate(self, t : int = -1) -> float:
        super().estimate(t)

        return self.outbreak.cumulative_deaths.iloc[t] / self._underestimation_coefficient(t)
