import numpy as np

from .base import AbstractCFREstimator
from .naive import NaiveCFREstimator


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