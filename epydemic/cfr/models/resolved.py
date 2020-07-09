import numpy as np

from .base import BaseCFRModel
from .naive import NaiveCFRModel


class ResolvedCFRModel(BaseCFRModel):
    name: str = "resolved"

    def estimate(self, t: int) -> float:
        super().estimate(t)

        resolved_cases = (self.outbreak.cumulative_deaths.iloc[t] + self.outbreak.cumulative_recoveries.iloc[t])

        # avoid divide by zero error
        if resolved_cases == 0:
            return np.nan

        return self.outbreak.cumulative_deaths.iloc[t] / resolved_cases


class WeightedResolvedCFRModel(BaseCFRModel):
    name: str = "weighted_resolved"

    def estimate(self, t: int = -1) -> float:
        nCFR_estimate = NaiveCFRModel(self.outbreak).estimate(t)
        rCFR_estimate = ResolvedCFRModel(self.outbreak).estimate(t)

        resolved_rate = self.outbreak.resolved_case_rate.iloc[t]

        return nCFR_estimate * (1 - resolved_rate) + rCFR_estimate * resolved_rate
