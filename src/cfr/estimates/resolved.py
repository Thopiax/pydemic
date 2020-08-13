from functools import lru_cache

import numpy as np

from src.cfr.estimates.base import BaseCFREstimate


class ResolvedCFREstimate(BaseCFREstimate):
    name: str = "resolved"

    @lru_cache(maxsize=32)
    def estimate(self, t: int, start: int = 0) -> float:
        self._verify_inputs(t, start)

        cumulative_deaths = self.outbreak.cumulative_deaths.iloc[t] - self.outbreak.cumulative_deaths.iloc[start]
        cumulative_resolved_cases = \
            self.outbreak.cumulative_resolved_cases.iloc[t] - self.outbreak.cumulative_resolved_cases.iloc[start]

        return cumulative_deaths / cumulative_resolved_cases


class ResolvedComplementCFREstimate(BaseCFREstimate):
    name: str = "resolved_complement"

    @lru_cache(maxsize=32)
    def estimate(self, t: int, start: int = 0) -> float:
        self._verify_inputs(t, start)

        cumulative_recoveries = self.outbreak.cumulative_recoveries.iloc[t] - self.outbreak.cumulative_recoveries.iloc[start]
        cumulative_resolved_cases = \
            self.outbreak.cumulative_resolved_cases.iloc[t] - self.outbreak.cumulative_resolved_cases.iloc[start]

        return 1 - (cumulative_recoveries / cumulative_resolved_cases)
