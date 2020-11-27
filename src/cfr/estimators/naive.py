from functools import lru_cache

from src.cfr.estimators.base import BaseCFREstimator


class NaiveCFREstimator(BaseCFREstimator):
    name: str = "naive"

    @lru_cache(maxsize=32)
    def estimate(self, t: int, start: int = 0) -> float:
        self._verify_inputs(t, start)

        cumulative_deaths = self.outbreak.cumulative_deaths.iloc[t] - self.outbreak.cumulative_deaths.iloc[start]
        cumulative_cases = self.outbreak.cumulative_cases.iloc[t] - self.outbreak.cumulative_cases.iloc[start]

        return cumulative_deaths / cumulative_cases


class NaiveComplementCFREstimator(BaseCFREstimator):
    name: str = "naive_complement"

    @lru_cache(maxsize=32)
    def estimate(self, t: int, start: int = 0) -> float:
        self._verify_inputs(t, start)

        cumulative_recoveries = self.outbreak.cumulative_recoveries.iloc[t] - self.outbreak.cumulative_recoveries.iloc[start]
        cumulative_cases = self.outbreak.cumulative_cases.iloc[t] - self.outbreak.cumulative_cases.iloc[start]

        return 1 - (cumulative_recoveries / cumulative_cases)
