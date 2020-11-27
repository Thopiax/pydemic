from functools import lru_cache

from src.cfr.estimators.base import BaseCFREstimator


class OutcomeLagAdjustedCFREstimator(BaseCFREstimator):
    name: str = "OLA"

    @lru_cache(maxsize=32)
    def estimate(self, t: int, start: int = 0) -> float:
        self._verify_inputs(t, start)

        return -1.0

