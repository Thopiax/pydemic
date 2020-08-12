from functools import lru_cache

from src.cfr.estimates.base import BaseCFREstimate


class OutcomeLagAdjustedCFREstimate(BaseCFREstimate):
    name: str = "OLA"

    @lru_cache(maxsize=32)
    def estimate(self, t: int, start: int = 0) -> float:
        self._verify_inputs(t, start)

