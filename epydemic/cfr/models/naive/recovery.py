from functools import lru_cache

from cfr.models.base import BaseCFRModel


class RecoveryCFRModel(BaseCFRModel):
    name: str = "recovery"

    @lru_cache(maxsize=1)
    def estimate(self, t: int, start: int = 0) -> float:
        return 1 - (self.outbreak.cumulative_recoveries.iloc[t] / self.outbreak.cumulative_cases.iloc[t])