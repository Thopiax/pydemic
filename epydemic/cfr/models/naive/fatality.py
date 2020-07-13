from functools import lru_cache

from cfr.models.base import BaseCFRModel


class FatalityCFRModel(BaseCFRModel):
    name: str = "fatality"

    @lru_cache(maxsize=1)
    def estimate(self, t: int, start: int = 0) -> float:
        return self.outbreak.cumulative_deaths.iloc[t] / self.outbreak.cumulative_cases.iloc[t]