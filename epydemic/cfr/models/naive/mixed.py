from functools import lru_cache

import numpy as np

from cfr.models.base import BaseCFRModel
from cfr.models.naive import FatalityCFRModel


class MixedCFRModel(BaseCFRModel):
    name: str = "mixed"

    @lru_cache(maxsize=10)
    def estimate(self, t: int, start: int = 0)-> float:
        resolved_cases = (self.outbreak.cumulative_deaths.iloc[t] + self.outbreak.cumulative_recoveries.iloc[t])

        # avoid divide by zero error
        if resolved_cases == 0:
            return np.nan

        return self.outbreak.cumulative_deaths.iloc[t] / resolved_cases
