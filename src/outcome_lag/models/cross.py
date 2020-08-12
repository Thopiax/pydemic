import numpy as np
from functools import lru_cache

from outcome_lag.models.base import BaseOutcomeModel


class CrossOutcomeModel(BaseOutcomeModel):
    name: str = "cross"

    @lru_cache(maxsize=8)
    def target(self, t: int, start: int = 0) -> np.array:
        # return fatality incidence from start up to t inclusive
        return self.outbreak.deaths.iloc[start:(t + 1)].values

    def predict(self, t: int, start: int = 0) -> np.array:
        result = np.zeros(t + 1 - start)

        for k in range(t + 1):
            result[k] = self._predict_incidence(k + start)

        return self.alpha * result

