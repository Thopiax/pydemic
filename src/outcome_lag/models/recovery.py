import numpy as np
from functools import lru_cache

from outcome_lag.models.base import BaseOutcomeModel


class RecoveryOutcomeModel(BaseOutcomeModel):
    name: str = "recovery"

    @lru_cache(maxsize=8)
    def target(self, t: int, start: int = 0) -> np.array:
        # return recovery incidence from start up to t inclusive
        return self.outbreak.recoveries.iloc[start:(t + 1)]

    def predict(self, t: int, start: int = 0) -> np.array:
        result = np.zeros((t + 1) - start)

        for k in range(t + 1):
            result[k] = self._predict_incidence(k + start)

        return (1 - self.alpha) * result

