import numpy as np
from functools import lru_cache
from typing import Type

from outcome_lag.models.base import BaseOutcomeModel


class FatalityOutcomeModel(BaseOutcomeModel):
    name: str = "fatality"

    @lru_cache(maxsize=8)
    def target(self, t: int, start: int = 0) -> np.array:
        # return fatality incidence from start up to t inclusive
        return self.outbreak.deaths.iloc[start:(t + 1)].values

    def predict(self, t: int, start: int = 0) -> np.array:
        self._verify_alpha(self.alpha, t, start)

        result = np.zeros(t + 1 - start)

        for k in range(t + 1):
            result[k] = self._predict_incidence(k + start)

        return self.alpha * result

    def _verify_alpha(self, alpha: float, t: int, start: int = 0) -> None:
        return True
