import numpy as np
from functools import lru_cache
from typing import Type

from cfr.models.naive import RecoveryCFRModel
from cfr.models.base import BaseCFRModel
from outcome.distribution.exceptions import InvalidParameterError
from outcome.models.base import BaseOutcomeModel


class FatalityOutcomeModel(BaseOutcomeModel):
    name: str = "fatality"
    BoundingCFRModel: Type[BaseCFRModel] = RecoveryCFRModel

    @lru_cache(maxsize=8)
    def target(self, t: int, start: int = 0) -> np.array:
        # return all the days from start up to t inclusive
        return self.outbreak.cumulative_deaths.iloc[start:(t + 1)].values

    def predict(self, t: int, start: int = 0) -> np.array:
        self._verify_alpha(self.alpha, t, start)

        result = np.zeros(t - start)

        for k in range(start, t + 1):
            result[k - start] = self._predict_incidence(k)

        return self.alpha * result

    def _verify_alpha(self, alpha: float, t: int, start: int = 0) -> None:
        if alpha > self._cfr_bounds[t]:
            raise InvalidParameterError
