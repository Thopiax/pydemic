import numpy as np
from functools import lru_cache
from typing import Type

from cfr.models.naive import FatalityCFRModel
from epydemic.cfr.models.base import BaseCFRModel
from outcome.distribution.exceptions import InvalidParameterError
from outcome.models.base import BaseOutcomeModel


class RecoveryOutcomeModel(BaseOutcomeModel):
    name: str = "recovery"
    BoundingCFRModel: Type[BaseCFRModel] = FatalityCFRModel

    @lru_cache(maxsize=8)
    def target(self, t: int, start: int = 0) -> np.array:
        # return recovery incidence from start up to t inclusive
        return self.outbreak.recoveries.iloc[start:(t + 1)]

    def predict(self, t: int, start: int = 0) -> np.array:
        self._verify_alpha(self.alpha, t, start)

        result = np.zeros((t + 1) - start)

        for k in range(t + 1):
            result[k] = self._predict_incidence(k + start)

        return (1 - self.alpha) * result

    def _verify_alpha(self, alpha: float, t: int, start: int = 0) -> None:
        return True
