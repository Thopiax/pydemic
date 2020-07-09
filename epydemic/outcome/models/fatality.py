from functools import lru_cache

from outcome.models.base import BaseOutcomeModel
from outcome.models.exceptions import TrivialTargetError
from outcome.optimizer.loss import SquaredErrorLoss
from outcome.optimizer.main import OutcomeOptimizer


class FatalityOutcomeModel(BaseOutcomeModel):
    name: str = "fatality"

    @lru_cache
    def target(self, t: int, start: int = 0) -> int:
        return self.outbreak.cumulative_deaths.loc[t] - self.outbreak.cumulative_deaths.loc[start]

    def alpha(self, t: int, start: int = 0) -> float:
        return self.target(t, start) / self.predict(t, start)
