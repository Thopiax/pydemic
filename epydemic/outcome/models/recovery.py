from functools import lru_cache

from outcome.models.base import BaseOutcomeModel
from outcome.models.exceptions import TrivialTargetError
from outcome.optimizer.loss import SquaredErrorLoss
from outcome.optimizer.main import OutcomeOptimizer


class RecoveryOutcomeModel(BaseOutcomeModel):
    name: str = "recovery"

    @lru_cache
    def target(self, t: int, start: int = 0) -> int:
        return self.outbreak.cumulative_recoveries.values[start:t]

    def fit(self, t: int, start: int = 0, verbose: bool = False, random_state: int = 1, **kwargs) -> None:
        if self.target(t, start) == 0:
            raise TrivialTargetError

        optimizer = OutcomeOptimizer(self, verbose=verbose, random_state=random_state)

        loss = SquaredErrorLoss(self, t, start)
        optimization_result = optimizer.optimize(loss, **kwargs)

        return optimization_result

