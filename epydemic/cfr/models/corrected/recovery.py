from abc import ABC
from functools import lru_cache

from cfr.models.base import BaseCFRModel
from outcome.distribution.discrete import NegBinomialOutcomeDistribution
from outcome.models.recovery import RecoveryOutcomeModel


class RecoveryCorrectedCFRModel(BaseCFRModel, ABC):
    name: str = "recovery_corrected"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.outcome_model = RecoveryOutcomeModel(self.outbreak, NegBinomialOutcomeDistribution())

    @lru_cache(maxsize=32)
    def estimate(self, t: int, start: int = 0) -> float:
        self.outcome_model.fit(t, start=start)
        print("estimated recovery_corrected", self.outbreak.region, t, self.outcome_model.alpha)

        return self.outcome_model.alpha
