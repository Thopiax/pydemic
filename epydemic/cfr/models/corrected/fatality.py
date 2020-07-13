from functools import lru_cache

from cfr.models.base import BaseCFRModel
from outcome.distribution.discrete import NegBinomialOutcomeDistribution
from outcome.models.fatality import FatalityOutcomeModel


class FatalityCorrectedCFRModel(BaseCFRModel):
    name: str = "fatality_corrected"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.outcome_model = FatalityOutcomeModel(self.outbreak, NegBinomialOutcomeDistribution())

    @lru_cache(maxsize=32)
    def estimate(self, t: int, start: int = 0) -> float:
        # TODO: fix -1 problem in early estimates
        self.outcome_model.fit(t, start=start)
        print("estimated fatality_corrected", self.outbreak.region, t, self.outcome_model.alpha)

        return self.outcome_model.alpha
