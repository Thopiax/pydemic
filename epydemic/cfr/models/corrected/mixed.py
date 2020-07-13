from functools import lru_cache

from cfr.models.corrected import RecoveryCorrectedCFRModel, FatalityCorrectedCFRModel
from cfr.models.base import BaseCFRModel
from cfr.models.naive import FatalityCFRModel


class MixedCorrectedCFRModel(BaseCFRModel):
    name: str = "mixed_corrected"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fatality_model = FatalityCorrectedCFRModel(self.outbreak)
        self.recovery_model = RecoveryCorrectedCFRModel(self.outbreak)

    @lru_cache(maxsize=10)
    def estimate(self, t: int, start: int = 0) -> float:
        fatality_estimate = self.fatality_model.estimate(t, start=start)
        recovery_estimate = self.recovery_model.estimate(t, start=start)

        print("estimated recovery_corrected", self.outbreak.region, t,
              fatality_estimate / (fatality_estimate + (1 - recovery_estimate)))

        return fatality_estimate / (fatality_estimate + (1 - recovery_estimate))
