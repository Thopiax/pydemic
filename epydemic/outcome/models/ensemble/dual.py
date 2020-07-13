from typing import Union
from pathlib import Path
from typing import Tuple, Collection, List, Optional

import pandas as pd
from matplotlib import pyplot as plt

from epydemic.outbreak import Outbreak
from epydemic.outcome.distribution.base import BaseOutcomeDistribution
from epydemic.outcome.optimizer.main import OutcomeOptimizer
from outcome.models.ensemble.main import verify_predictions, OutcomeRegression
from outcome.models.exceptions import TrivialTargetError
from outcome.optimizer.utils import get_optimal_parameters


class DualOutcomeRegression(OutcomeRegression):
    def __init__(self, outbreak: Outbreak, fatality_distribution: BaseOutcomeDistribution,
                 recovery_distribution: BaseOutcomeDistribution, **kwargs):
        super().__init__(outbreak, fatality_distribution, recovery_distribution, **kwargs)

    @property
    def cache_path(self):
        return Path(self.outbreak.region)

    @property
    def distribution_names(self) -> Collection[str]:
        return "fatality", "recovery"

    def get_target(self, name: str) -> pd.Series:
        if name == "fatality":
            return self.outbreak.deaths.values
        elif name == "recovery":
            return self.outbreak.recoveries.values
        else:
            raise LookupError

    def targets(self, t: Optional[int] = None) -> Union[Tuple[pd.Series, pd.Series], Tuple[int, int]]:
        fatality_target, recovery_target = self.outbreak.deaths.values, self.outbreak.recoveries.values

        if t is None:
            return fatality_target, recovery_target

        return fatality_target[t], recovery_target[t]

    def fit(self, t: int, verbose: bool = False, random_state: int = 1, **kwargs) -> None:
        fatality_target, recovery_target = self.targets(t)

        # if either targets are zero, we should skip fitting
        if fatality_target == 0 or recovery_target == 0:
            raise TrivialTargetError

        for distribution in self.distribution_names:
            optimizer = OutcomeOptimizer(self, distribution, verbose=verbose, random_state=random_state)

            optimization_result = optimizer.optimize(t, **kwargs)

        self.parameters = get_optimal_parameters(optimization_result)

    def predict(self, t: int) -> Tuple[int, int]:
        fatality_prediction, recovery_prediction = self._predictors["fatality"](t), self._predictors["recovery"](t)

        fatality_target, recovery_target = self.targets(t)
        verify_predictions(fatality_target, fatality_prediction, recovery_target, recovery_prediction)

        return fatality_prediction, recovery_prediction

    def alpha(self, t: int) -> float:
        fatality_prediction, recovery_prediction = self.predict(t)
        fatality_target, recovery_target = self.targets(t)

        fatality_alpha = (fatality_target / fatality_prediction)
        recovery_alpha = 1 - (recovery_target / recovery_prediction)

        return fatality_alpha / (fatality_alpha + (1 - recovery_alpha))

    def plot(self) -> List[plt.axis]:
        pass
