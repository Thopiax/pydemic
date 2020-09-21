import heapq
from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from pathlib import PosixPath, Path
from typing import List, Type, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from enums import Outcome
from optimization.utils import get_optimal_parameters
from outbreak import Outbreak
from outcome_lag.distributions.base import BaseOutcomeLagDistribution

from optimization.loss import MeanAbsoluteScaledErrorLoss, BaseLoss
from outcome_lag.distributions.discrete.negbinomial import NegBinomialOutcomeDistribution
from outcome_lag.models.base import BaseOutcomeLagModel
from outcome_lag.models.utils import expected_case_outcome_lag
from utils.plot import save_figure


class CrossOutcomeLagModel(BaseOutcomeLagModel):
    name: str = "XOL"

    def __init__(self, outbreak: Outbreak,
                 fatality_distribution: Optional[BaseOutcomeLagDistribution] = None,
                 recovery_distribution: Optional[BaseOutcomeLagDistribution] = None,
                 Loss: Type[BaseLoss] = MeanAbsoluteScaledErrorLoss):
        super().__init__(outbreak, Loss)

        self._cases = self.outbreak.cases.to_numpy()

        self.distributions = {
            Outcome.DEATH: fatality_distribution or NegBinomialOutcomeDistribution(),
            Outcome.RECOVERY: recovery_distribution or NegBinomialOutcomeDistribution()
        }

        self._actual_outcomes = {
            Outcome.DEATH: self.outbreak.deaths.to_numpy(),
            Outcome.RECOVERY: self.outbreak.recoveries.to_numpy()
        }

        self._target = np.column_stack((
            self._actual_outcomes[Outcome.DEATH],
            self._actual_outcomes[Outcome.RECOVERY],
            # replace NaN with -1.0 so that optimization doesn't break
            (self.outbreak.cumulative_deaths / self.outbreak.cumulative_resolved_cases).fillna(-1.0).to_numpy()
        ))

    @cached_property
    def cache_path(self) -> PosixPath:
        path_name = str(f"F{self.distributions[Outcome.DEATH].name}_R{self.distributions[Outcome.RECOVERY].name}")

        return super().cache_path / path_name

    @cached_property
    def dimensions(self):
        return self.distributions[Outcome.DEATH].dimensions + self.distributions[Outcome.RECOVERY].dimensions

    @property
    def parameters(self) -> List[float]:
        return list(self.distributions[Outcome.DEATH].parameters) + list(self.distributions[Outcome.RECOVERY].parameters)

    @parameters.setter
    def parameters(self, parameters: List[float]):
        n_fatality_parameters = self.distributions[Outcome.DEATH].n_parameters

        self.distributions[Outcome.DEATH].parameters = parameters[:n_fatality_parameters]
        self.distributions[Outcome.RECOVERY].parameters = parameters[n_fatality_parameters:]

    def plot(self, t: int, start: int = 0):
        fig, axes = plt.subplots(nrows=2)

        if (t, start) in self.results:
            self.parameters = get_optimal_parameters(self.results[(t, start)])

        fig.suptitle("XOL Model")

        deaths, recoveries, naive_cfr = self.predict(t, start=start).T

        axes[0].set_ylabel("# of people")
        axes[0].set_title("Deaths")
        pd.Series(self._actual_outcomes[Outcome.DEATH]).plot(ax=axes[0], label="actual deaths")
        pd.Series(deaths).plot(ax=axes[0], label="predicted deaths")

        axes[1].set_ylabel("# of people")
        axes[1].set_title("Recoveries")
        pd.Series(self._actual_outcomes[Outcome.RECOVERY]).plot(ax=axes[1], label="actual recoveries")
        pd.Series(recoveries).plot(ax=axes[1], label="predicted recoveries")

        plt.legend()

        plt.show()

    @lru_cache
    def target(self, t: int, start: int = 0) -> np.ndarray:
        return self._target[start:(t + 1)]

    def sample_weight(self, t: int, start: int = 0) -> Optional[np.ndarray]:
        return None

    def cumulative_outcome_lag(self, outcome: Outcome, t: int, start: int = 0):
        return sum(
            expected_case_outcome_lag(k, self._cases, self.distributions[outcome].incidence_rate)
            for k in range(start, t + 1)
        )

    def alpha(self, t: int, start: int = 0):
        F_D = sum(expected_case_outcome_lag(k, self._cases, self.distributions[Outcome.DEATH].incidence_rate) for k in range(start, t + 1))
        F_D = sum(expected_case_outcome_lag(k, self._cases, self.distributions[Outcome.DEATH].incidence_rate) for k in range(start, t + 1))

        return ()

    def predict(self, t: int, start: int = 0) -> np.ndarray:
        result = np.zeros((t + 1 - start, 3))

        for k in range(start, t + 1):
            # calculate the expected outcomes
            f_D = self.expected_case_outcome_lag(k, Outcome.DEATH)
            f_R = self.expected_case_outcome_lag(k, Outcome.RECOVERY)

            # create predictions
            pred_D = f_R - self._actual_outcomes[Outcome.RECOVERY][k]
            pred_R = f_D - self._actual_outcomes[Outcome.DEATH][k]

            pred_alpha_R = pred_D / (pred_R + pred_D)

            result[k - start] = np.array([pred_D, pred_R, pred_alpha_R])

        return result

