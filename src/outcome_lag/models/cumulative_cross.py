import heapq
from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from pathlib import PosixPath, Path
from typing import List, Type, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from enums import Outcome
from optimization.utils import get_optimal_parameters
from outbreak import Outbreak
from outcome_lag.distributions.base import BaseOutcomeLagDistribution

from optimization.loss import MeanAbsoluteScaledErrorLoss, BaseLoss
from outcome_lag.distributions.discrete.negbinomial import NegBinomialOutcomeDistribution
from outcome_lag.distributions.exceptions import InvalidParameterError
from outcome_lag.models.base import BaseOutcomeLagModel
from outcome_lag.models.utils import expected_case_outcome_lag
from utils.plot import save_figure


class CumulativeCrossOutcomeLagModel(BaseOutcomeLagModel):
    name: str = "cum_XOL"

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

        self.actual_observed_cumulative_outcomes = {
            Outcome.DEATH: self.outbreak.cumulative_deaths.to_numpy(),
            Outcome.RECOVERY: self.outbreak.cumulative_recoveries.to_numpy()
        }

    @cached_property
    def cache_path(self) -> PosixPath:
        path_name = f"F{self.distributions[Outcome.DEATH].name}_R{self.distributions[Outcome.RECOVERY].name}"

        print(path_name)

        return super().cache_path / path_name

    @cached_property
    def dimensions(self):
        return self.distributions[Outcome.DEATH].dimensions + self.distributions[Outcome.RECOVERY].dimensions

    @property
    def parameters(self) -> List[float]:
        return list(self.distributions[Outcome.DEATH].parameters) + list(
            self.distributions[Outcome.RECOVERY].parameters)

    @parameters.setter
    def parameters(self, parameters: List[float]):
        n_fatality_parameters = self.distributions[Outcome.DEATH].n_parameters

        self.distributions[Outcome.DEATH].parameters = parameters[:n_fatality_parameters]
        self.distributions[Outcome.RECOVERY].parameters = parameters[n_fatality_parameters:]

    def plot_incidence(self):
        plt.gca()

        plt.suptitle(self.parameters)

        self.distributions[Outcome.RECOVERY].plot_incidence(label="recovery", support_offset=-0.25)
        self.distributions[Outcome.DEATH].plot_incidence(color="red", support_offset=0.25, label="death")

        plt.legend()
        plt.show()


    @lru_cache
    def target(self, t: int, start: int = 0) -> np.ndarray:
        cumulative_cases = self.outbreak.cumulative_cases
        cumulative_deaths = self.actual_observed_cumulative_outcomes[Outcome.DEATH]
        cumulative_recoveries = self.actual_observed_cumulative_outcomes[Outcome.RECOVERY]

        # remove incidence before start
        target_deaths = cumulative_deaths[t + 1] - cumulative_deaths[start]
        target_recoveries = cumulative_recoveries[t + 1] - cumulative_recoveries[start]

        return np.array([[
            target_deaths, target_recoveries,
            target_deaths / cumulative_cases,
            target_deaths / (target_deaths + target_recoveries)
        ]])

    def sample_weight(self, t: int, start: int = 0) -> Optional[np.ndarray]:
        return None

    def expected_observed_outcomes(self, outcome: Outcome, t: int):
        return expected_case_outcome_lag(t, self._cases, self.distributions[outcome].incidence_rate)

    def predict(self, t: int, start: int = 0) -> np.ndarray:
        F_D = sum(self.expected_observed_outcomes(Outcome.DEATH, k) for k in range(start, t + 1))
        F_R = sum(self.expected_observed_outcomes(Outcome.RECOVERY, k) for k in range(start, t + 1))

        [[actual_D, actual_R, _, _]] = self.target(t, start=start)

        penalties = [
            F_R - actual_R,
            F_D - actual_D
        ]

        if any([x < 0 for x in penalties]):
            raise InvalidParameterError

        pred_D = F_R - actual_R
        pred_R = F_D - actual_D

        pred_alpha_N = pred_D / self.outbreak.cumulative_cases[t + 1]
        pred_alpha_R = pred_D / (pred_R + pred_D)

        return np.array([[pred_D, pred_R, pred_alpha_N, pred_alpha_R]])

