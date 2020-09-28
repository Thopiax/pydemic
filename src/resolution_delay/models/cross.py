from functools import cached_property, lru_cache
from pathlib import PosixPath, Path
from typing import List, Type, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.cfr.estimates.resolved import ResolvedCFREstimate
from src.enums import Outcome
from src.outbreak import Outbreak
from src.resolution_delay.distributions.base import BaseResolutionDelayDistribution

from src.optimization.loss import MeanAbsoluteScaledErrorLoss, BaseLoss
from src.resolution_delay.distributions.discrete.negbinomial import NegBinomialResolutionDelayDistribution
from src.resolution_delay.models.base import BaseResolutionDelayModel
from src.resolution_delay.models.utils import expected_case_outcome_lag


class CrossResolutionDelayModel(BaseResolutionDelayModel):
    name: str = "XOL"

    def __init__(self, outbreak: Outbreak,
                 fatality_distribution: Optional[BaseResolutionDelayDistribution] = None,
                 recovery_distribution: Optional[BaseResolutionDelayDistribution] = None,
                 Loss: Type[BaseLoss] = MeanAbsoluteScaledErrorLoss):
        super().__init__(outbreak, Loss)

        self._cases = self.outbreak.cases.to_numpy()

        self.distributions = {
            Outcome.DEATH: fatality_distribution or NegBinomialResolutionDelayDistribution(),
            Outcome.RECOVERY: recovery_distribution or NegBinomialResolutionDelayDistribution()
        }

        self.actual_observed_outcomes = {
            Outcome.DEATH: self.outbreak.cumulative_deaths.to_numpy(),
            Outcome.RECOVERY: self.outbreak.cumulative_recoveries.to_numpy()
        }

        self._cfr_estimate = ResolvedCFREstimate(outbreak)

        self._target = np.column_stack((
            self.actual_observed_outcomes[Outcome.DEATH],
            self.actual_observed_outcomes[Outcome.RECOVERY],
            self.actual_observed_outcomes[Outcome.DEATH] + self.actual_observed_outcomes[Outcome.RECOVERY]
        ))


    @cached_property
    def cache_path(self) -> PosixPath:
        path_name = f"F{self.distributions[Outcome.DEATH].name}_R{self.distributions[Outcome.RECOVERY].name}"
        #
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

    def plot_incidence(self):
        plt.gca()

        plt.suptitle(self.parameters)

        self.distributions[Outcome.RECOVERY].plot_incidence(label="recovery", support_offset=-0.25)
        self.distributions[Outcome.DEATH].plot_incidence(color="red", support_offset=0.25, label="death")

        plt.legend()
        plt.show()

    def plot(self, t: int, start: int = 0):
        fig, axes = plt.subplots(nrows=2)

        fig.suptitle("XOL Model")

        deaths, recoveries, _ = self.predict(t, start=start).T

        axes[0].set_ylabel("# of people")
        axes[0].set_title("Deaths")
        pd.Series(self.actual_observed_outcomes[Outcome.DEATH][start:(t + 1)]).plot(ax=axes[0],
                                                                                               label="actual deaths")
        pd.Series(deaths).plot(ax=axes[0], label="predicted deaths")

        axes[1].set_ylabel("# of people")
        axes[1].set_title("Recoveries")
        pd.Series(self.actual_observed_outcomes[Outcome.RECOVERY][start:(t + 1)]).plot(ax=axes[1],
                                                                                                  label="actual recoveries")
        pd.Series(recoveries).plot(ax=axes[1], label="predicted recoveries")

        plt.legend()

        plt.show()

    @lru_cache
    def target(self, t: int, start: int = 0) -> np.ndarray:
        return self._target[start:(t + 1)]

    def sample_weight(self, t: int, start: int = 0) -> Optional[np.ndarray]:
        return None

    def expected_observed_outcomes(self, outcome: Outcome, t: int):
        return expected_case_outcome_lag(t, self._cases, self.distributions[outcome].incidence_rate)

    def predict(self, t: int, start: int = 0) -> np.ndarray:
        result = np.zeros((t + 1 - start, 3))

        alpha_R = self._cfr_estimate.estimate(t, start=start)

        for k in range(start, t + 1):
            # calculate the expected outcomes
            f_D = self.expected_observed_outcomes(Outcome.DEATH, k)
            f_R = self.expected_observed_outcomes(Outcome.RECOVERY, k)

            actual_deaths = self.actual_observed_outcomes[Outcome.DEATH][k]
            actual_recoveries = self.actual_observed_outcomes[Outcome.RECOVERY][k]

            predicted_deaths     = max(f_R - actual_recoveries, 0)
            predicted_recoveries = max(f_D - actual_deaths, 0)

            predicted_resolved_cases = alpha_R * f_D + (1 - alpha_R) * f_R

            result[k - start] = np.array([predicted_deaths, predicted_recoveries, predicted_resolved_cases])

        return result

