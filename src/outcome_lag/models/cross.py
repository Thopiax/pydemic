from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from pathlib import PosixPath, Path
from typing import List, Type, Optional

import numpy as np

from outbreak import Outbreak
from outcome_lag.distributions.base import BaseOutcomeLagDistribution

from optimization.loss import MeanAbsoluteScaledErrorLoss, BaseLoss
from outcome_lag.models.base import BaseOutcomeLagModel
from outcome_lag.models.utils import expected_case_outcome_lag


class CrossOutcomeLagModel(BaseOutcomeLagModel):
    name: str = "XOL"

    def __init__(self, outbreak: Outbreak, fatality_distribution: BaseOutcomeLagDistribution,
                 recovery_distribution: BaseOutcomeLagDistribution, Loss: Type[BaseLoss] = MeanAbsoluteScaledErrorLoss):
        super().__init__(outbreak, Loss)

        self.fatality_distribution = fatality_distribution
        self.recovery_distribution = recovery_distribution

        self._cases = self.outbreak.cases.to_numpy()
        self._deaths = self.outbreak.deaths.to_numpy()
        self._recoveries = self.outbreak.recoveries.to_numpy()

        self._target = np.column_stack((
            self._deaths,
            self._recoveries,
            # replace NaN with -1.0 so that optimization doesn't break
            (self.outbreak.cumulative_deaths / self.outbreak.cumulative_resolved_cases).fillna(-1.0).to_numpy()
        ))

    @cached_property
    def cache_path(self) -> PosixPath:
        return super().cache_path / f"F{self.fatality_distribution.name}_R{self.recovery_distribution}"

    @cached_property
    def dimensions(self):
        return self.fatality_distribution.dimensions + self.recovery_distribution.dimensions

    @property
    def parameters(self) -> List[float]:
        return list(self.fatality_distribution.parameters) + list(self.recovery_distribution.parameters)

    @parameters.setter
    def parameters(self, parameters: List[float]):
        n_fatality_parameters = self.fatality_distribution.n_parameters

        self.fatality_distribution.parameters = parameters[:n_fatality_parameters]
        self.recovery_distribution.parameters = parameters[n_fatality_parameters:]

    @lru_cache
    def target(self, t: int, start: int = 0) -> np.ndarray:
        return self._target[start:(t + 1)]

    def sample_weight(self, t: int, start: int = 0) -> Optional[np.ndarray]:
        return None

    def predict(self, t: int, start: int = 0) -> np.ndarray:
        result = np.zeros((t + 1 - start, 3))

        for k in range(start, t + 1):
            f_D = expected_case_outcome_lag(k, self._cases, self.fatality_distribution.incidence_rate)
            f_R = expected_case_outcome_lag(k, self._cases, self.recovery_distribution.incidence_rate)

            pred_D = f_R - self._recoveries
            pred_R = f_D - self._deaths

            pred_alpha_R = pred_D / (pred_R + pred_D)

            result[k - start] = [pred_D, pred_R, pred_alpha_R]

        return result

