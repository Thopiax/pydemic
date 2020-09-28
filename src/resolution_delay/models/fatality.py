from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from pathlib import PosixPath, Path
from typing import List, Type, Optional

import numpy as np

from cfr.estimates.base import BaseCFREstimate
from cfr.estimates.naive import NaiveCFREstimate

from outbreak import Outbreak
from resolution_delay.distributions.base import BaseResolutionDelayDistribution

from optimization.loss import MeanAbsoluteScaledErrorLoss, BaseLoss
from resolution_delay.models.base import BaseResolutionDelayModel


class FatalityResolutionDelayModel(BaseResolutionDelayModel):
    name: str = "FOL"

    def __init__(self, outbreak: Outbreak, distribution: BaseResolutionDelayDistribution,
                 Loss: Type[BaseLoss] = MeanAbsoluteScaledErrorLoss,
                 CFR_estimate: Type[BaseCFREstimate] = NaiveCFREstimate):
        super().__init__(outbreak, Loss)

        self.distribution = distribution
        self._base_cfr_estimate = CFR_estimate(self.outbreak)

        self._cases = self.outbreak.cases.to_numpy()
        self._target = self.outbreak.deaths.to_numpy()

    @property
    def dimensions(self):
        return self.distribution.dimensions

    @property
    def parameters(self) -> List[float]:
        return list(self.distribution.parameters)

    @parameters.setter
    def parameters(self, parameters: List[float]):
        self.distribution.parameters = parameters

    @cached_property
    def cache_path(self) -> PosixPath:
        return super().cache_path / self.distribution.name

    def target(self, t: int, start: int = 0) -> np.ndarray:
        return self._target[start:(t + 1)]

    def sample_weight(self, t: int, start: int = 0) -> Optional[np.ndarray]:
        return None

    def predict(self, t: int, start: int = 0) -> np.ndarray:
        expected_case_resolutions = np.convolve(
            self._cases[:(t + 1)],
            self.distribution.incidence_rate,
            mode="full"
        )

        result = expected_case_resolutions[start:]

        return result
