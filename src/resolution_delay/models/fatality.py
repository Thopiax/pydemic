from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from pathlib import PosixPath, Path
from typing import List, Type, Optional

import numpy as np

from cfr.estimates.base import BaseCFREstimate
from cfr.estimates.naive import NaiveCFREstimate

from outbreak import Outbreak
from resolution_delay.distributions.base import BaseResolutionDelayDistribution

from optimization.loss import MeanAbsoluteScaledErrorLoss
from optimization.loss.base import BaseLoss
from resolution_delay.distributions.discrete.negbinomial import NegBinomialResolutionDelayDistribution
from resolution_delay.models.base import BaseResolutionDelayModel
from resolution_delay.models.utils import verify_pred


class FatalityResolutionDelayModel(BaseResolutionDelayModel):
    name: str = "FOL"

    def __init__(self, outbreak: Outbreak, distribution: Optional[BaseResolutionDelayDistribution] = None,
                 Loss: Type[BaseLoss] = MeanAbsoluteScaledErrorLoss):
        super().__init__(outbreak, Loss)

        self.distribution = distribution or NegBinomialResolutionDelayDistribution()

        self._target = self.outbreak.cumulative_deaths.to_numpy(dtype="float32")

        self._sample_weight = (self.outbreak.cumulative_resolved_cases / self.outbreak.cumulative_cases).to_numpy(dtype="float32")

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

    @lru_cache
    def target(self, t: int, start: int = 0) -> np.ndarray:
        return self._target[start:(t + 1)] - self._target[start]

    @lru_cache
    def sample_weight(self, t: int, start: int = 0) -> Optional[np.ndarray]:
        return self._sample_weight[start:(t + 1)]

    def _calculate_cecr(self, t: int, start: int = 0):
        expected_case_resolutions = np.convolve(
            self._cases[:(t + 1)],
            self.distribution.incidence_rate,
            mode="full"
        )

        verify_pred(expected_case_resolutions)

        return np.cumsum(expected_case_resolutions[start:(t + 1)])

    def _calculate_alpha(self, t: int, start: int = 0, cecr: Optional[np.array] = None):
        if cecr is None:
            cecr = self._calculate_cecr(t, start=start)

        return np.average(self.target(t, start=start) / cecr, weights=self.sample_weight(t, start=start))

    def predict(self, t: int, start: int = 0) -> np.ndarray:
        cecr = self._calculate_cecr(t, start=start)
        alpha = self._calculate_alpha(t, start=start, cecr=cecr)

        verify_pred(alpha * cecr)

        return alpha * cecr
