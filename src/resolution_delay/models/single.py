from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from pathlib import PosixPath, Path
from typing import List, Type, Optional

import numpy as np

from outbreak import Outbreak
from resolution_delay.distributions.base import BaseResolutionDelayDistribution

from optimization.loss import MeanAbsoluteScaledErrorLoss
from optimization.loss.base import BaseLoss
from resolution_delay.distributions.discrete.negative_binomial import NegativeBinomialResolutionDelayDistribution
from resolution_delay.models.base import BaseResolutionDelayModel
from resolution_delay.models.utils import verify_pred, verify_probability


class SingleResolutionDelayModel(BaseResolutionDelayModel, ABC):
    def __init__(self, outbreak: Outbreak, distribution: Optional[BaseResolutionDelayDistribution] = None,
                 Loss: Type[BaseLoss] = MeanAbsoluteScaledErrorLoss, **kwargs):
        super().__init__(outbreak, Loss)

        self.distribution = distribution or NegativeBinomialResolutionDelayDistribution()

        # must be overwritten in child classes
        self._target = None

        self._sample_weight = (self.outbreak.cumulative_resolved_cases / self.outbreak.cumulative_cases).to_numpy(dtype="float32")

        self._optimizer_kwargs = kwargs

    @property
    def dimensions(self):
        return self.distribution.dimensions

    @property
    def parameters(self) -> List[float]:
        return list(self.distribution.parameters)

    @parameters.setter
    def parameters(self, parameters: List[float]):
        self.distribution.parameters = parameters

    @property
    def cache_path(self) -> PosixPath:
        # return super().cache_path / self.distribution.name
        return Path(self.outbreak.region) / self.__class__.name / self.distribution.name

    @lru_cache
    def target(self, t: int, start: int = 0) -> np.ndarray:
        assert self._target is not None

        result = self._target[start:(t + 1)]

        if start > 0:
            result -= self._target[start - 1]

        return result

    @lru_cache
    def sample_weight(self, t: int, start: int = 0) -> Optional[np.ndarray]:
        return self._sample_weight[start:(t + 1)]

    def regulizer(self, t: int, start: int = 0):
        alpha = self.alpha(t, start=start)

        return

    def _calculate_cecr(self, t: int, start: int = 0):
        expected_case_resolutions = np.convolve(
            self._cases[:(t + 1)],
            self.distribution.incidence_rate,
            mode="full"
        )

        verify_pred(expected_case_resolutions)

        return np.cumsum(expected_case_resolutions[start:(t + 1)])

    def _calculate_alpha(self, t: int, start: int, cecr: np.array, with_variance: bool = False):
        ratios = self.target(t, start=start) / cecr

        alpha = np.average(ratios, weights=self.sample_weight(t, start=start))
        verify_probability(alpha)

        if with_variance:
            alpha_variance = np.average((ratios - alpha) ** 2, weights=self.sample_weight(t, start=start))

            return alpha, alpha_variance

        return alpha

    def predict(self, t: int, start: int = 0) -> np.ndarray:
        cecr = self._calculate_cecr(t, start=start)
        alpha = self._calculate_alpha(t, start=start, cecr=cecr)

        verify_pred(alpha * cecr)

        return alpha * cecr
