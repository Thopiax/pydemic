from functools import cached_property
from typing import Optional

import numpy as np
from abc import ABC, abstractmethod

from resolution_delay.distributions.exceptions import InvalidParameterError
from sklearn.metrics import mean_absolute_error


class BaseLoss(ABC):
    name = "base"

    def __init__(self, model, t: int, start: int = 0):
        self.model = model

        self.t = t
        self.start = start

    def __call__(self, parameters):
        try:
            self.model.parameters = parameters

            return self._loss()

        except InvalidParameterError:
            return 100_000

    @cached_property
    def y_true(self):
        return self.model.target(self.t, start=self.start)

    @cached_property
    def sample_weight(self):
        return self.model.sample_weight(self.t, start=self.start)

    @property
    def y_pred(self):
        return self.model.predict(self.t, start=self.start)

    @property
    def tag(self):
        return f"{self.__class__.name}__{self.start}_{self.t}"

    @abstractmethod
    def _loss(self):
        raise NotImplementedError


class MeanAbsoluteErrorLoss(BaseLoss):
    name: str = "MAE"

    def _loss(self):
        return mean_absolute_error(self.y_true, self.y_pred, sample_weight=self.sample_weight)


class MeanAbsoluteScaledErrorLoss(BaseLoss):
    name: str = "MASE"

    def __init__(self, model, t: int, **kwargs):
        super().__init__(model, t, **kwargs)

        self._scaling_coefficient = self.y_true.sum(axis=0)

    def _loss(self):
        loss = mean_absolute_error(
            self.y_true,
            self.y_pred,
            sample_weight=self.sample_weight,
            multioutput=(1.0 / self._scaling_coefficient)
        )

        return loss
