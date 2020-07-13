from functools import cached_property
from typing import Optional

import numpy as np
from abc import ABC, abstractmethod

from outcome.distribution.exceptions import InvalidParameterError
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error


def build_scaling_coefficient(y_true):
    if y_true.shape[0] == 1:
        return y_true[0]
    else:
        return np.mean(np.abs(np.diff(y_true)))


class BaseOutcomeLoss(ABC):
    def __init__(self, model, t: int, start: int = 0, sample_weight: Optional[np.array] = None):
        self.model = model

        self.t = t
        self.start = start

        self.sampled_weight = sample_weight

        if self.sample_weight is not None:
            self.sampled_weight = self.sampled_weight.iloc[start:(t + 1)]

    def __call__(self, parameters):
        try:
            self.model.parameters = parameters

            return self._loss()

        except InvalidParameterError:
            return 100_000

    @cached_property
    def y_true(self):
        return self.model.target(self.t, start=self.start)

    @property
    def y_pred(self):
        return self.model.predict(self.t, start=self.start)

    @property
    @abstractmethod
    def tag(self):
        raise NotImplementedError

    @abstractmethod
    def _loss(self, sample_weight=None):
        yield
        raise NotImplementedError


class MeanAbsoluteErrorLoss(BaseOutcomeLoss):
    @property
    def tag(self):
        return f"mae__{self.start}_{self.t}"

    def _loss(self):
        return mean_absolute_error(self.y_true, self.y_pred, sample_weight=self.sampled_weight)


class MeanAbsoluteScaledErrorLoss(BaseOutcomeLoss):
    @property
    def tag(self):
        return f"mase__{self.start}_{self.t}"

    def _loss(self):
        scaling_coefficient = build_scaling_coefficient(self.y_true)

        return mean_absolute_error(self.y_true, self.y_pred, sample_weight=self.sample_weight) / scaling_coefficient


class MeanSquaredLogErrorLoss(BaseOutcomeLoss):
    @property
    def tag(self):
        return f"mlse__{self.start}_{self.t}"

    def _loss(self):
        return mean_squared_log_error(self.y_true, self.y_pred, sample_weight=self.sample_weight)
