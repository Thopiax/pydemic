from functools import cached_property
from typing import Optional

import numpy as np
from abc import ABC, abstractmethod

from outcome_lag.distributions.exceptions import InvalidParameterError
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error


class BaseOutcomeLoss(ABC):
    def __init__(self, model, t: int, start: int = 0, sample_weight: Optional[np.array] = None):
        self.model = model

        self.t = t
        self.start = start

        self.sample_weight = sample_weight

        if self.sample_weight is not None:
            self.sample_weight = self.sample_weight.iloc[start:(t + 1)]

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
    def _loss(self):
        raise NotImplementedError


class MeanAbsoluteErrorLoss(BaseOutcomeLoss):
    @property
    def tag(self):
        return f"mae__{self.start}_{self.t}"

    def _loss(self):
        return mean_absolute_error(self.y_true, self.y_pred, sample_weight=self.sample_weight)


class MeanAbsoluteScaledErrorLoss(BaseOutcomeLoss):

    def __init__(self, model, t: int, **kwargs):
        super().__init__(model, t, **kwargs)

        self._scaling_coefficient = self.y_true.sum()

    @property
    def tag(self):
        return f"mase__{self.start}_{self.t}"

    def _loss(self):
        return mean_absolute_error(self.y_true, self.y_pred, sample_weight=self.sample_weight) \
               / self._scaling_coefficient


class MeanSquaredLogErrorLoss(BaseOutcomeLoss):
    @property
    def tag(self):
        return f"mlse__{self.start}_{self.t}"

    def _loss(self):
        return mean_squared_log_error(self.y_true, self.y_pred, sample_weight=self.sample_weight)
