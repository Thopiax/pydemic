import numpy as np
from abc import ABC

from outcome.models.base import BaseOutcomeModel
from outcome.distribution.exceptions import InvalidParameterError


class BaseOutcomeLoss(ABC):
    def __init__(self, model: BaseOutcomeModel, t: int, start: int = 0):
        self.model = model

        self.t = t
        self.start = start

        self.tag = f"{start}_{t}"

    def __call__(self, parameters):
        try:
            self.model.parameters = parameters

            return self._loss(
                self.model.target(self.t, start=self.start),
                self.model.predict(self.t, start=self.start)
            )

        except InvalidParameterError:
            return 100_000

    @staticmethod
    def _loss(y_true, y_pred):
        raise NotImplementedError


class AbsoluteErrorLoss(BaseOutcomeLoss):
    def _loss(self, y_true, y_pred):
        return np.abs(y_true - y_pred)


class SquaredErrorLoss(BaseOutcomeLoss):
    def _loss(self, y_true, y_pred):
        return np.square(y_true - y_pred)
