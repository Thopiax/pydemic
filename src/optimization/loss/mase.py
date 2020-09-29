from typing import Optional
import numpy as np

from sklearn.metrics import mean_absolute_error

from optimization.loss.base import BaseLoss


class MeanAbsoluteScaledErrorLoss(BaseLoss):
    name: str = "MASE"

    def __init__(self, model, t: int, **kwargs):
        super().__init__(model, t, **kwargs)

        self._scaling_coefficient = self.y_true[-1]

    def _loss(self):
        mae_loss = mean_absolute_error(
            self.y_true,
            self.y_pred,
            sample_weight=self.sample_weight
            # multioutput=(1 / self._scaling_coefficient)
        )

        return mae_loss / self._scaling_coefficient
