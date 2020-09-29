import numpy as np

from sklearn.metrics import mean_squared_log_error

from optimization.loss.base import BaseLoss


# def weighted_avg_and_std(values, weights):
#     # Fast and numerically precise:
#     variance = np.average((values - average) ** 2, weights=weights)
#     return (average, np.sqrt(variance))


class MeanSquaredLogErrorLoss(BaseLoss):
    name: str = "MSLE"

    def _loss(self):
        return mean_squared_log_error(self.y_true, self.y_pred, sample_weight=self.sample_weight)