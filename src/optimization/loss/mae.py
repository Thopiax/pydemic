from sklearn.metrics import mean_absolute_error

from optimization.loss.base import BaseLoss


class MeanAbsoluteErrorLoss(BaseLoss):
    name: str = "MAE"

    def _loss(self):
        return mean_absolute_error(self.y_true, self.y_pred, sample_weight=self.sample_weight)