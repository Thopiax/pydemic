from typing import Optional

from epydemic.inversion.population.learner.loss import LearnerLoss, MASELearnerLoss
from epydemic.inversion.population.models.base import BasePopulationModel
from epydemic.inversion.individual.exceptions import InvalidParametersError


class SingleLearnerObjectiveFunction(object):
    def __init__(self, model: BasePopulationModel, loss: Optional[LearnerLoss]):
        self.model = model
        self.loss = loss
        self.loss_weights = model.otw.resolved_case_rate.values

    def __call__(self, parameters):
        try:
            self.model.parameters = parameters
            y_pred = self.model.predict()
            y_true = self.model.target()

            if type(y_pred) is not tuple and type(y_pred) is not tuple:
                return self.loss(y_true, y_pred, sample_weight=self.loss_weights)

            # if both values are tuples
            assert len(y_true) == len(y_pred)

            return sum(self.loss(y_true[i], y_pred[i], sample_weight=self.loss_weights) for i in range(len(y_true)))

        except InvalidParametersError:
            # return a large penalty for parameters that lead to invalid cfr rate
            return 1_000
