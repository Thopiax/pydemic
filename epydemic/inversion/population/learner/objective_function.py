from typing import Optional

from epydemic.inversion.population.learner.loss import LearnerLoss, MASELearnerLoss
from epydemic.inversion.population.models.base import PopulationFatalityModel
from epydemic.inversion.exceptions import InvalidFatalityRateException


class LearnerObjectiveFunction(object):
    def __init__(self, model: PopulationFatalityModel, loss: Optional[LearnerLoss]):
        self.model = model
        self.loss = loss

        self.y_true = model.otw.deaths.values
        self.loss_weights = model.otw.resolved_case_rate.values

    def __call__(self, parameters):
        try:
            self.model.parameters = parameters
            y_pred = self.model.predict()

            result = self.loss(self.y_true, y_pred, sample_weight=self.loss_weights)

            return result

        except InvalidFatalityRateException:
            # return a large penalty for parameters that lead to invalid fatality rate
            return 100
