import numpy as np
import matplotlib.pyplot as plt

from epydemic.inversion.population.models.base import BasePopulationModel
from epydemic.inversion.population.learner import PopulationModelLearner
from epydemic.inversion.individual import DualIndividualModel
from epydemic.inversion.population.utils import verify_prediction


class DualPopulationModel(BasePopulationModel):
    IndividualModel = DualIndividualModel

    def __repr__(self):
        return "dual"

    def _update_model(self):
        self.expected_fatality_matrix = self._build_expectation_matrix(self.individual_model.fatality_rate)
        self.expected_recovery_matrix = self._build_expectation_matrix(self.individual_model.recovery_rate)

    def fit(self, **kwargs):
        self.learner = PopulationModelLearner(self)

        self.best_loss, self.best_parameters = self.learner.minimize_loss(**kwargs)
        self.parameters = self.best_parameters

    def target(self):
        return self.otw.deaths.values, self.otw.recoveries.values

    def predict(self):
        fatality_prediction = self.individual_model.alpha * np.sum(self.expected_fatality_matrix, axis=1)
        recovery_prediction = (1 - self.individual_model.alpha) * np.sum(self.expected_recovery_matrix, axis=1)

        verify_prediction(fatality_prediction)
        verify_prediction(recovery_prediction)

        return fatality_prediction, recovery_prediction
