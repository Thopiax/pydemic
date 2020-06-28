import numpy as np
import matplotlib.pyplot as plt


from epydemic.inversion.population.models.base import BasePopulationModel
from epydemic.inversion.population.learner import PopulationModelLearner
from epydemic.inversion.individual import FatalityIndividualModel
from epydemic.inversion.individual.exceptions import InvalidParameters
from inversion.population.utils import verify_prediction


class FatalityPopulationModel(BasePopulationModel):
    IndividualModel = FatalityIndividualModel

    def __repr__(self):
        return "fatality"

    def _update_model(self):
        self.expected_fatality_matrix = self._build_expectation_matrix(self.individual_model.fatality_rate)

    def fit(self, **kwargs):
        self.learner = PopulationModelLearner(self)

        self.best_loss, self.best_parameters = self.learner.minimize_loss(**kwargs)
        self.parameters = self.best_parameters

    def target(self):
        return self.otw.deaths.values

    def predict(self):
        prediction = self.individual_model.alpha * np.sum(self.expected_fatality_matrix, axis=1)

        verify_prediction(prediction)

        return prediction

