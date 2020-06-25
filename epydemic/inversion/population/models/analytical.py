import numpy as np
import matplotlib.pyplot as plt


from epydemic.inversion.population.models.base import AbstractPopulationModel
from epydemic.inversion.population.learner import PopulationModelLearner


class AnalyticalPopulationModel(AbstractPopulationModel):
    def __repr__(self):
        return "analytical"

    def _update_model(self):
        T = len(self.otw)
        K = self.individual_model.K

        # set cases to be the entire outbreak cases (consider pre burn-in data)
        cases = self.otw.outbreak.cases

        self.expected_fatality_matrix = np.zeros((T, K))

        for t in range(T):
            otw_t = self.otw.start + t

            for k in range(min(otw_t + 1, K)):
                self.expected_fatality_matrix[t, k] = self.individual_model.fatality_rate[k] * cases[otw_t - k]

    def fit(self, **kwargs):
        self.learner = PopulationModelLearner(self)

        self.best_loss, self.best_parameters = self.learner.minimize_loss(**kwargs)
        self.parameters = self.best_parameters

    def predict(self):
        return self.individual_model.alpha * np.sum(self.expected_fatality_matrix, axis=1)

