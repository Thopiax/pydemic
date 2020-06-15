import numpy as np
import matplotlib.pyplot as plt


from epydemic.inversion.population.models.base import PopulationFatalityModel
from epydemic.inversion.population.learner import PopulationFatalityLearner


class AnalyticalPopulationFatalityModel(PopulationFatalityModel):
    def __repr__(self):
        return "analytical"

    def _update_model(self):
        T = len(self.otw)
        K = self.individual_model.K

        self.expected_fatality_matrix = np.zeros((T, K))

        cumulative_survival_probability = np.cumprod(1 - self.individual_model.hazard_rate)

        for t in range(T):
            for i in range(min(t + 1, K)):
                self.expected_fatality_matrix[t, i] = self.individual_model.hazard_rate[i] * self.otw.cases[t - i]

                if i >= 1:
                    self.expected_fatality_matrix[t, i] *= cumulative_survival_probability[i - 1]

    def fit(self, **kwargs):
        self.learner = PopulationFatalityLearner(self)

        self.best_loss, self.best_parameters = self.learner.minimize_loss(**kwargs)
        self.parameters = self.best_parameters

    def predict(self):
        return self.individual_model.alpha * np.sum(self.expected_fatality_matrix, axis=1)

