import numpy as np

from epydemic.inversion.population.models import AnalyticalPopulationModel
from .base import AbstractCFREstimator


class MortalityRateCFREstimator(AbstractCFREstimator):
    # source: Early Epidemiological Assessment of the Virulence of Emerging Infectious Diseases
    def __repr__(self):
        return "mrCFR"

    def estimate(self, t: int = -1, **kwargs) -> float:
        super().estimate(t)

        if self.outbreak.cumulative_deaths[t] == 0:
            return np.nan

        if t < 0:
            t = len(self.outbreak) - t

        model = AnalyticalPopulationModel(self.outbreak, verbose=False)
        model.fit(dry_run=True, **kwargs)

        return model.individual_model.alpha


class ExpectedMortalityRateCFREstimator(AbstractCFREstimator):
    def __repr__(self):
        return "EmrCFR"

    def estimate(self, t: int = -1, **kwargs) -> float:
        super().estimate(t)

        if t < 0:
            t = len(self.outbreak) - t

        model = AnalyticalPopulationModel(self.outbreak, verbose=False)
        model.fit(dry_run=True, **kwargs)

        (best_expected_alpha, _, _), _ = model.learner.best_expected_loss

        return best_expected_alpha
