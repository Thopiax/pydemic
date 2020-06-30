from typing import Callable, Iterable, Sized, Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Dimension

from epydemic.outcome.distribution.main import OutcomeDistribution
from epydemic.outbreak import Outbreak


class InvalidPrediction(Exception):
    pass


def verify_predictions(fatality_target, fatality_prediction, recovery_target, recovery_prediction):
    fatality_alpha = fatality_target / fatality_prediction
    recovery_alpha = 1 - recovery_target / recovery_prediction

    # assumption: fatality and recovery alpha estimates are a lower and upper bound on the true estimate, respectively.
    if fatality_alpha > 1 - recovery_alpha:
        raise InvalidPrediction


class DualOutcomeEstimator:
    def __init__(self, outbreak: Outbreak, fatality_distribution: OutcomeDistribution, recovery_distribution: OutcomeDistribution):
        self.outbreak = outbreak

        self.alphas = pd.Series()

        self.cases = self.outbreak.cases.values

        self.fatality_distribution = fatality_distribution
        self.fatality_target = self.outbreak.deaths.values

        self.recovery_distribution = self.fatality_distribution
        self.recovery_target = self.outbreak.recoveries.values

        if self.fatality_distribution.is_valid and self.recovery_distribution.is_valid:
            self._update_model()

    @property
    def dimensions(self) -> List[Dimension]:
        return list(self.fatality_distribution.dimensions) + list(self.recovery_distribution.dimensions)

    @property
    def parameters(self) -> List[float]:
        return list(self.fatality_distribution.parameters) + list(self.recovery_distribution.parameters)

    @parameters.setter
    def parameters(self, parameters: List[float]):
        # TODO: make generic
        assert len(parameters) == len(self.dimensions)

        self.fatality_distribution.parameters = parameters[:2]
        self.recovery_distribution.parameters = parameters[2:]

        self._update_model()

    def _update_model(self) -> None:
        self.fatality_rate = self.fatality_distribution.incidence_rate
        self.predict_fatality = self._build_predict(self.fatality_rate)

        self.recovery_rate = self.recovery_distribution.incidence_rate
        self.predict_recovery = self._build_predict(self.recovery_rate)

    def predict(self, t: int) -> Tuple[int, int]:
        fatality_target, recovery_target = self.target(t)
        fatality_prediction, recovery_prediction = self.predict_fatality(t), self.predict_recovery(t)

        verify_predictions(fatality_target, fatality_prediction, recovery_target, recovery_prediction)

        return fatality_prediction, recovery_target

    def target(self, t: int) -> Tuple[int, int]:
        return self.fatality_target[t], self.recovery_target[t]

    def error(self, t: int) -> float:
        fatality_prediction, recovery_prediction = self.predict(t)
        fatality_target, recovery_target = self.target(t)

        return np.abs(1 - (recovery_target / recovery_prediction) - fatality_target / fatality_prediction)

    def alpha(self, t: int) -> float:
        fatality_prediction, recovery_prediction = self.predict(t)
        fatality_target, recovery_target = self.target(t)

        return np.average([1 - (recovery_target / recovery_prediction), fatality_target / fatality_prediction])

    def _build_predict(self, rate: pd.Series) -> Callable:
        # set cases to be the entire outbreak cases (consider pre burn-in data)
        cases = self.outbreak.cases.values
        reversed_rate = rate.values[::-1]

        def _predict(t: int) -> int:
            K = np.minimum(t, len(rate))

            # sum the expected number of deaths at t, for each of the last K days, including t
            return (cases[(t + 1) - K:(t + 1)] * reversed_rate[len(rate) - K:]).sum()

        return _predict

    def plot_fatality(self) -> plt.axis:
        ax = plt.gca()
        ax.set_title("Fatality Prediction")

        self._plot_prediction(self.predict_fatality, self.fatality_target)

        return ax

    def plot_recovery(self) -> plt.axis:
        ax = plt.gca()
        ax.set_title("Recovery Prediction")

        self._plot_prediction(self.predict_recovery, self.recovery_target)

        return ax

    def _plot_prediction(self, predict: Callable, target: np.ndarray) -> plt.axis:
        ax = plt.gca()

        support = np.arange(len(self.outbreak))
        prediction = [predict(t) for t in support]

        plt.bar(support, prediction, width=0.5, label="Prediction", color="blue", ax=ax)
        plt.bar(support + 0.5, target, width=0.5, label="Target", color="orange", ax=ax)

        return ax

