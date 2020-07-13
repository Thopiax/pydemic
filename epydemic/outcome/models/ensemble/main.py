from abc import ABC, abstractmethod
from pathlib import PosixPath
from typing import List, Dict, Optional, Collection

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skopt.space import Dimension

from outbreak import Outbreak
from outcome.distribution.base import BaseOutcomeDistribution
from outcome.models.exceptions import InvalidPredictionError


def verify_predictions(fatality_target, fatality_prediction, recovery_target, recovery_prediction):
    for pred in [fatality_prediction, recovery_prediction]:
        if np.isnan(pred).any() or np.isinf(pred).any() or pred == 0:
            raise InvalidPredictionError

    fatality_alpha = (fatality_target / fatality_prediction)
    recovery_alpha = 1 - (recovery_target / recovery_prediction)

    # assumption: cfr and recovery alpha estimates are a lower and upper bound on the true estimate, respectively.
    if fatality_alpha > recovery_alpha or fatality_alpha > 1 or recovery_alpha > 1:
        raise InvalidPredictionError


class OutcomeRegression(ABC):
    def __init__(self, outbreak: Outbreak, *distributions: BaseOutcomeDistribution, **kwargs):
        self.outbreak = outbreak
        self.domain = outbreak.cases.values

        self._distributions: List[BaseOutcomeDistribution] = list(distributions)

        self._distributions_map: Dict[str, BaseOutcomeDistribution] = {
            self.distribution_names[i]: dist for i, dist in enumerate(self._distributions)
        }

        # if the distributions are all valid (have set parameters), then we set the model's parameter
        if all(dist.is_valid for dist in self._distributions):
            self.parameters = self.parameters

    def get_distribution(self, name: str) -> BaseOutcomeDistribution:
        return self._distributions_map[name]

    def get_incidence_rate(self, name: str) -> Optional[pd.Series]:
        return self._distributions_map[name].incidence_rate

    @property
    def dimensions(self) -> List[Dimension]:
        return [dim for dist in self._distributions for dim in dist.dimensions]

    @property
    def parameters(self) -> List[float]:
        return [param for dist in self._distributions for param in list(dist.parameters)]

    @parameters.setter
    def parameters(self, parameters: List[float]):
        n_parameters_seen = 0

        for distribution in self._distributions:
            n_parameters_in_distribution = distribution.n_parameters

            distribution.parameters = parameters[n_parameters_seen:n_parameters_seen + n_parameters_in_distribution]

            n_parameters_seen += n_parameters_in_distribution

        assert n_parameters_seen == len(parameters)

        self._build_model()

    def plot(self) -> List[plt.axis]:
        raise NotImplementedError

    def _build_model(self) -> None:
        self._predictors = {}

        for name, distribution in self._distributions_map.items():
            self._predictors[name] = self._build_distribution_predictor(self.get_incidence_rate(name))

    def _build_distribution_predictor(self, rate: pd.Series):
        def _predict(t: int) -> int:
            K = np.minimum(t + 1, len(rate))

            # sum the expected number of deaths at t, for each of the last K days, including t
            return (self.domain[(t + 1) - K:(t + 1)] * rate[K - 1::-1]).sum()

        return _predict

    @abstractmethod
    def targets(self, t: Optional[int] = None) -> Collection[pd.Series]:
        raise NotImplementedError

    @abstractmethod
    def distribution_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def cache_path(self) -> PosixPath:
        raise NotImplementedError

    @abstractmethod
    def fit(self, t: int, verbose: bool = False, random_state: int = 1, **kwargs) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def predict(self, t: int) -> Collection[int]:
        raise NotImplementedError
