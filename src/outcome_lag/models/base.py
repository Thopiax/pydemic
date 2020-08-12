from abc import ABC, abstractmethod
from functools import lru_cache, cached_property
from pathlib import PosixPath, Path
from typing import List, Optional, Type, Dict, Collection

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import OptimizeResult
from skopt.space import Dimension, Real

from outbreak import Outbreak
from outcome_lag.distributions.base import BaseOutcomeLagDistribution
from outcome_lag.optimizer.loss import MeanSquaredLogErrorLoss, MeanAbsoluteScaledErrorLoss
from outcome_lag.optimizer.main import OutcomeLagOptimizer
from outcome_lag.optimizer.utils import get_optimal_parameters

ALPHA_DIMENSION = Real(0.0, 1.0)


class BaseOutcomeModel(ABC):
    name: str = "base"

    def __init__(self, outbreak: Outbreak, distribution: BaseOutcomeLagDistribution):
        self.outbreak = outbreak

        self._optimal_parameters: pd.DataFrame = pd.DataFrame(columns=self.parameter_names)

        self.alpha: float = -1.0

        self.distribution: BaseOutcomeLagDistribution = distribution
        self.domain: np.ndarray = outbreak.cases.values

    @cached_property
    def dimensions(self) -> List[Dimension]:
        return [ALPHA_DIMENSION, *self.distribution.dimensions]

    @property
    def parameters(self) -> List[float]:
        return [self.alpha] + list(self.distribution.parameters)

    @parameters.setter
    def parameters(self, parameters: List[float]):
        self.alpha = parameters[0]
        self.distribution.parameters = parameters[1:]

    @property
    def parameter_names(self) -> List[str]:
        return ["alpha"] + list(self.distribution.parameters._fields)

    @property
    def cache_path(self) -> PosixPath:
        return Path(self.outbreak.region) / self.name

    def _predict_incidence(self, t: int) -> float:
        K = np.minimum(t + 1, self.distribution.max_rate_support_size)

        # sum the expected number of deaths at t, for each of the last K days, including t
        return (self.domain[(t + 1) - K:(t + 1)] * self.distribution.incidence_rate[K - 1::-1]).sum()

    def fit(self, t: int, start: int = 0, verbose: bool = True, random_state: int = 1, **kwargs) -> OptimizeResult:
        optimizer = OutcomeLagOptimizer(self, verbose=verbose, random_state=random_state)

        loss = MeanAbsoluteScaledErrorLoss(self, t, start=start)
        initial_parameter_points = self._optimal_parameters.values.tolist()

        optimization_result = optimizer.optimize(loss, initial_parameter_points=initial_parameter_points, **kwargs)

        self._optimal_parameters.loc[t, :] = get_optimal_parameters(optimization_result)
        self.parameters = self._optimal_parameters.loc[t, :]

        return optimization_result

    @abstractmethod
    def predict(self, t: int, start: int = 0) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def target(self, t: int, start: int = 0) -> np.array:
        raise NotImplementedError