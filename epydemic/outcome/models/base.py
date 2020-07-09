from abc import ABC, abstractmethod
from pathlib import PosixPath, Path
from typing import List, Dict, Optional, Collection, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skopt.space import Dimension

from outbreak import Outbreak
from outcome.distribution.main import BaseOutcomeDistribution
from outcome.models.exceptions import TrivialTargetError
from outcome.optimizer.loss import SquaredErrorLoss
from outcome.optimizer.main import OutcomeOptimizer


class BaseOutcomeModel(ABC):
    name: str = "base"

    def __init__(self, outbreak: Outbreak, distribution: BaseOutcomeDistribution):
        self.outbreak = outbreak
        self.distribution = distribution

        self.domain = outbreak.cases.values

    @property
    def dimensions(self) -> Collection[Dimension]:
        return self.distribution.dimensions

    @property
    def parameters(self) -> List[float]:
        return list(self.distribution.parameters)

    @property
    def cache_path(self) -> PosixPath:
        return Path(self.outbreak.region) / self.name

    @parameters.setter
    def parameters(self, parameters: List[float]):
        self.distribution.parameters = parameters

    def _predict_incidence(self, t: int) -> int:
        K = np.minimum(t + 1, len(self.distribution.incidence_rate))

        # sum the expected number of deaths at t, for each of the last K days, including t
        return (self.domain[(t + 1) - K:(t + 1)] * self.distribution.incidence_rate[K - 1::-1]).sum()

    def predict(self, t: int, start: int = 0) -> int:
        return sum(self._predict_incidence(k) for k in range(start, t))

    def fit(self, t: int, start: int = 0, verbose: bool = False, random_state: int = 1, **kwargs) -> None:
        if self.target(t, start) == 0:
            raise TrivialTargetError

        optimizer = OutcomeOptimizer(self, verbose=verbose, random_state=random_state)

        loss = SquaredErrorLoss(self, t, start)
        optimization_result = optimizer.optimize(loss, **kwargs)

        return optimization_result

    @abstractmethod
    def target(self, t: int, start: int = 0) -> int:
        raise NotImplementedError
