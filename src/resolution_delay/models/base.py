from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from pathlib import PosixPath, Path
from typing import List, Type, Optional, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

from optimization.utils import get_optimal_parameters, get_n_best_parameters
from outbreak import Outbreak

from optimization import Optimizer
from optimization.loss import MeanAbsoluteScaledErrorLoss
from optimization.loss.base import BaseLoss


class BaseResolutionDelayModel(ABC):
    name: str = "base"

    def __init__(self, outbreak: Outbreak, Loss: Type[BaseLoss] = MeanAbsoluteScaledErrorLoss):
        self.outbreak = outbreak
        self._Loss = Loss

        self._cases = self.outbreak.cases.to_numpy(dtype="float32")

        self.results = {}

    @cached_property
    def cache_path(self) -> PosixPath:
        return Path(self.outbreak.region) / self.__class__.name

    def fit(self, t: int, start: int = 0, n_best_parameters=1, **kwargs) -> Union[List[float], List[Tuple[float, List[float]]]]:
        loss = self._Loss(self, t, start=start)
        optimizer = Optimizer(self.dimensions, cache_path=self.cache_path)

        optimization_result = optimizer.optimize(loss, **kwargs)

        self.results[(t, start)] = optimization_result
        self.parameters = get_optimal_parameters(optimization_result)

        if n_best_parameters == 1:
            return get_optimal_parameters(optimization_result)

        return get_n_best_parameters(n_best_parameters, optimization_result)

    def plot_prediction(self, t: int, start: int = 0):
        ax = plt.gca()

        ax.plot(np.cumsum(self._cases[start:(t + 1)]), label="cases")
        ax.plot(self.target(t, start=start), label="true")
        ax.plot(self.predict(t, start=start), label="prediction")

        plt.legend()

    # ABSTRACT #

    @property
    @abstractmethod
    def dimensions(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> List[float]:
        raise NotImplementedError

    @parameters.setter
    @abstractmethod
    def parameters(self, parameters: List[float]):
        raise NotImplementedError

    @abstractmethod
    def target(self, t: int, start: int = 0) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def sample_weight(self, t: int, start: int = 0) -> Optional[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, t: int, start: int = 0) -> np.ndarray:
        raise NotImplementedError
