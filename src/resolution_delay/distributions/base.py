import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

from scipy.stats._distn_infrastructure import rv_frozen
from typing import Optional, NamedTuple, Iterable, Union, List

from skopt.space import Dimension

from src.resolution_delay.distributions.exceptions import InvalidParameterError
from src.resolution_delay.distributions.utils import MAX_RATE_PPF, MAX_SUPPORT_SIZE, verify_random_variable, verify_rate, \
    describe


def build_support(max_support_size: int = MAX_SUPPORT_SIZE, freq: float = 1) -> np.ndarray:
    return np.arange(max_support_size, step=freq)


class BaseResolutionDelayDistribution(ABC):
    Parameters: Optional[NamedTuple] = None
    name: str = "base"

    def __init__(self, *parameters, max_support_size: int = MAX_SUPPORT_SIZE,
                 max_rate_ppf: int = MAX_RATE_PPF, support_offset: float = 0.5):
        self._parameters: Optional[NamedTuple] = None
        self.random_variable: Optional[rv_frozen] = None

        self.support: Optional[np.ndarray] = None

        self.incidence_rate: Optional[pd.Series] = None
        self.hazard_rate: Optional[pd.Series] = None

        self.max_support_size = max_support_size
        self.support_offset = support_offset

        self.max_rate_ppf = max_rate_ppf

        if len(parameters) > 0:
            self.parameters = parameters

    def describe_random_variable(self):
        return describe(self.random_variable)

    @property
    def is_valid(self):
        return self._parameters is not None

    @property
    def n_parameters(self) -> int:
        return len(self.dimensions)

    @property
    def parameters(self) -> NamedTuple:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: Iterable[float]):
        # build & verify parameters
        self._parameters = self.build_parameters(parameters)
        self.verify_parameters(self._parameters)

        # build & verify random variable
        self.random_variable = self.build_random_variable(self._parameters)
        verify_random_variable(self.random_variable)

        self.support = build_support(max_support_size=self.max_support_size)

        # build & verify resolution_delay rate
        self.incidence_rate = self.build_incidence_rate(self.support, self.random_variable, offset=self.support_offset)
        verify_rate(self.incidence_rate)

        # build & verify hazard rate
        self.hazard_rate = self.build_hazard_rate(self.support, self.random_variable, self.incidence_rate,
                                                  offset=self.support_offset)
        verify_rate(self.hazard_rate)

    def build_parameters(self, parameters: Iterable[float]) -> NamedTuple:
        return self.__class__.Parameters._make(parameters)

    @abstractmethod
    def build_random_variable(self, parameters: NamedTuple) -> rv_frozen:
        raise NotImplementedError

    @abstractmethod
    def build_incidence_rate(self, support: np.ndarray, random_variable: rv_frozen, **kwargs) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def build_hazard_rate(self, support: np.ndarray, random_variable: rv_frozen, incidence_rate: pd.Series, **kwargs) -> pd.Series:
        raise NotImplementedError

    def plot_incidence(self, freq: float = 0.001, **kwargs):
        support = build_support(freq=freq, max_support_size=self.max_support_size)
        incidence_rate = self.build_incidence_rate(support, self.random_variable)

        self._plot_rate(self.incidence_rate, support, incidence_rate, **kwargs)

    def plot_hazard(self, freq: float = 0.001, **kwargs):
        support = build_support(freq=freq, max_support_size=self.max_support_size)
        incidence_rate = self.build_incidence_rate(support, self.random_variable)
        hazard_rate = self.build_hazard_rate(support, self.random_variable, incidence_rate)

        self._plot_rate(self.hazard_rate, support, hazard_rate, label="Hazard", color="orange", **kwargs)

    @abstractmethod
    def _plot_rate(self, rate, hf_support, hf_rate, color: str = "blue", label: str = "Incidence",
                   support_offset: Optional[float] = None, **kwargs):
        raise NotImplementedError

    @property
    def dimensions(self) -> List[Dimension]:
        raise NotImplementedError

    def verify_parameters(self, parameters: NamedTuple) -> bool:
        for i, parameter_dims in enumerate(self.dimensions):
            if parameters[i] not in parameter_dims:
                raise InvalidParameterError

        return True

