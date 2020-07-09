import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

from scipy.stats._distn_infrastructure import rv_frozen
from typing import Optional, NamedTuple, Iterable, Union, Collection

from skopt.space import Dimension

from epydemic.outcome.distribution.exceptions import InvalidParameterError
from outcome.distribution.utils import MAX_RATE_PPF, MAX_RATE_SUPPORT_SIZE, verify_random_variable, verify_rate, \
    describe


class BaseOutcomeDistribution(ABC):
    Parameters: Optional[NamedTuple] = None
    name: str = "base"

    def __init__(self, *parameters, max_rate_support_size: int = MAX_RATE_SUPPORT_SIZE,
                 max_rate_ppf: int = MAX_RATE_PPF, rate_support_offset: float = 0.5):
        self._parameters: Optional[NamedTuple] = None
        self.random_variable: Optional[rv_frozen] = None

        self.support: Optional[np.ndarray] = None

        self.incidence_rate: Optional[pd.Series] = None
        self.hazard_rate: Optional[pd.Series] = None

        self.max_rate_support_size = max_rate_support_size
        self.rate_support_offset = rate_support_offset

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
    def parameters(self) -> Optional[NamedTuple]:
        if self._parameters is None:
            return None

        return self._parameters

    @parameters.setter
    def parameters(self, parameters: Iterable[float]):
        # build & verify parameters
        self._parameters = self.build_parameters(parameters)
        self.verify_parameters(self._parameters)

        # build & verify random variable
        self.random_variable = self.build_random_variable(self._parameters)
        verify_random_variable(self.random_variable)

        self.support = self.build_support(max_rate_support_size=self.max_rate_support_size)

        # build & verify outcome rate
        self.incidence_rate = self.build_incidence_rate(self.support, self.random_variable, offset=self.rate_support_offset)
        verify_rate(self.incidence_rate)

        # build & verify hazard rate
        self.hazard_rate = self.build_hazard_rate(self.support, self.random_variable, self.incidence_rate,
                                             offset=self.rate_support_offset)
        verify_rate(self.hazard_rate)

    def build_parameters(self, parameters: Iterable[float]) -> NamedTuple:
        return self.__class__.Parameters._make(parameters)

    @abstractmethod
    def build_random_variable(self, parameters: NamedTuple) -> rv_frozen:
        raise NotImplementedError

    def build_support(self, max_rate_support_size: int = MAX_RATE_SUPPORT_SIZE,
                      freq: Union[float, int] = 1) -> np.ndarray:
        return np.arange(max_rate_support_size, step=freq)

    @abstractmethod
    def build_incidence_rate(self, support: np.ndarray, random_variable: rv_frozen, **kwargs) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def build_hazard_rate(self, support: np.ndarray, random_variable: rv_frozen, incidence_rate: pd.Series, **kwargs) -> pd.Series:
        raise NotImplementedError

    def plot_incidence(self, freq: float = 0.001, **kwargs):
        support = self.build_support(freq=freq, max_rate_support_size=self.max_rate_support_size)
        incidence_rate = self.build_incidence_rate(support, self.random_variable)

        self._plot_rate(self.incidence_rate, support, incidence_rate, **kwargs)

        plt.show()

    def plot_hazard(self, freq: float = 0.001, **kwargs):
        support = self.build_support(freq=freq, max_rate_support_size=self.max_rate_support_size)
        incidence_rate = self.build_incidence_rate(support, self.random_variable)
        hazard_rate = self.build_hazard_rate(support, self.random_variable, incidence_rate)

        self._plot_rate(self.hazard_rate, support, hazard_rate, label="Hazard", color="orange", **kwargs)

        plt.show()

    def _plot_rate(self, rate, hf_support, hf_rate, color: str = "blue", label: str = "Incidence", **kwargs):
        plt.gca()

        # plot high-frequency rates
        plt.plot(hf_support, hf_rate, label=label, c=color, alpha=0.5)

        # plot probability dots
        plt.hlines(rate, self.support, self.support + self.rate_support_offset, linestyles='--',
                   colors='red')

        plt.bar(self.support, rate, width=0.3, alpha=0.6, color=color)

        plt.legend()

    @property
    def dimensions(self) -> Collection[Dimension]:
        raise NotImplementedError

    def verify_parameters(self, parameters: NamedTuple) -> bool:
        for i, parameter_dims in enumerate(self.dimensions):
            if parameters[i] not in parameter_dims:
                raise InvalidParameterError

        return True

