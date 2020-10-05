import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

from scipy.stats._distn_infrastructure import rv_frozen
from typing import Optional, NamedTuple, Iterable, List

from skopt.space import Dimension

from src.resolution_delay.distributions.utils import MAX_RATE_PPF, MAX_SUPPORT_SIZE, verify_rate, describe


class BaseResolutionDelayDistribution(ABC):
    _dist = None
    Parameters: Optional[NamedTuple] = None

    def __init__(self, *parameters, max_support_size: int = MAX_SUPPORT_SIZE,
                 max_rate_ppf: int = MAX_RATE_PPF, support_offset: float = 0.5):
        self._parameters: Optional[NamedTuple] = None
        self._rv: Optional[rv_frozen] = None

        self.support: Optional[np.ndarray] = None
        self.support_offset = support_offset

        self.max_support_size = max_support_size
        self.max_rate_ppf = max_rate_ppf

        self.incidence_rate: Optional[pd.Series] = None

        self.hazard_rate: Optional[pd.Series] = None

        if len(parameters) > 0:
            self.parameters = parameters

    def describe(self):
        return describe(self._rv) if self._rv is not None else None

    @property
    def max_ppf(self):
        ppf = self._rv.ppf(self.max_rate_ppf)

        if np.isnan(ppf) or np.isinf(ppf):
            return self.max_support_size

        return max(int(np.ceil(ppf)), 1)

    @property
    def scale(self):
        return 1.0

    @property
    def name(self):
        return self.__class__._dist.name

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
    def parameters(self, parameters: Iterable[float], build_hazard_rate: bool = False):
        # build & verify parameters
        self._parameters = self.build_parameters(parameters)
        self._rv = self.build_random_variable()

        self.support = self.build_support()

        # build & verify resolution_delay rate
        self.incidence_rate = self.build_incidence_rate(self.support, offset=self.support_offset)
        verify_rate(self.incidence_rate)

        if build_hazard_rate:
            # build & verify hazard rate
            self.hazard_rate = self.build_hazard_rate(self.support, self.incidence_rate, offset=self.support_offset)
            verify_rate(self.hazard_rate)

    def build_parameters(self, parameters: Iterable[float]) -> NamedTuple:
        return self.__class__.Parameters._make(parameters)

    def build_support(self, freq: float = 1.0):
        support_size = min(self.max_support_size, self.max_ppf)

        return np.arange(support_size, step=freq)

    def plot_incidence(self, freq: float = 0.001, **kwargs):
        support = self.build_support(freq=freq)
        incidence_rate = self.build_incidence_rate(support)

        self._plot_rate(self.incidence_rate, support, incidence_rate, **kwargs)

    def plot_hazard(self, freq: float = 0.001, **kwargs):
        support = self.build_support(freq=freq)
        incidence_rate = self.build_incidence_rate(support)
        hazard_rate = self.build_hazard_rate(support, incidence_rate)

        self._plot_rate(self.hazard_rate, support, hazard_rate, label="Hazard", color="orange", **kwargs)

    @abstractmethod
    def build_random_variable(self):
        pass

    @abstractmethod
    def _plot_rate(self, rate, hf_support, hf_rate, color: str = "blue", label: str = "Incidence",
                   support_offset: Optional[float] = None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def build_incidence_rate(self, support: np.ndarray, **kwargs) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def build_hazard_rate(self, support: np.ndarray, incidence_rate: pd.Series, **kwargs) -> pd.Series:
        raise NotImplementedError

    @property
    def dimensions(self) -> List[Dimension]:
        raise NotImplementedError


