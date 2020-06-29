import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod
from collections import namedtuple

import scipy
from scipy.stats import weibull_min, lognorm
from scipy.stats._distn_infrastructure import rv_frozen
from typing import Optional, List, NamedTuple, Iterable

from skopt.space import Real

from epydemic.inversion.individual.utils import build_distribution_rates
from inversion.individual.exceptions import InvalidParameters

MAX_RATE_PPF = 0.999 # only 1 in 1000 cases are not considered
MAX_RATE_SUPPORT_SIZE = 60 # days
MAX_RATE_VARIANCE = 1000 # days => std < 30 days


def verify_random_variable(random_variable: rv_frozen):
    if random_variable.median() > MAX_RATE_VARIANCE:
        raise InvalidParameters


def verify_rate(rate: pd.Series):
    if np.isnan(rate).any() or np.isinf(rate).any():
        raise InvalidParameters


def describe(random_variable: rv_frozen):
    mean, var, skew, kurtosis = random_variable.stats(moments="mvsk")

    interval_size = 0.95
    lower, upper = random_variable.interval(interval_size)

    return pd.Series(dict(
        mean=float(mean),
        std=random_variable.std(),
        variance=float(var),
        skew=float(skew),
        kurtosis=float(kurtosis),
        median=random_variable.median(),
        entropy=random_variable.entropy(),
        lower_interval_bound=lower,
        upper_interval_bound=upper,
        interval_size=interval_size
    ))


def build_support(max_rate_support_size: int = MAX_RATE_SUPPORT_SIZE, freq: float = 1.0) -> np.ndarray:
    return np.arange(max_rate_support_size, step=freq)


def build_incidence_rate(support: np.ndarray, random_variable: rv_frozen, offset: float = 0.0) -> pd.Series:
    return pd.Series(
        random_variable.pdf(support + offset),
        index=support,
        name="incidence"
    )


def build_hazard_rate(support: np.ndarray, random_variable: rv_frozen, incidence_rate: pd.Series, offset: float = 0.0) -> pd.Series:
    return pd.Series(
        incidence_rate / random_variable.sf(support + offset),
        index=support,
        name="hazard"
    )


class IncidenceRate(ABC):
    Parameters: Optional[NamedTuple] = None

    def __init__(self, *parameters, max_rate_support_size: int = MAX_RATE_SUPPORT_SIZE, max_rate_ppf: int = MAX_RATE_PPF, rate_support_offset: float = 0.5):
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

    @property
    def parameters(self):
        if self._parameters is None:
            return None

        return self._parameters._asdict()

    @parameters.setter
    def parameters(self, parameters: Iterable[float]) -> None:
        # build & verify parameters
        self._parameters = self.build_parameters(parameters)
        self.verify_parameters(self._parameters)

        # build & verify random variable
        self.random_variable = self.build_random_variable(self._parameters)
        verify_random_variable(self.random_variable)

        self.support = build_support(max_rate_support_size=self.max_rate_support_size)

        # build & verify incidence rate
        self.incidence_rate = build_incidence_rate(self.support, self.random_variable, offset=self.rate_support_offset)
        verify_rate(self.incidence_rate)

        # build & verify hazard rate
        self.hazard_rate = build_hazard_rate(self.support, self.random_variable, self.incidence_rate,
                                             offset=self.rate_support_offset)
        verify_rate(self.hazard_rate)

    def build_parameters(self, parameters: Iterable[float]) -> NamedTuple:
        return self.__class__.Parameters._make(parameters)

    def plot_incidence(self, freq: float = 0.001, **kwargs):
        support = build_support(freq=freq, max_rate_support_size=self.max_rate_support_size)
        incidence_rate = build_incidence_rate(support, self.random_variable)

        self._plot_rate(self.incidence_rate, support, incidence_rate, **kwargs)

        plt.show()

    def plot_hazard(self, freq: float = 0.001, **kwargs):
        support = build_support(freq=freq, max_rate_support_size=self.max_rate_support_size)
        incidence_rate = build_incidence_rate(support, self.random_variable)
        hazard_rate = build_hazard_rate(support, self.random_variable, incidence_rate)

        self._plot_rate(self.hazard_rate, support, hazard_rate, label="Hazard", color="orange", **kwargs)

        plt.show()

    def _plot_rate(self, rate, hf_support, hf_rate, color: str = "blue",  label: str = "Incidence", **kwargs):
        plt.gca()

        # plot high-frequency rates
        plt.plot(hf_support, hf_rate, label=label, c=color, alpha=0.5)

        # plot probability dots
        plt.hlines(rate, self.support, self.support + self.rate_support_offset, linestyles='--',
                   colors='red')

        plt.bar(self.support, rate, width=0.3, alpha=0.6, color=color)

        plt.legend()

    @abstractmethod
    def dimensions(self):
        raise NotImplementedError

    @abstractmethod
    def build_random_variable(self, parameters: NamedTuple) -> rv_frozen:
        raise NotImplementedError

    @abstractmethod
    def verify_parameters(self, parameters: NamedTuple) -> bool:
        raise NotImplementedError


class LognormIncidenceRate(IncidenceRate):
    Parameters = namedtuple("LognormParameters", ["mu", "sigma"])

    @property
    def dimensions(self):
        return [Real(-100.0, 100.0), Real(0.0, 100.0)]

    def build_random_variable(self, parameters: Parameters) -> rv_frozen:
        # parametrize lognorm in terms of the parameters of the characteristic normal distribution.
        return lognorm(parameters.sigma, scale=np.exp(parameters.mu))

    def verify_parameters(self, parameters: Parameters) -> bool:
        return True


class WeibullIncidenceRate(IncidenceRate):
    Parameters = namedtuple("WeibullParameters", ["beta", "eta"])

    @property
    def dimensions(self):
        return [Real(0.0, 100.0), Real(0.0, 100.0)]

    def verify_parameters(self, parameters: Parameters) -> bool:
        return True

    def build_random_variable(self, parameters: Parameters) -> rv_frozen:
        return weibull_min(parameters.beta, scale=parameters.beta)

