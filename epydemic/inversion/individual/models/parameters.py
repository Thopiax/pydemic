from abc import ABC, abstractmethod
import numpy as np
from collections import namedtuple

from typing import Optional, NamedTuple, List

from scipy.stats import weibull_min
from skopt.space import Real

from epydemic.inversion.individual.exceptions import InvalidParameters
from epydemic.inversion.individual.utils import build_hazard_rate, verify_valid_K


class AbstractIndividualModel(ABC):
    parameter_dimensions = {}
    parameter_named_tuple = None

    def __init__(self, dimensions_key: str = "relaxed", **parameters):
        self._dimensions = self.__class__.get_dimensions(dimensions_key)

        self._parameters = self.__class__.build_parameter_named_tuple(**parameters)
        self.K = None

        self._build_model()

    @abstractmethod
    def _build_model(self, **kwargs):
        if self.is_valid is False:
            raise InvalidParameters

        pass

    @classmethod
    def build_parameter_named_tuple(cls, **parameters):
        # if no parameters were provided, return None
        if len(parameters) == 0:
            return None

        return cls.parameter_named_tuple(**parameters)

    @classmethod
    def get_dimensions(cls, dimensions_key):
        return cls.parameter_dimensions[dimensions_key]

    def is_valid(self):
        return self._parameters is None

    def build_incidence_rate(self, rv):
        x = np.arange(self.K)

        incidence_rate = rv.cdf(x + 0.5) - rv.cdf(x - 0.5)
        hazard_rate = build_hazard_rate(incidence_rate)

        return incidence_rate, hazard_rate


class FatalityIndividualModel(AbstractIndividualModel):

    parameter_dimensions = {
        "initial": [
            Real(0.01, 0.20),
            Real(0.5, 10.0),
            Real(1.0, 20.0)
        ], "relaxed": [
            Real(0.0, 1.0),
            Real(0.0, 10.0),
            Real(0.0, 50.0)
        ]
    }

    parameter_named_tuple = namedtuple("FatalityParameters", ["alpha", "beta", "eta"])

    def _build_model(self, max_ppf: int = 0.9999, max_K: int = 100):
        super()._build_model()

        self.rv = weibull_min(self._parameters.beta, scale=self._parameters.eta)

        # truncate up until (max_ppf * 100) percentile
        K = np.ceil(self.rv.ppf(max_ppf))

        # check that K is valid
        verify_valid_K(K)

        # ensure K is less than max_K
        self.K = min(int(K), max_K)

        self.mortality_rate, self.hazard_rate = self.build_incidence_rate(self.rv)


class DualIndividualModel(AbstractIndividualModel):

    parameter_dimensions = {
        "initial": [
            Real(0.01, 0.20),
            Real(0.5, 10.0),
            Real(1.0, 20.0),
            Real(0.5, 10.0),
            Real(1.0, 20.0)
        ], "relaxed": [
            Real(0.0, 1.0),
            Real(0.0, 10.0),
            Real(0.0, 50.0),
            Real(0.0, 10.0),
            Real(0.0, 50.0)
        ]
    }

    parameter_named_tuple = namedtuple("DualParameters", ["alpha", "beta_f", "eta_f", "beta_r", "eta_r"])

    def _build_model(self, max_ppf: int = 0.9999, max_K: int = 100):
        super()._build_model()

        self.mortality_rv = weibull_min(self._parameters.beta_f, scale=self._parameters.eta_f)
        self.recovery_rv = weibull_min(self._parameters.beta_r, scale=self._parameters.eta_r)

        # truncate up until (max_ppf * 100) percentile
        K = np.ceil(max(self.mortality_rv.ppf(max_ppf), self.recovery_rv.ppf(max_ppf)))

        # check that K is valid
        verify_valid_K(K)

        # ensure K is less than max_K
        self.K = min(int(K), max_K)

        self.mortality_rate, self.mortality_hazard_rate = self.build_incidence_rate(self.mortality_rv)
        self.recovery_rate, self.recovery_hazard_rate = self.build_incidence_rate(self.recovery_rv)

