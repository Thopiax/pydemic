from abc import ABC, abstractmethod
from typing import Optional, List, NamedTuple

import numpy as np

from epydemic.inversion.individual.exceptions import InvalidParameters

from epydemic.inversion.individual.utils import build_hazard_rate


class BaseIndividualModel(ABC):
    parameter_dimensions = {}
    parameter_named_tuple = None

    def __init__(self, dimensions_key: str = "relaxed", parameters: Optional[List] = None):
        self._dimensions_key = dimensions_key
        self._dimensions = self.__class__.get_dimensions(dimensions_key)

        self._parameters: Optional[NamedTuple] = None
        self.K: int = -1

        if parameters is not None:
            self.parameters = parameters

    def __getattr__(self, item):
        if self.is_valid and item in self.parameters.keys():
            return self.parameters[item]

        raise AttributeError

    @property
    def dimensions(self):
        return self._dimensions

    @classmethod
    def build_parameter_named_tuple(cls, parameters):
        return cls.parameter_named_tuple(*parameters)

    @classmethod
    def get_dimensions(cls, dimensions_key):
        return cls.parameter_dimensions[dimensions_key]

    @property
    def parameters(self):
        if self._parameters is None:
            return None

        return self._parameters._asdict()

    @parameters.setter
    def parameters(self, parameters: List[float]):
        self._parameters = self.__class__.build_parameter_named_tuple(parameters)

        self._build_model()

    @property
    def is_valid(self):
        return self._parameters is not None

    def build_incidence_rate(self, rv):
        x = np.arange(self.K)

        incidence_rate = rv.cdf(x + 0.5) - rv.cdf(x - 0.5)
        hazard_rate = build_hazard_rate(incidence_rate)

        return incidence_rate, hazard_rate

    @property
    def tag(self):
        raise NotImplementedError

    @abstractmethod
    def _build_model(self, **kwargs):
        if self.is_valid is False:
            raise InvalidParameters

        pass

    @abstractmethod
    def describe(self):
        if self.is_valid is False:
            raise InvalidParameters

        pass
