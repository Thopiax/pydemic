from abc import ABC, abstractmethod
import numpy as np

from epydemic.inversion.individual.exceptions import InvalidParameters

from epydemic.inversion.individual.utils import build_hazard_rate


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

    @abstractmethod
    def describe(self):
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
