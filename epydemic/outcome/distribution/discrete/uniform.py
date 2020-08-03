from collections import namedtuple

from scipy.stats import uniform
from scipy.stats._distn_infrastructure import rv_frozen

from skopt.space import Real, Integer

from epydemic.outcome.distribution.discrete.main import DiscreteOutcomeDistribution


class UniformOutcomeDistribution(DiscreteOutcomeDistribution):
    name = "Uniform"
    Parameters = namedtuple(name, ["p"])

    @property
    def dimensions(self):
        return [Real(0.0, 1.0)]

    def build_random_variable(self, parameters: Parameters) -> rv_frozen:
        return uniform(parameters.p)
