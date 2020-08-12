from collections import namedtuple

from scipy.stats import uniform
from scipy.stats._distn_infrastructure import rv_frozen

from skopt.space import Real, Integer

from src.outcome_lag.distributions.continuous import ContinuousOutcomeLagDistribution


class UniformOutcomeDistribution(ContinuousOutcomeLagDistribution):
    name = "Uniform"
    Parameters = namedtuple(name, ["p"])

    @property
    def dimensions(self):
        return [Real(0.0, 1.0)]

    def build_random_variable(self, parameters: Parameters) -> rv_frozen:
        return uniform(parameters.p)
