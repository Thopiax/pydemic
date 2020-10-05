from collections import namedtuple

from scipy.stats import hypergeom
from scipy.stats._distn_infrastructure import rv_frozen

from skopt.space import Real, Integer

from src.resolution_delay.distributions.discrete.main import DiscreteResolutionDelayDistribution


class HypergeometricResolutionDelayDistribution(DiscreteResolutionDelayDistribution):
    _dist = hypergeom
    Parameters = namedtuple(_dist.name, ["N", "K", "n"])

    @property
    def dimensions(self):
        return [Integer(0, 10_000), Integer(0, 10_000), Integer(0, 10_000)]
