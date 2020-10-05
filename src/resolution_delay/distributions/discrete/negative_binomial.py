from collections import namedtuple

from scipy.stats import nbinom
from scipy.stats._distn_infrastructure import rv_frozen

from skopt.space import Real, Integer

from src.resolution_delay.distributions.discrete.main import DiscreteResolutionDelayDistribution


class NegativeBinomialResolutionDelayDistribution(DiscreteResolutionDelayDistribution):
    _dist = nbinom
    Parameters = namedtuple(_dist.name, ["n", "p"])

    @property
    def dimensions(self):
        return [Integer(0, 100), Real(0.0, 1.0)]
