from collections import namedtuple

from scipy.stats import uniform
from scipy.stats._distn_infrastructure import rv_frozen

from skopt.space import Real

from src.resolution_delay.distributions.continuous.main import ContinuousResolutionDelayDistribution


class UniformResolutionDelayDistribution(ContinuousResolutionDelayDistribution):
    _dist = uniform
    Parameters = namedtuple(_dist.name, ["p"])

    @property
    def dimensions(self):
        return [Real(0.0, 1.0)]

    @property
    def shape(self):
        return self._parameters.p
