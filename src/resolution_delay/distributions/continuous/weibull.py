from collections import namedtuple
from functools import cached_property

from scipy.stats import weibull_min

from skopt.space import Real

from src.resolution_delay.distributions.continuous.main import ContinuousResolutionDelayDistribution


class WeibullResolutionDelayDistribution(ContinuousResolutionDelayDistribution):
    _dist = weibull_min
    Parameters = namedtuple(_dist.name, ["beta", "eta"])

    @property
    def dimensions(self):
        return [Real(0.0, 100.0), Real(0.0, 100.0)]

    @property
    def scale(self):
        return self.parameters.beta

    @property
    def shape(self):
        return self.parameters.eta

