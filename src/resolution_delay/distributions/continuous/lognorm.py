import numpy as np

from collections import namedtuple

from scipy.stats import lognorm

from skopt.space import Real

from src.resolution_delay.distributions.continuous.main import ContinuousResolutionDelayDistribution


class LognormResolutionDelayDistribution(ContinuousResolutionDelayDistribution):
    _dist = lognorm
    Parameters = namedtuple(_dist.name, ["mu", "sigma"])

    @property
    def dimensions(self):
        return [Real(-100.0, 100.0), Real(0.0, 100.0)]

    @property
    def shape(self):
        return self._parameters.sigma

    @property
    def shape(self):
        # parametrize lognorm in terms of the parameters of the characteristic normal distributions.
        return np.exp(self._parameters.mu)

