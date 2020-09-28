from collections import namedtuple

from scipy.stats import erlang

from skopt.space import Real, Integer

from src.resolution_delay.distributions.continuous.main import ContinuousResolutionDelayDistribution


class ErlangResolutionDelayDistribution(ContinuousResolutionDelayDistribution):
    _dist = erlang
    Parameters = namedtuple(_dist.name, ["k", "theta"])

    def __init__(self, *parameters):
        super().__init__(*parameters)

        self.support_offset = 0

    @property
    def dimensions(self):
        return [Integer(0, 1_000), Real(0.0, 1_000.0)]

    @property
    def shape(self):
        return self.parameters.k

    @property
    def scale(self):
        return self.parameters.theta

