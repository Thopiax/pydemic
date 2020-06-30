from collections import namedtuple

from scipy.stats import weibull_min, lognorm, nbinom
from scipy.stats._distn_infrastructure import rv_frozen

from skopt.space import Real, Integer


class LognormOutcomeDistribution(ContinuousOutcomeDistribution):
    Parameters = namedtuple("LognormParameters", ["mu", "sigma"])

    @property
    def dimensions(self):
        return [Real(-100.0, 100.0), Real(0.0, 100.0)]

    def build_random_variable(self, parameters: Parameters) -> rv_frozen:
        # parametrize lognorm in terms of the parameters of the characteristic normal distribution.
        return lognorm(parameters.sigma, scale=np.exp(parameters.mu))

