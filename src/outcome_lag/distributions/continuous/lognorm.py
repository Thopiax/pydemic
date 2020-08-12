from collections import namedtuple

from scipy.stats import lognorm
from scipy.stats._distn_infrastructure import rv_frozen

from skopt.space import Real, Integer

from src.outcome_lag.distributions.continuous import ContinuousOutcomeDistribution


class LognormOutcomeDistribution(ContinuousOutcomeDistribution):
    name = "Lognormal"
    Parameters = namedtuple(name, ["mu", "sigma"])

    @property
    def dimensions(self):
        return [Real(-100.0, 100.0), Real(0.0, 100.0)]

    def build_random_variable(self, parameters: Parameters) -> rv_frozen:
        # parametrize lognorm in terms of the parameters of the characteristic normal distributions.
        return lognorm(parameters.sigma, scale=np.exp(parameters.mu))

