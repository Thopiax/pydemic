from collections import namedtuple

from scipy.stats import weibull_min
from scipy.stats._distn_infrastructure import rv_frozen

from skopt.space import Real

from src.outcome_lag.distributions.continuous.main import ContinuousOutcomeLagDistribution


class WeibullOutcomeDistribution(ContinuousOutcomeLagDistribution):
    name = "Weibull"
    Parameters = namedtuple(name, ["beta", "eta"])

    @property
    def dimensions(self):
        return [Real(0.0, 100.0), Real(0.0, 100.0)]

    def build_random_variable(self, parameters: Parameters) -> rv_frozen:
        return weibull_min(parameters.beta, scale=parameters.beta)


