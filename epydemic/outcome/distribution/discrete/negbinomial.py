from collections import namedtuple

from scipy.stats import nbinom
from scipy.stats._distn_infrastructure import rv_frozen

from skopt.space import Real

from epydemic.outcome.distribution.discrete.main import DiscreteOutcomeDistribution


class NegBinomialOutcomeDistribution(DiscreteOutcomeDistribution):
    Parameters = namedtuple("NegBinomialParameters", ["r", "p"])

    @property
    def dimensions(self):
        return [Real(0, 100.0), Real(0.0, 1.0)]

    def build_random_variable(self, parameters: Parameters) -> rv_frozen:
        return nbinom(parameters.r, parameters.p)
