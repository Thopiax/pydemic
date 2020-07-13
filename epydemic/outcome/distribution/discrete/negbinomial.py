from collections import namedtuple

from scipy.stats import nbinom
from scipy.stats._distn_infrastructure import rv_frozen

from skopt.space import Real, Integer

from epydemic.outcome.distribution.discrete.main import DiscreteOutcomeDistribution


class NegBinomialOutcomeDistribution(DiscreteOutcomeDistribution):
    name = "NegBinomial"
    Parameters = namedtuple(name, ["n", "p"])

    @property
    def dimensions(self):
        return [Integer(0, 1_000), Real(0.0, 1.0)]

    def build_random_variable(self, parameters: Parameters) -> rv_frozen:
        return nbinom(parameters.r, parameters.p)
