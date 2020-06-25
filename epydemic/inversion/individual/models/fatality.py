import numpy as np
import pandas as pd
from collections import namedtuple

from scipy.stats import weibull_min
from skopt.space import Real

from epydemic.inversion.individual.utils import build_hazard_rate, verify_valid_K, describe_rv

from epydemic.inversion.individual.models.base import BaseIndividualModel


class FatalityIndividualModel(BaseIndividualModel):
    parameter_dimensions = {
        "initial": [
            Real(0.01, 0.20),
            Real(0.5, 10.0),
            Real(1.0, 20.0)
        ], "relaxed": [
            Real(0.0, 1.0),
            Real(0.0, 10.0),
            Real(0.0, 50.0)
        ]
    }

    parameter_named_tuple = namedtuple("FatalityParameters", ["alpha", "beta", "eta"])

    @property
    def tag(self):
        return f"fatality_{self._dimensions_key}"

    def _build_model(self, max_ppf: int = 0.9999, max_K: int = 100):
        super()._build_model()

        self.rv = weibull_min(self.beta, scale=self.eta)

        # truncate up until (max_ppf * 100) percentile
        K = np.ceil(self.rv.ppf(max_ppf))

        # check that K is valid
        verify_valid_K(K)

        # ensure K is less than max_K
        self.K = min(int(K), max_K)

        self.fatality_rate, self.hazard_rate = self.build_incidence_rate(self.rv)

    def describe(self):
        super().describe()

        return describe_rv(self.rv)

