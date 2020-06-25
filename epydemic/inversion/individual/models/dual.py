import pandas as pd
import numpy as np
from collections import namedtuple

from scipy.stats import weibull_min
from skopt.space import Real

from epydemic.inversion.individual.utils import verify_valid_K, describe_rv
from epydemic.inversion.individual.models.base import BaseIndividualModel


class DualIndividualModel(BaseIndividualModel):
    parameter_dimensions = {
        "initial": [
            Real(0.01, 0.20),
            Real(0.5, 10.0),
            Real(1.0, 20.0),
            Real(0.5, 10.0),
            Real(1.0, 20.0)
        ], "relaxed": [
            Real(0.0, 1.0),
            Real(0.0, 10.0),
            Real(0.0, 50.0),
            Real(0.0, 10.0),
            Real(0.0, 50.0)
        ]
    }

    parameter_named_tuple = namedtuple("DualParameters", ["alpha", "beta_f", "eta_f", "beta_r", "eta_r"])

    @property
    def tag(self):
        return f"dual_{self._dimensions_key}"

    def _build_model(self, max_ppf: int = 0.9999, max_K: int = 100):
        super()._build_model()

        self.fatality_rv = weibull_min(self._parameters.beta_f, scale=self._parameters.eta_f)
        self.recovery_rv = weibull_min(self._parameters.beta_r, scale=self._parameters.eta_r)

        # truncate up until (max_ppf * 100) percentile
        K = np.ceil(max(self.fatality_rv.ppf(max_ppf), self.recovery_rv.ppf(max_ppf)))

        # check that K is valid
        verify_valid_K(K)

        # ensure K is less than max_K
        self.K = min(int(K), max_K)

        self.fatality_rate, self.fatality_hazard_rate = self.build_incidence_rate(self.fatality_rv)
        self.recovery_rate, self.recovery_hazard_rate = self.build_incidence_rate(self.recovery_rv)

    def describe(self):
        super().describe()

        return pd.DataFrame({
            "Fatality": describe_rv(self.fatality_rv),
            "Recovery": describe_rv(self.recovery_rv)
        })