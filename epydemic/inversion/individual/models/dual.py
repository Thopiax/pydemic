import pandas as pd
import numpy as np
from collections import namedtuple

import scipy
from scipy.stats import weibull_min, lognorm
from skopt.space import Real

from epydemic.inversion.individual.utils import describe_rv, build_distribution_rates
from epydemic.inversion.individual.models.base import BaseIndividualModel


class DualIndividualModel(BaseIndividualModel):
    parameter_dimensions = {
        "initial": [
            Real(0.01, 0.20),
            Real(0.5, 10.0),
            Real(1.0, 20.0),
            Real(0.0, 10.0),
        ], "relaxed": [
            Real(0.0, 1.0),
            Real(0.0, 10.0),
            Real(0.0, 50.0),
            Real(0.0, 100.0),
        ]
    }

    parameter_named_tuple = namedtuple("DualParameters", ["alpha", "beta_f", "eta_f", "s"])

    def __init__(self, *args, fatality_distribution: scipy.stats = weibull_min, recovery_distribution: scipy.stats = lognorm, **kwargs):
        super().__init__(*args, **kwargs)

        self.fatality_distribution = fatality_distribution
        self.recovery_distribution = recovery_distribution

    @property
    def tag(self):
        return f"dual__{self.fatality_distribution.name}_{self.recovery_distribution.name}__{self._dimensions_key}"

    def _build_model(self, max_ppf: int = 0.9999):
        super()._build_model()

        self.fatality_rv = self.fatality_distribution(self._parameters.beta_f, scale=self._parameters.eta_f)
        self.fatality_rate, self.fatality_hazard_rate = build_distribution_rates(self.fatality_rv)

        self.recovery_rv = self.recovery_distribution(self._parameters.s)
        self.recovery_rate, self.recovery_hazard_rate = self.build_incidence_rate(self.recovery_rv)

        # self.K = min(len(self.fatality_rate), len(self.recovery_rate))
        #
        # # trim the rates to the
        # self.fatality_rate, self.fatality_hazard_rate = self.fatality_rate[:self.K], self.fatality_hazard_rate[:self.K]
        # self.recovery_rate, self.recovery_hazard_rate = self.recovery_rate[:self.K], self.recovery_hazard_rate[:self.K]

    def describe(self):
        super().describe()

        return pd.DataFrame({
            "Fatality": describe_rv(self.fatality_rv),
            "Recovery": describe_rv(self.recovery_rv)
        })