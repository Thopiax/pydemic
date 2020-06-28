from typing import Optional

import numpy as np
import pandas as pd
from collections import namedtuple

import scipy
from scipy.stats import weibull_min
from skopt.space import Real

from epydemic.inversion.individual.utils import describe_rv, verify_distribution, \
    build_distribution_rates

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

    def __init__(self, *args, distribution: scipy.stats = weibull_min, **kwargs):
        super().__init__(*args, **kwargs)

        self.distribution = distribution

    @property
    def tag(self):
        return f"fatality__{self.distribution.name}__{self._dimensions_key}"

    def _build_model(self, max_ppf: int = 0.9999):
        super()._build_model()

        self.rv = self.distribution(self.beta, scale=self.eta)

        self.fatality_rate, self.hazard_rate = build_distribution_rates(self.rv)

    def describe(self):
        super().describe()

        return describe_rv(self.rv)

