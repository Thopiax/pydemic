import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.stats import weibull_min
import matplotlib.pyplot as plt

from epydemic.inversion.exceptions import InvalidMortalityRateException


class IndividualFatalityModel:
    def __init__(self, parameters: Optional[Tuple[float, float, float]] = (None, None, None)):
        self.alpha = parameters[0]
        self.beta = parameters[1]
        self.lam = parameters[2]

        self.K = 0
        self.mortality_rate = np.ndarray([])
        self.hazard_rate = np.ndarray([])

        if self.none_parameters is not True:
            self.build_model()

    def build_model(self, max_ppf: int = 0.999, max_K: int = 100):
        if self.none_parameters: raise InvalidMortalityRateException

        self.weibull_rv = weibull_min(self.beta, scale=self.lam)
        weibull_ppf = self.weibull_rv.ppf(max_ppf)

        # ppf should be defined
        if np.isnan(weibull_ppf) or np.isinf(weibull_ppf):
            raise InvalidMortalityRateException

        # K is the span of the distribution that covers max_ppf % of cases
        K = int(weibull_ppf)

        # K must be at most the max_K described above
        K = min(K, max_K)

        # K must be positive for the _legacy to be valid (have at least one day)
        if K <= 0:
            raise InvalidMortalityRateException

        self.K = K
        x = np.arange(self.K)

        # construct discrete mortality rate
        self.mortality_rate = self.weibull_rv.cdf(x + 0.5) - self.weibull_rv.cdf(x - 0.5)
        cumulative_fatality_density = np.cumsum(self.mortality_rate)

        # construct discrete hazard rate
        self.hazard_rate = np.zeros(self.K)
        self.hazard_rate[0] = self.mortality_rate[0]
        self.hazard_rate[1:] = self.mortality_rate[1:] / (1 - cumulative_fatality_density[:-1])

    @classmethod
    def from_records(cls, records):
        return pd.Series(
            [cls(parameters=row[["alpha", "beta", "lambda"]]) for _, row in records.iterrows()],
            index=records["region"]
        )

    @property
    def none_parameters(self):
        return self.parameters == (None, None, None)

    @property
    def parameters(self):
        return self.alpha, self.beta, self.lam

    @parameters.setter
    def parameters(self, parameters: Tuple[Optional[float], Optional[float], Optional[float]]):
        # noop if new parameters match current ones.
        if self.parameters == parameters:
            return

        self.alpha, self.beta, self.lam = parameters
        self.build_model()

    def describe(self, region=None):
        if self.weibull_rv is None:
            raise InvalidMortalityRateException

        mean, var, skew, kurtosis = self.weibull_rv.stats(moments="mvsk")

        return pd.Series(dict(
            mean=mean,
            median=self.weibull_rv.median(),
            std=self.weibull_rv.std(),
            variance=var,
            skew=skew,
            kurtosis=kurtosis,
            entropy=self.weibull_rv.entropy(),
            interval90=self.weibull_rv.interval(0.90)
        ), name=region)
