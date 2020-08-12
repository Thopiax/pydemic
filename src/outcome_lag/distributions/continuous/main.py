import pandas as pd
import numpy as np

from abc import ABC

from scipy.stats import rv_frozen

from outcome_lag.distributions.base import BaseOutcomeLagDistribution


class ContinuousOutcomeLagDistribution(BaseOutcomeLagDistribution, ABC):
    def build_incidence_rate(self, support: np.ndarray, random_variable: rv_frozen, offset: float = 0.0) -> pd.Series:
        return pd.Series(
            random_variable.pdf(support + offset),
            index=support,
            name="incidence"
        )

    def build_hazard_rate(self, support: np.ndarray, random_variable: rv_frozen, incidence_rate: pd.Series,
                          offset: float = 0.0) -> pd.Series:
        return pd.Series(
            incidence_rate / random_variable.sf(support + offset),
            index=support,
            name="hazard"
        )