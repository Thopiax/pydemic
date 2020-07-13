import pandas as pd
import numpy as np

from abc import ABC

from scipy.stats._distn_infrastructure import rv_frozen

from epydemic.outcome.distribution.base import BaseOutcomeDistribution


class DiscreteOutcomeDistribution(BaseOutcomeDistribution, ABC):
    def build_incidence_rate(self, support: np.ndarray, random_variable: rv_frozen, **kwargs) -> pd.Series:
        return pd.Series(
            random_variable.pmf(support),
            index=support,
            name="outcome"
        )

    def build_hazard_rate(self, support: np.ndarray, random_variable: rv_frozen, incidence_rate: pd.Series,
                          **kwargs) -> pd.Series:
        return pd.Series(
            incidence_rate / random_variable.sf(support),
            index=support,
            name="hazard"
        )
