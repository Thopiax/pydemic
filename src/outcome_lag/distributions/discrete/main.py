from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC

from scipy.stats._distn_infrastructure import rv_frozen

from src.outcome_lag.distributions.base import BaseOutcomeLagDistribution


class DiscreteOutcomeLagDistribution(BaseOutcomeLagDistribution, ABC):
    def build_incidence_rate(self, support: np.ndarray, random_variable: rv_frozen, **kwargs) -> pd.Series:
        return pd.Series(
            random_variable.pmf(support),
            index=support,
            name="incidence"
        )

    def build_hazard_rate(self, support: np.ndarray, random_variable: rv_frozen, incidence_rate: pd.Series,
                          **kwargs) -> pd.Series:
        return pd.Series(
            incidence_rate / random_variable.sf(support),
            index=support,
            name="hazard"
        )

    def _plot_rate(self, rate, _support, _rate, color: str = "blue", label: str = "Incidence",
                   support_offset: Optional[float] = None, **kwargs):
        plt.gca()

        support_with_offset = self.support + (support_offset or self.support_offset)

        plt.bar(support_with_offset, rate, width=0.3, alpha=0.6, color=color, label=label, **kwargs)

        plt.legend()

