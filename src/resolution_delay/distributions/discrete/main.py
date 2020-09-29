from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC

from scipy.stats._distn_infrastructure import rv_frozen

from src.resolution_delay.distributions.base import BaseResolutionDelayDistribution


class DiscreteResolutionDelayDistribution(BaseResolutionDelayDistribution, ABC):

    def build_incidence_rate(self, support: np.ndarray, **kwargs) -> pd.Series:
        return pd.Series(
            self._rv.pmf(support),
            index=support,
            name="incidence"
        )

    def build_hazard_rate(self, support: np.ndarray, incidence_rate: pd.Series, **kwargs) -> pd.Series:
        return pd.Series(
            incidence_rate / self._rv.sf(support),
            index=support,
            name="hazard"
        )

    def build_random_variable(self) -> rv_frozen:
        return self.__class__._dist(*self.parameters)

    def _plot_rate(self, rate, _support, _rate, color: str = "blue", label: str = "Incidence",
                   support_offset: Optional[float] = None, **kwargs):
        plt.gca()

        support_with_offset = self.support + (support_offset or self.support_offset)

        plt.bar(support_with_offset, rate, width=0.3, alpha=0.6, color=color, label=label, **kwargs)

        plt.legend()

