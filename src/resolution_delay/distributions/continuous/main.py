from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC

from scipy.stats._distn_infrastructure import rv_frozen

from resolution_delay.distributions.base import BaseResolutionDelayDistribution


class ContinuousResolutionDelayDistribution(BaseResolutionDelayDistribution, ABC):
    @property
    def shape(self):
        raise NotImplementedError

    @property
    def max_ppf(self):
        return int(np.ceil(self.__class__._dist.ppf(self.max_rate_ppf, self.shape, scale=self.scale)))

    def build_incidence_rate(self, support: np.ndarray, offset: float = 0.0) -> pd.Series:
        return pd.Series(
            self.__class__._dist.pdf(support + offset, self.shape, scale=self.scale),
            index=support,
            name="incidence"
        )

    def build_hazard_rate(self, support: np.ndarray, incidence_rate: pd.Series, offset: float = 0.0) -> pd.Series:
        return pd.Series(
            incidence_rate / self.__class__._dist.sf(support + offset, self.shape, scale=self.scale),
            index=support,
            name="hazard"
        )

    def _plot_rate(self, rate, hf_support, hf_rate, color: str = "blue", label: str = "Incidence",
                   support_offset: Optional[float] = None, **kwargs):
        plt.gca()

        support = self.support
        support_with_offset = support + (support_offset or self.support_offset)

        # plot high-frequency rates
        plt.plot(hf_support, hf_rate, label=label, c=color, alpha=0.5)

        # # plot probability dots
        plt.hlines(rate, support, support_with_offset, linestyles='--', colors='red')

        plt.bar(support, rate, width=0.3, alpha=0.6, color=color)

        plt.legend()