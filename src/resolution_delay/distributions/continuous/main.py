from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC

from scipy.stats import rv_frozen

from resolution_delay.distributions.base import BaseResolutionDelayDistribution


class ContinuousResolutionDelayDistribution(BaseResolutionDelayDistribution, ABC):
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

    def _plot_rate(self, rate, hf_support, hf_rate, color: str = "blue", label: str = "Incidence", support_offset: Optional[float] = None, **kwargs):
        plt.gca()

        support = self.support
        support_with_offset = support + (support_offset or self.support_offset)

        # plot high-frequency rates
        plt.plot(hf_support, hf_rate, label=label, c=color, alpha=0.5)

        # # plot probability dots
        plt.hlines(rate, support, support_with_offset, linestyles='--',
                   colors='red')

        plt.bar(support, rate, width=0.3, alpha=0.6, color=color)

        plt.legend()