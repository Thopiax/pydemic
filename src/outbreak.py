import os
from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from src.utils.path import DATA_ROOTPATH
from utils.plot import save_figure


def _ffx_index(cumulative_x, threshold):
    x_max = cumulative_x.iloc[-1]

    if threshold > x_max:
        return None

    return np.argmax(cumulative_x > threshold)


class Outbreak:
    required_fields = ["cases", "deaths", "recoveries"]

    def __init__(self, region: str, cases: Optional[pd.Series] = None, deaths: Optional[pd.Series] = None,
                 recoveries: Optional[pd.Series] = None, df: Optional[pd.DataFrame] = None, smoothing_window: int = 3, **kwargs):

        self.region = region

        if (cases is None) and (deaths is None) and (recoveries is None):
            assert df is not None
            assert all(field in df.columns for field in Outbreak.required_fields)

            self._df: pd.DataFrame = df
        else:
            self._df: pd.DataFrame = pd.DataFrame({"cases": cases, "deaths": deaths, "recoveries": recoveries, **kwargs})

            # join with df if attribute present
            if df is not None:
                self._df = self._df.join(df)

        # construct the cumulative columns if they are not present in the df
        for field in Outbreak.required_fields:
            self._df[f"cumulative_{field}"] = self._df[field].cumsum()

        # must be set after df is created
        self.smoothing_window = smoothing_window

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(self, item):
        if type(item) is slice:
            return Outbreak(
                self.region,
                df=self._df.iloc[item]
            )

        return getattr(self, item)

    def __getattr__(self, item):
        if item in self._df.columns:
            return self._df[item]

        if item.startswith("smooth"):
            item = item[item.find("_") + 1:]
            return self._smooth_df[item]

        return self.__getattribute__(item)

    @cached_property
    def start(self):
        return self._df.index.min()

    @cached_property
    def cumulative_resolved_cases(self):
        return self.cumulative_deaths + self.cumulative_recoveries

    @cached_property
    def df(self):
        return self._df

    @cached_property
    def peak_date(self):
        # extract smooth peak idx (normalize daily anomalies due to changes in counting strategies...)
        smooth_peak_idx, _ = find_peaks(self.smooth_deaths, prominence=1)

        if len(smooth_peak_idx) > 0:
            if len(smooth_peak_idx) == 1:
                smooth_peak_id = smooth_peak_idx[0]
            else:
                smooth_peak_idxmax = self.smooth_deaths.iloc[smooth_peak_idx].idxmax()
                smooth_peak_id = self.smooth_deaths.index.get_loc(smooth_peak_idxmax)

            peak_idxmax = self.deaths.iloc[
                          max(smooth_peak_id - self.smoothing_window, 0):min(smooth_peak_id + self.smoothing_window,
                                                                             len(self) - 1)].idxmax()
            peak_id = self.deaths.index.get_loc(peak_idxmax)

            if self._verify_peak(peak_id):
                return peak_id

        return -1

    @cached_property
    def is_peak_reached(self):
        return self.peak_date != -1

    def _verify_peak(self, peak_id, min_gradient_since_peak_threshold=0.05, min_days_since_peak_threshold=7):
        peak_deaths = self.smooth_deaths.iloc[peak_id]
        last_deaths = self.smooth_deaths.iloc[-1]

        gradient_since_peak = (peak_deaths - last_deaths) / peak_deaths

        days_since_peak = (self.smoothed_deaths.shape[0] - 1) - peak_id

        return days_since_peak > min_days_since_peak_threshold and gradient_since_peak > min_gradient_since_peak_threshold

    @property
    def smoothing_window(self):
        return self._smoothing_window

    @smoothing_window.setter
    def smoothing_window(self, smoothing_window):
        assert 0 < smoothing_window < 60

        self._smoothing_window = smoothing_window
        self._smooth_df = self._build_smooth_df(smoothing_window)

    def _build_smooth_df(self, smoothing_window):
        return self.df.copy().rolling(smoothing_window, center=True).mean().dropna()

    def expanding_cutoffs(self, start: int = 0, window_size: int = 7, ffx: int = 10) -> np.ndarray:
        burn_in = self.ffx(ffx, x_type="deaths")

        return np.arange(burn_in + start, len(self), window_size)

    def ffx(self, *xs, x_type="cases"):
        cumulative_x = self._df[f"cumulative_{x_type}"]

        if len(xs) > 1:
            return [_ffx_index(cumulative_x, x) for x in xs]

        return _ffx_index(cumulative_x, xs[0])

    @save_figure(lambda outbreak: f"outbreaks/{outbreak.region}.pdf")
    def plot(self):
        ax = plt.gca()

        plt.suptitle(self.region)

        ax.set_ylabel("# of people")

        self.cases.plot(ax=ax, label="cases")
        self.deaths.plot(ax=ax, label="deaths")
        self.recoveries.plot(ax=ax, label="recoveries")

        plt.legend()

        plt.show()

    @staticmethod
    def from_simulation(region, simulation, dt: float = 1.0):
        observation_index = np.arange(0, max(simulation.index), dt)

        def observe(col):
            return simulation.loc[observation_index, col].diff().fillna(0).astype(int)

        # perfect visibility
        deaths = observe("D")
        recoveries = observe("R")
        cases = abs(observe("S"))

        return Outbreak(
            region,
            cases=cases,
            deaths=deaths,
            recoveries=recoveries
        )

    @staticmethod
    def from_csv(region, epidemic="covid"):
        filepath = DATA_ROOTPATH / f"{epidemic}/{region}.csv"

        if os.path.isfile(filepath):
            outbreak_df = pd.read_csv(filepath, index_col=0, parse_dates=[0])

            return Outbreak(region, df=outbreak_df)

        raise Exception(f"File for {region} not found.")

    def to_csv(self, epidemic="covid"):
        if os.path.exists(DATA_ROOTPATH / f"{epidemic}") is False:
            os.makedirs(DATA_ROOTPATH / f"{epidemic}")

        self._df.to_csv(DATA_ROOTPATH / f"{epidemic}/{self.region}.csv")
        self._smooth_df.to_csv(DATA_ROOTPATH / f"{epidemic}/{self.region}_s{self.smoothing_window}.csv")

