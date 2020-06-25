import os
from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .time_window import OutbreakTimeWindow


def _ffx_index(cumulative_x, threshold):
    x_max = cumulative_x.iloc[-1]

    if threshold > x_max:
        return None

    return np.argmax(cumulative_x > threshold)


class Outbreak:
    def __init__(self, region: str, cases: pd.Series, deaths: pd.Series, recoveries: pd.Series,
                 smoothing_window: int = 3, **data):
        self.region = region

        self.cases = cases
        self.cases.name = region
        self.cumulative_cases = self.cases.cumsum()

        self.deaths = deaths
        self.deaths.name = region
        self.cumulative_deaths = self.deaths.cumsum()

        self.recoveries = recoveries
        self.recoveries.name = region
        self.cumulative_recoveries = self.recoveries.cumsum()

        self.smoothing_window = smoothing_window

        self._secondary_data = data

        # set all items in data as attributes
        for item, value in data.items():
            setattr(self, item, value)

    def __len__(self):
        return self.cases.shape[0]

    @property
    def smoothing_window(self):
        return self._smoothing_window

    @smoothing_window.setter
    def smoothing_window(self, smoothing_window):
        assert 0 < smoothing_window < 60

        self._smoothing_window = smoothing_window

        self.smoothed_cases = self._apply_smoothing(self.cases)
        self.smoothed_deaths = self._apply_smoothing(self.deaths)
        self.smoothed_recoveries = self._apply_smoothing(self.recoveries)

    def _apply_smoothing(self, sequence):
        return sequence.rolling(self.smoothing_window, center=True).mean().dropna()

    @property
    def resolved_case_rate(self):
        return (self.cumulative_deaths + self.cumulative_recoveries) / self.cumulative_cases

    @cached_property
    def peak_id(self):
        # extract smooth peak idx (normalize daily anomalies due to changes in counting strategies...)
        smooth_peak_idx, _ = find_peaks(self.smoothed_deaths, prominence=1)

        if len(smooth_peak_idx) > 0:
            if len(smooth_peak_idx) == 1:
                smooth_peak_id = smooth_peak_idx[0]
            else:
                smooth_peak_idxmax = self.smoothed_deaths.iloc[smooth_peak_idx].idxmax()
                smooth_peak_id = self.smoothed_deaths.index.get_loc(smooth_peak_idxmax)

            peak_idxmax = self.deaths.iloc[
                          max(smooth_peak_id - self.smoothing_window, 0):min(smooth_peak_id + self.smoothing_window,
                                                                             len(self) - 1)].idxmax()
            peak_id = self.deaths.index.get_loc(peak_idxmax)

            if self._verify_peak(peak_id):
                return peak_id

        return -1

    @cached_property
    def is_peak_reached(self):
        return self.peak_id != -1

    def _verify_peak(self, peak_id, min_gradient_since_peak_threshold=0.05, min_days_since_peak_threshold=7):
        peak_deaths = self.smoothed_deaths.iloc[peak_id]
        last_deaths = self.smoothed_deaths.iloc[-1]

        gradient_since_peak = (peak_deaths - last_deaths) / peak_deaths

        days_since_peak = (self.smoothed_deaths.shape[0] - 1) - peak_id

        return days_since_peak > min_days_since_peak_threshold and gradient_since_peak > min_gradient_since_peak_threshold

    def expanding_windows(self, start=0, window_size=7, **kwargs):
        # base otw is useful for burn-in
        base_otw = OutbreakTimeWindow(self, **kwargs)
        expanding_cutoffs = np.arange(base_otw.start + window_size, len(self), window_size)

        return [OutbreakTimeWindow(self, start=start, end=t) for t in expanding_cutoffs]

    def ffx(self, *xs, x_type="cases"):
        cumulative_x = self.cumulative_cases
        if x_type == "deaths":
            cumulative_x = self.cumulative_deaths

        if len(xs) > 1:
            return [_ffx_index(cumulative_x, x) for x in xs]

        return _ffx_index(cumulative_x, xs[0])

    @staticmethod
    def from_csv(region):
        filepath = f"./data/coronavirus/{region}.csv"

        if os.path.isfile(filepath):
            outbreak_df = pd.read_csv(filepath, index_col=0)
            outbreak_df.index = pd.to_timedelta(outbreak_df.index)

            return Outbreak(region, outbreak_df["Infected"], outbreak_df["Dead"], outbreak_df["Recovered"])

        raise Exception(f"File for {region} not found.")

    def to_df(self):
        return pd.DataFrame({
            "cases": self.cases,
            "deaths": self.deaths,
            "recoveries": self.recoveries,
            **self._secondary_data
        })

    def to_csv(self):
        outbreak_df = pd.DataFrame({"Infected": self.cases, "Dead": self.deaths, "Recovered": self.recoveries})
        outbreak_df.to_csv(f"../data/coronavirus/{self.region}.csv")

        smooth_outbreak_df = pd.DataFrame({
            "Infected": self.smoothed_cases.round().astype(int),
            "Dead": self.smoothed_deaths.round().astype(int),
            "Recovered": self.smoothed_recoveries.round().astype(int)
        })
        smooth_outbreak_df.to_csv(f"../data/coronavirus/{self.region}_s{self.smoothing_window}.csv")
