import os
from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from utils.path import DATA_ROOTPATH
from .time_window import OutbreakTimeWindow


def _ffx_index(cumulative_x, threshold):
    x_max = cumulative_x.iloc[-1]

    if threshold > x_max:
        return None

    return np.argmax(cumulative_x > threshold)


class Outbreak:
    required_fields = ["cases", "deaths", "recoveries"]

    def __init__(self, region: str, cases: pd.Series, deaths: pd.Series, recoveries: pd.Series,
                 smoothing_window: int = 3, df: Optional[pd.DataFrame] = None, **kwargs):
        self._df: pd.DataFrame = pd.DataFrame({"cases": cases, "deaths": deaths, "recoveries": recoveries, **kwargs})
        self._df.name = region

        # join with df if attribute present
        if df is not None:
            self._df = self._df.join(df)

        # construct the cumulative columns if they are not present in the df
        for field in Outbreak.required_fields:
            if f"cumulative_{field}" not in self._df.columns:
                self._df[f"cumulative_{field}"] = self._df[field].cumsum()

        self.smoothing_window = smoothing_window

    def __getitem__(self, item):
        if type(item) is slice:
            return self._df.iloc[item]

        return getattr(self, item)

    def __getattr__(self, item):
        df = self._df

        if item.startswith("smooth"):
            df = self._smooth_df
            item = item[item.find("_") + 1:]

        return df[item]

    def __len__(self):
        return self._df.shape[0]

    @cached_property
    def region(self):
        return self._df.name

    @cached_property
    def resolved_case_rate(self):
        return (self.cumulative_deaths + self.cumulative_recoveries) / self.cumulative_cases

    @cached_property
    def df(self):
        return self._df

    @cached_property
    def peak_id(self):
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
        return self.peak_id != -1

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
        self._smooth_df = self._df.copy().rolling(self.smoothing_window, center=True).mean().dropna()

    def expanding_windows(self, start=0, window_size=7, **kwargs):
        # base otw is useful for burn-in
        base_otw = OutbreakTimeWindow(self, **kwargs)
        expanding_cutoffs = np.arange(base_otw.start + window_size, len(self), window_size)

        return [OutbreakTimeWindow(self, start=start, end=t) for t in expanding_cutoffs]

    def ffx(self, *xs, x_type="cases"):
        cumulative_x = self._df[f"cumulative_{x_type}"]

        if len(xs) > 1:
            return [_ffx_index(cumulative_x, x) for x in xs]

        return _ffx_index(cumulative_x, xs[0])

    @staticmethod
    def from_df(df: pd.DataFrame):
        assert all(field in df.columns for field in Outbreak.required_fields)

        other_columns = df[df.columns[~df.columns.isin(Outbreak.required_fields)]]

        return Outbreak(
            df.name,
            df["cases"],
            df["deaths"],
            df["recoveries"],
            df=other_columns
        )

    @staticmethod
    def from_csv(region):
        filepath = f"./data/coronavirus/{region}.csv"

        if os.path.isfile(filepath):
            outbreak_df = pd.read_csv(filepath, index_col=0)
            outbreak_df.index = pd.to_timedelta(outbreak_df.index)
            outbreak_df.name = region

            return Outbreak.from_df(outbreak_df)

        raise Exception(f"File for {region} not found.")

    def to_csv(self):
        self._df.to_csv(DATA_ROOTPATH / f"coronavirus/{self.region}.csv")
        self._smooth_df.to_csv(DATA_ROOTPATH / f"coronavirus/{self.region}_s{self.smoothing_window}.csv")

