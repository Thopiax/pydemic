import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class Outbreak:
    def __init__(self, region, cases, deaths, smoothing_window=3, **data):
        self.region = region

        self.cases = cases
        self.cumulative_cases = cases.cumsum()

        self.deaths = deaths
        self.cumulative_deaths = deaths.cumsum()
        
        self.recoveries = data.get("recoveries")
        if self.recoveries is not None:
            self.cumulative_recoveries = self.recoveries.cumsum()
        
        self.advanced_testing_policy_adopted = data.get("advanced_testing_policy_adopted", False)
        self.latest_testing_per_thousand = data.get("latest_testing_per_thousand")

        self.smoothing_window = smoothing_window
        
    @property
    def smoothing_window(self):
        return self._smoothing_window

    @smoothing_window.setter
    def smoothing_window(self, smoothing_window):
        assert 0 < smoothing_window < 60

        self._smoothing_window = smoothing_window

        self.smoothed_cases = self._apply_smoothing(self.cases)
        self.smoothed_deaths = self._apply_smoothing(self.deaths)

        self.smoothed_recoveries = None
        if self.recoveries is not None:
            self.smoothed_recoveries = self._apply_smoothing(self.recoveries)
    
    def _apply_smoothing(self, sequence):
        return sequence.rolling(self.smoothing_window, center=True).mean().dropna()
    
    @property
    def duration(self):
        return self.cases.shape[0]

    @property
    def resolved_case_rate(self):
        return (self.cumulative_deaths + self.cumulative_recoveries) / (self.cumulative_cases)

    def range(self, start=0, end=None, frequency=7):
        end = self.duration if end is None else end

        return np.arange(start, end, frequency)

    def get_peak_id(self, **kwargs):        
        # extract smooth peak idx (normalize daily anomalies due to changes in counting strategies...)
        smooth_peak_idx, _ = find_peaks(self.smoothed_deaths, prominence=1)
    
        if len(smooth_peak_idx) > 0:
            if len(smooth_peak_idx) == 1:
                smooth_peak_id = smooth_peak_idx[0]
            else:
                smooth_peak_idxmax = self.smoothed_deaths.iloc[smooth_peak_idx].idxmax()
                smooth_peak_id = self.smoothed_deaths.index.get_loc(smooth_peak_idxmax)

            peak_idxmax = self.deaths.iloc[max(smooth_peak_id - self.smoothing_window, 0):min(smooth_peak_id + self.smoothing_window, self.duration - 1)].idxmax()
            peak_id = self.deaths.index.get_loc(peak_idxmax)

            if self._verify_peak(peak_id, **kwargs):
                return peak_id
        
        return -1
    
    def is_peak_reached(self, **kwargs):
        return self.get_peak_id(**kwargs) != -1
        
    def _verify_peak(self, peak_id, min_gradient_since_peak_threshold=0.05, min_days_since_peak_threshold=7):
        peak_deaths = self.smoothed_deaths.iloc[peak_id]
        last_deaths = self.smoothed_deaths.iloc[-1]
        
        gradient_since_peak = (peak_deaths - last_deaths) / peak_deaths
        
        days_since_peak = (self.smoothed_deaths.shape[0] - 1) - peak_id
        
        return days_since_peak > min_days_since_peak_threshold and gradient_since_peak > min_gradient_since_peak_threshold

    def ffx_cases(self, *xs):
        return [np.argmax(self.cumulative_cases > x) for x in xs]
    
    def ffx_deaths(self, *xs):
        return [np.argmax(self.cumulative_deaths > x) for x in xs]

    @staticmethod
    def from_csv(region):
        filepath = f"../data/coronavirus/{region}.csv"

        if os.path.isfile(filepath):
            outbreak_df = pd.read_csv(filepath, index_col=0)
            outbreak_df.index = pd.to_timedelta(outbreak_df.index)

            return Outbreak(region, outbreak_df["Infected"], outbreak_df["Dead"], outbreak_df["Recovered"])

        raise Exception(f"File for {region} not found.")

    def to_csv(self):
        outbreak_df = pd.DataFrame({"Infected": self.cases, "Dead": self.deaths, "Recovered": self.recoveries})
        outbreak_df.to_csv(f"../data/coronavirus/{self.region}.csv")

        smooth_outbreak_df = pd.DataFrame({
            "Infected": self.smoothed_cases.round().astype(int),
            "Dead": self.smoothed_deaths.round().astype(int),
            "Recovered": self.smoothed_recoveries.round().astype(int)
            })
        smooth_outbreak_df.to_csv(f"../data/coronavirus/{self.region}_s{self.smoothing_window}.csv")

    # def plot(self, ax, plotname, savefig=False):
    #     plt.title(f"{self.region} - {plotname}")
    #
    #     if savefig:
    #         if self.region not in os.listdir("../plots/coronavirus"):
    #             os.mkdir(f"../plots/coronavirus/{self.region}")
    #         plt.savefig(f"../plots/coronavirus/{self.region}/{plotname}.pdf")
    #
    #     plt.legend()
    #
    #     plt.show()
