import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class Outbreak:
    def __init__(self, region, epidemic_curve, fatality_curve, recovery_curve=None, advanced_testing_policy_adopted=False, smoothing_window=3, latest_testing_per_thousand=None):
        self.region = region

        self.epidemic_curve = epidemic_curve
        self.cumulative_epidemic_curve = epidemic_curve.cumsum()

        self.fatality_curve = fatality_curve
        self.cumulative_fatality_curve = fatality_curve.cumsum()
        
        self.recovery_curve = recovery_curve
        self.cumulative_recovery_curve = recovery_curve.cumsum() if recovery_curve is not None else None
        
        self.advanced_testing_policy_adopted = advanced_testing_policy_adopted
        self.latest_testing_per_thousand = latest_testing_per_thousand

        self.smoothing_window = smoothing_window
        
    @property
    def smoothing_window(self):
        return self._smoothing_window

    @smoothing_window.setter
    def smoothing_window(self, smoothing_window):
        assert smoothing_window > 0 and smoothing_window < 60
        
        self._smoothing_window = smoothing_window
        
        self.smooth_epidemic_curve = self._apply_smoothing(self.epidemic_curve)
        self.smooth_fatality_curve = self._apply_smoothing(self.fatality_curve)
        
        if self.recovery_curve is not None:
            self.smooth_recovery_curve = self._apply_smoothing(self.recovery_curve)
    
    def _apply_smoothing(self, curve):
        return curve.rolling(self.smoothing_window, center=True).mean().dropna()
    
    @property
    def duration(self):
        return self.epidemic_curve.shape[0]

    @property
    def resolved_case_rate(self):
        return (self.cumulative_fatality_curve + self.cumulative_recovery_curve) / (self.cumulative_epidemic_curve)

    def get_peak_id(self, **kwargs):        
        # extract smooth peak idx (normalize daily anomallies due to changes in counting strategies...)
        smooth_peak_idx, _ = find_peaks(self.smooth_fatality_curve, prominence=1)
    
        if len(smooth_peak_idx) > 0:
            if len(smooth_peak_idx) == 1:
                smooth_peak_id = smooth_peak_idx[0]
            else:
                smooth_peak_idxmax = self.smooth_fatality_curve.iloc[smooth_peak_idx].idxmax()
                smooth_peak_id = self.smooth_fatality_curve.index.get_loc(smooth_peak_idxmax)

            peak_idxmax = self.fatality_curve.iloc[max(smooth_peak_id - self.smoothing_window, 0):min(smooth_peak_id + self.smoothing_window, self.duration - 1)].idxmax()
            peak_id = self.fatality_curve.index.get_loc(peak_idxmax)

            if self._verify_peak(peak_id, **kwargs):
                return peak_id
        
        return -1
    
    def is_peak_reached(self, **kwargs):
        return self.get_peak_id(**kwargs) != -1
        
    def _verify_peak(self, peak_id, min_gradient_since_peak_threshold=0.05, min_days_since_peak_threshold=7):
        peak_deaths = self.smooth_fatality_curve.iloc[peak_id]
        last_deaths = self.smooth_fatality_curve.iloc[-1]
        
        gradient_since_peak = (peak_deaths - last_deaths) / peak_deaths
        
        days_since_peak = (self.smooth_fatality_curve.shape[0] - 1) - peak_id
        
        return days_since_peak > min_days_since_peak_threshold and gradient_since_peak > min_gradient_since_peak_threshold

    @property
    def cfr_curve(self):
        return (self.fatality_curve.cumsum(axis=0) / self.epidemic_curve.cumsum(axis=0))
    
    def ffx_cases(self, *xs):
        return [np.argmax(self.cumulative_epidemic_curve > x) for x in xs]
    
    def ffx_deaths(self, *xs):
        return [np.argmax(self.cumulative_fatality_curve > x) for x in xs]

    @staticmethod
    def load(region):
        filepath = f"../data/coronavirus/{region}.csv"

        if os.path.isfile(filepath):
            outbreak_df = pd.read_csv(filepath, index_col=0)
            outbreak_df.index = pd.to_timedelta(outbreak_df.index)

            return Outbreak(region, outbreak_df["Infected"], outbreak_df["Dead"], outbreak_df["Recovered"])

        raise Exception(f"File for {region} not found.")

    def save(self):
        outbreak_df = pd.DataFrame({"Infected": self.epidemic_curve, "Dead": self.fatality_curve, "Recovered": self.recovery_curve})
        outbreak_df.to_csv(f"../data/coronavirus/{self.region}.csv")

        smooth_outbreak_df = pd.DataFrame({
            "Infected": self.smooth_epidemic_curve.round().astype(int),
            "Dead": self.smooth_fatality_curve.round().astype(int),
            "Recovered": self.smooth_recovery_curve.round().astype(int)
            })
        smooth_outbreak_df.to_csv(f"../data/coronavirus/{self.region}_s{self.smoothing_window}.csv")

    def plot(self, ax, plotname, savefig=False):
        plt.title(f"{self.region} - {plotname}")

        if savefig:
            if self.region not in os.listdir("../plots/coronavirus"):
                os.mkdir(f"../plots/coronavirus/{self.region}")
            plt.savefig(f"../plots/coronavirus/{self.region}/{plotname}.pdf")

        plt.legend()

        plt.show()
