import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Outbreak:
    def __init__(self, region, epidemic_curve, fatality_curve, recovery_curve, smoothing_coefficient=3):
        self.region = region
        
        self.epidemic_curve = epidemic_curve
        self.cumulative_epidemic_curve = epidemic_curve.cumsum()
        
        self.fatality_curve = fatality_curve
        self.cumulative_fatality_curve = fatality_curve.cumsum()

        self.recovery_curve = recovery_curve
        self.cumulative_recovery_curve = recovery_curve.cumsum()
        
        # number of days considered per time period
        self.smoothing_coefficient = smoothing_coefficient
        self._smooth_epidemic_curves = {}
        self._smooth_fatality_curves = {}

    @property
    def duration(self):
        return self.epidemic_curve.shape[0]

    @property
    def resolved_case_rate(self):
        return  (self.cumulative_fatality_curve + self.cumulative_recovery_curve) / (self.cumulative_epidemic_curve)
        
    @property
    def smooth_epidemic_curve(self):
        if self.smoothing_coefficient not in self._smooth_epidemic_curves.keys():
            self._smooth_epidemic_curves[self.smoothing_coefficient] = self.epidemic_curve.rolling(self.smoothing_coefficient).mean().dropna()
            
        return self._smooth_epidemic_curves[self.smoothing_coefficient]
    
    @property
    def smooth_fatality_curve(self):
        if self.smoothing_coefficient not in self._smooth_fatality_curves.keys():
            self._smooth_fatality_curves[self.smoothing_coefficient] = self.fatality_curve.rolling(self.smoothing_coefficient).mean().dropna()
            
        return self._smooth_fatality_curves[self.smoothing_coefficient]

    @property
    def cfr_curve(self):
        return (self.fatality_curve.cumsum(axis=0) / self.epidemic_curve.cumsum(axis=0))
    
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
    
    def plot(self, ax, plotname, savefig=False):
        plt.title(f"{self.region} - {plotname}")
        
        if savefig:
            if self.region not in os.listdir("../plots/coronavirus"):
                os.mkdir(f"../plots/coronavirus/{self.region}")
            plt.savefig(f"../plots/coronavirus/{self.region}/{plotname}.pdf")
            
        plt.legend()
            
        plt.show()
