import pandas as pd
import matplotlib.pyplot as plt


class CFREvaluator:
    def __init__(self, outbreak, *estimator_classes, analysis_frequency=7):
        self.outbreak = outbreak
        self.estimators = [cls(outbreak) for cls in estimator_classes]

        self.analysis_frequency = analysis_frequency

        self._build_estimates()

    def _build_estimates(self):
        self._estimates = pd.DataFrame()

        for estimator in self.estimators:
            self._estimates[repr(estimator)] = estimator.range_estimate(frequency=self.analysis_frequency)

    def plot_estimates(self, **kwargs):
        self._estimates.plot(ax=plt.gca(), **kwargs)