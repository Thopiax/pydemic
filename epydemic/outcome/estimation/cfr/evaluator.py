import matplotlib.pyplot as plt
import pandas as pd


class CFREvaluator:
    def __init__(self, outbreak, *estimators, **kwargs):
        self.outbreak = outbreak
        self.estimators = [cls(outbreak) for cls in estimators]

        self._build_estimates(**kwargs)

    def _build_estimates(self, **kwargs):
        self._estimates = pd.DataFrame()

        for estimator in self.estimators:
            self._estimates[repr(estimator)] = estimator.window_estimates(**kwargs)

    @property
    def estimates(self):
        return self._estimates