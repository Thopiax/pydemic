import numpy as np
from abc import ABC, abstractmethod

from scipy.stats import weibull_min

from .utils import build_logger
from .exceptions import *


class HazardModel(ABC):
    """
    The OubreakEstimator abstract class provides a common foundation to build the different estimators devised for the study.

    The main parameters for the estimators are:

        - alpha: CFR
        - beta: Weibull shape
        - lambda: Weibull scale (called `lam` due to the Python primitive)

    There are also optional parameters:

        - verbose
        - random_state
        - warm_start, warm_start_min_deaths_observed: whether the model should only predict the daily fatality curve after X cumulative deaths have been seen.
        - error, cumulative_error: error method between realization and estimate to be used on the daily curve.

    """
    def __init__(self, outbreak, alpha=0.5, beta=1.0, lam=10., verbose=False, random_state=1, **kwds):
        self.outbreak = outbreak
        self.alpha, self.beta, self.lam = alpha, beta, lam

        self._build_hazard_rate()
        self._build_cases_and_deaths()

        self.verbose = verbose
        self.log = build_logger(verbose, prefix=outbreak.region)

        self.random_state = random_state

        self._update_model()

    def _build_cases_and_deaths(self):
        self.T = self.outbreak.duration
        self.cases = self.outbreak.epidemic_curve.values
        self.deaths = self.outbreak.fatality_curve.values

    def _build_hazard_rate(self, ppf_end=0.999, max_K=90):
        W = weibull_min(self.beta, scale=self.lam)
        self.K = min(int(W.ppf(ppf_end)), max_K)

        if self.K == 0: raise InvalidHazardRateException

        x = np.arange(self.K)

        # set fatality rate
        self.fatality_rate = W.cdf(x + 0.5) - W.cdf(x - 0.5)
        cumulative_fatality_rate = np.cumsum(self.fatality_rate)

        # set hazard rate
        self.hazard_rate = np.zeros(self.K)
        self.hazard_rate[0] = self.fatality_rate[0]
        self.hazard_rate[1:] = self.fatality_rate[1:] / (1 - cumulative_fatality_rate[:-1])

    @classmethod
    def fromDataFrame(cls, pandemic, df):
        result = []
        for index, row in df.iterrows():
            row_outbreak = pandemic.outbreaks.get(row["region"])

            if row_outbreak is None: continue

            result.append(cls(row_outbreak, alpha=row["alpha"], beta=row["beta"], lam=row["lambda"]))

        return result

    @property
    def parameters(self):
        return self.alpha, self.beta, self.lam

    @parameters.setter
    def parameters(self, parameters):
        assert len(parameters) == 3 and type(parameters) == tuple and all(type(param) == float for param in parameters)

        if self.parameters == parameters: return

        self.alpha, self.beta, self.lam = parameters
        self._update_model()

    # Abstract Methods
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def _update_model(self):
        self._build_hazard_rate()

    @abstractmethod
    def forecast(self, **kwargs):
        pass


class SimulationHazardModel(HazardModel):
    def __init__(self, *args, n_sims=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sims = n_sims
        self._simulation_cache = {}

    def _simulate_outcomes(self, n_cases):
        probability_samples = np.random.rand(n_cases, self.K + 1)

        fatality_mask = probability_samples[:, 0] < self.alpha
        # argmax returns the index of the first "True" in an array of booleans
        fatality_delays = np.argmax(probability_samples[:, 1:] < self.hazard_rate, axis=1)

        return fatality_delays, fatality_mask

    def __str__(self):
        return "simulated"

    def _update_model(self):
        super()._update_model()

    def forecast(self, overwrite=False):
        # perform calculations if necessary
        if overwrite or self.parameters not in self._simulation_cache:
            fatalities = np.zeros((self.n_sims, self.T))

            # simulate every day of the outbreak
            for t, n_cases in enumerate(self.cases.astype(int)):
                fatality_delays, fatality_mask = self._simulate_outcomes(n_cases * self.n_sims)

                for sim in range(self.n_sims):
                    sim_ptr = n_cases * sim

                    # get fatality delays and fatality bitmask for sim
                    sim_fatality_delays = fatality_delays[sim_ptr: sim_ptr + n_cases]
                    sim_fatality_mask = fatality_mask[sim_ptr: sim_ptr + n_cases]

                    # reduce fatality delays to fatality tolls per day
                    sim_fatality_days = np.bincount(sim_fatality_delays[sim_fatality_mask])

                    # censor results to avoid index out of bounds
                    censoring = min(sim_fatality_days.shape[0], self.T - t)

                    # add fatality tolls per date to simulation counts
                    fatalities[sim, t:(t + censoring)] += sim_fatality_days[:censoring]

            # update caches
            self._simulation_cache[self.parameters] = fatalities

        return np.mean(self._simulation_cache[self.parameters], axis=0)


class AnalyticalHazardModel(HazardModel):
    def __str__(self):
        return "base"

    def forecast(self):
        return self.alpha * np.sum(self.support_vectors, axis=1)

    def _update_model(self):
        super()._update_model()
        self._build_support_vectors()

    def _build_support_vectors(self):
        self.support_vectors = np.zeros((self.T, self.K))

        cumulative_survival_probability = np.cumprod(1 - self.hazard_rate)

        for t in range(1, self.T):
            for i in range(min(t, self.K)):
                self.support_vectors[t, i] = self.hazard_rate[i] * self.cases[t - i]

                if i >= 1:
                    self.support_vectors[t, i] *= cumulative_survival_probability[i - 1]
