from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from scipy.stats import weibull_min

from outbreak import Outbreak
from .exceptions import OutbreakAlreadyFittedException, InvalidHazardRateException
from ..utils.logging import build_logger


class PopulationModel(ABC):
    def __init__(self, parameters : Optional[Tuple[float, float, float]] = None, verbose : bool = False, random_state : int = 1, **kwds):
        if parameters is not None:
            self.parameters = parameters

        self.random_state = random_state
        self.verbose = verbose

        self.outbreak = None
        self.log = build_logger(self.verbose, prefix="no_outbreak")

    def fit(self, outbreak : Outbreak):
        if self.outbreak is not None:
            raise OutbreakAlreadyFittedException

        self.outbreak = outbreak
        self.log = build_logger(self.verbose, prefix=outbreak.region)

        self.refresh()

    def _build_hazard_rate(self, max_ppf : int = 0.999, max_K : int = 100):
        W = weibull_min(self.beta, scale=self.lam)

        # K is the span of the distribution that covers max_ppf % of cases
        self.K = int(W.ppf(max_ppf))

        # K must be at most the max_K described above
        self.K = min(self.K, max_K)

        # K must be positive for the _legacy to be valid (have at least one day)
        if self.K <= 0:
            raise InvalidHazardRateException

        x = np.arange(self.K)

        # construct discrete inversion density
        self.fatality_density = W.cdf(x + 0.5) - W.cdf(x - 0.5)
        cumulative_fatality_density = np.cumsum(self.fatality_density)

        # set _legacy rate
        self.hazard_rate = np.zeros(self.K)
        self.hazard_rate[0] = self.fatality_density[0]
        self.hazard_rate[1:] = self.fatality_density[1:] / (1 - cumulative_fatality_density[:-1])

    @classmethod
    def from_records(cls, records, epidemic):
        result = []

        for index, row in records.iterrows():
            outbreak = epidemic.outbreaks.get(row["region"])

            if outbreak is None:
                continue

            model = cls(parameters=row[["alpha", "beta", "lambda"]])
            model.fit(outbreak)

            result.append(model)

        return result

    @property
    def parameters(self):
        return self.alpha, self.beta, self.lam

    @parameters.setter
    def parameters(self, parameters : Tuple[float, float, float]):
        # noop if new parameters match current ones.
        if self.parameters == parameters: return

        self.alpha, self.beta, self.lam = parameters
        self.refresh()

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def refresh(self):
        self._build_hazard_rate()

    @abstractmethod
    def predict(self, **kwargs):
        pass


class SimulationBasedPopulationModel(PopulationModel):

    def __init__(self, *args, n_sims=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sims = n_sims
        self._simulation_cache = {}

    def _simulate_case_outcomes(self, n_cases):
        probability_samples = np.random.rand(n_cases, self.K + 1)

        fatality_mask = probability_samples[:, 0] < self.alpha
        # argmax returns the index of the first "True" in an array of booleans
        fatality_delays = np.argmax(probability_samples[:, 1:] < self.hazard_rate, axis=1)

        return fatality_delays, fatality_mask

    def __repr__(self):
        return "simulated"

    def refresh(self):
        super().refresh()

    def predict(self, overwrite=False):
        # perform calculations if necessary
        if overwrite or self.parameters not in self._simulation_cache:
            fatalities = np.zeros((self.n_sims, self.outbreak.duration))

            # simulate every day of the outbreak
            for t, n_cases in enumerate(self.outbreak.cases.values.astype(int)):
                fatality_delays, fatality_mask = self._simulate_case_outcomes(n_cases * self.n_sims)

                for sim in range(self.n_sims):
                    sim_ptr = n_cases * sim

                    # get inversion delays and inversion bitmask for sim
                    sim_fatality_delays = fatality_delays[sim_ptr: sim_ptr + n_cases]
                    sim_fatality_mask = fatality_mask[sim_ptr: sim_ptr + n_cases]

                    # reduce inversion delays to inversion tolls per day
                    sim_fatality_days = np.bincount(sim_fatality_delays[sim_fatality_mask])

                    # censor results to avoid index out of bounds
                    censoring = min(sim_fatality_days.shape[0], self.T - t)

                    # add inversion tolls per date to simulation counts
                    fatalities[sim, t:(t + censoring)] += sim_fatality_days[:censoring]

            # update caches
            self._simulation_cache[self.parameters] = fatalities

        return np.mean(self._simulation_cache[self.parameters], axis=0)


class BasePopulationModel(PopulationModel):
    def _build_support_vectors(self):
        self.support_vectors = np.zeros((self.outbreak.duration, self.K))

        cumulative_survival_probability = np.cumprod(1 - self.hazard_rate)

        for t in range(1, self.outbreak.duration):
            for i in range(min(t, self.K)):
                self.support_vectors[t, i] = self.hazard_rate[i] * self.outbreak.cases[t - i]

                if i >= 1:
                    self.support_vectors[t, i] *= cumulative_survival_probability[i - 1]

    def __repr__(self):
        return "base"

    def predict(self):
        return self.alpha * np.sum(self.support_vectors, axis=1)

    def refresh(self):
        super().refresh()
        self._build_support_vectors()


class DensityBasedPopulationModel(PopulationModel):
    def _build_density_vectors(self):
        self.support_vectors = np.zeros((self.outbreak.duration, self.K))

        for t in range(1, self.outbreak.duration):
            for i in range(min(t, self.K)):
                self.support_vectors[t, i] = self.fatality_density[i] * self.outbreak.cases[t - i]

    def __repr__(self):
        return "density"

    def predict(self):
        return self.alpha * np.sum(self.support_vectors, axis=1)

    def refresh(self):
        super().refresh()
        self._build_density_vectors()
