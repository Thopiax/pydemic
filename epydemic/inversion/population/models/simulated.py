import numpy as np
import pandas as pd

from epydemic.inversion.population.models.base import PopulationFatalityModel


class SimulatedPopulationFatalityModel(PopulationFatalityModel):
    def __init__(self, *args, n_sims=1000, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_sims = n_sims

        self._simulation_cache = {}

    def simulate_case_outcomes(self, n_cases):
        probability_samples = np.random.rand(n_cases, self.patient_hazard.K + 1)

        fatality_mask = probability_samples[:, 0] < self.patient_hazard.alpha
        # argmax returns the index of the first "True" in an array of booleans
        fatality_delays = np.argmax(probability_samples[:, 1:] < self.patient_hazard.hazard_rate, axis=1)

        return fatality_delays, fatality_mask

    def __repr__(self):
        return "simulation"

    def update(self, parameters, **kwargs):
        super().update(parameters, **kwargs)

    def fit(self, cases: pd.Series, deaths: pd.Series):
        pass

    def predict(self, cases: pd.Series, overwrite_cache=False, n_sims=None):
        if n_sims is None:
            n_sims = self.n_sims

        T = cases.shape[0]

        # perform calculations if necessary
        if overwrite_cache or self.patient_hazard.parameters not in self._simulation_cache:
            deaths = np.zeros((n_sims, T))

            # simulate every day of the outbreak
            for t, n_cases in enumerate(cases):
                fatality_delays, fatality_mask = self.simulate_case_outcomes(n_cases * n_sims)

                for sim in range(n_sims):
                    sim_ptr = n_cases * sim

                    # get inversion delays and inversion bitmask for sim
                    sim_fatality_delays = fatality_delays[sim_ptr: sim_ptr + n_cases]
                    sim_fatality_mask = fatality_mask[sim_ptr: sim_ptr + n_cases]

                    # reduce inversion delays to inversion tolls per day
                    sim_fatality_days = np.bincount(sim_fatality_delays[sim_fatality_mask])

                    # censor results to avoid index out of bounds
                    censoring = min(sim_fatality_days.shape[0], T - t)

                    # add inversion tolls per date to simulation counts
                    deaths[sim, t:(t + censoring)] += sim_fatality_days[:censoring]

            # update caches
            self._simulation_cache[self.patient_hazard.parameters] = deaths

        return np.mean(self._simulation_cache[self.patient_hazard.parameters], axis=0)

