from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.random import binomial, multinomial
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from data.synthetic.sei4rd import SEI4RD


class SEI4RDSimulator(ABC):
    STABILITY_THRESHOLD = 1e-8

    def __init__(self, model: SEI4RD):
        self.model = model

        self._previous_simulations = []

    def convert_to_SEIRD(self, simulation: Optional[pd.DataFrame] = None):
        if simulation is None:
            simulation = self._previous_simulations[-1]

        df = simulation.copy()

        df["I"] = simulation[self.model.infected_compartments].sum(axis=1)
        df.drop(columns=self.model.infected_compartments, inplace=True)

        # rerange the columns
        df = df[["S", "E", "I", "R", "D"]]

        return df

    def plot(self, simulation: Optional[pd.DataFrame] = None):
        df = self.convert_to_SEIRD(simulation)

        ax = plt.gca()
        df.plot(ax=ax)
        plt.show()

    def _verify_stability(self, simulation: pd.DataFrame):
        # numerical stability: sum of all compartments should not deviate from N
        assert all(abs(simulation.sum(axis=1) - self.model.N) < SEI4RDSimulator.STABILITY_THRESHOLD)

    @abstractmethod
    def simulate(self, T: int, dt: float = 0.05):
        pass


class StochasticSEI4RDSimulator(SEI4RDSimulator):
    def __init__(self, model: SEI4RD):
        super().__init__(model)

        self._multinomial_rates = []

    def _diff(self, dt: float):
        state_diff = {comp: 0 for comp in self.model.compartments}

        for (src, edges) in self.model.transition_rates.items():
            if len(edges) == 1:
                [(dest, rate)] = edges

                flow = binomial(self.model.get_compartment(src), rate * dt)

                # remove flow from src component
                state_diff[src] -= flow
                # add flow to dest component
                state_diff[dest] += flow
            else:
                assert src == "E"

                # add the last compartment as the "catchall" i.e. the probability that they stay exposed.
                rates = np.array([rate for (_, rate) in edges] + [0])

                # for numerical consistency, we should choose dt for the expression below to approximate 1/2
                self._multinomial_rates.append(sum(rates) * dt)

                flow = multinomial(self.model.get_compartment(src), rates * dt)

                state_diff[src] -= sum(flow[:-1])

                for index, (dest, _) in enumerate(edges):
                    state_diff[dest] += flow[index]

        return state_diff

    def _run_simulation(self, T: int, dt: float):
        time_index = np.arange(0, T, dt)

        simulation = pd.DataFrame(index=time_index, columns=self.model.compartments)

        for t in time_index:
            simulation.loc[t, :] = self.model.state

            self.model.update_state(self._diff(dt))

        self._verify_stability(simulation)

        return simulation

    def simulate(self, T: int, dt: float = 0.05, iters: int = 1) -> pd.DataFrame:
        self.model.init_graph()

        simulation = self._run_simulation(T, dt)

        # if more iterations are needed
        if iters > 1:
            for i in range(1, iters):
                self.model.init_graph()

                print(f"Iteration {i}...")

                extra_simulation = self._run_simulation(T, dt)

                simulation = simulation.add(extra_simulation, axis=1)

            simulation = simulation.div(float(iters))

        self._verify_stability(simulation)
        self._previous_simulations.append(simulation)

        return simulation


class DeterministicSEI4RDSimulator(SEI4RDSimulator):
    def _diff(self, _t, X):
        self.model.set_state_from_vector(X)

        state_diff = {comp: 0 for comp in self.model.compartments}

        for (src, edge) in self.model.transition_rates.items():
            for (dest, rate) in edge:
                flow = self.model.get_compartment(src) * rate

                # remove flow from src component
                state_diff[src] -= flow
                # add flow to dest component
                state_diff[dest] += flow

        return [state_diff[comp] for comp in self.model.compartments]

    def simulate(self, T: int, dt: float = 0.05) -> pd.DataFrame:
        self.model.init_graph()

        # create the initial state vector
        state = [val for (_, val) in self.model.state.items()]

        # solve the initial value problem
        solution = solve_ivp(self._diff, (0, T), state, t_eval=np.arange(0, T, dt))

        # build a history dataframe from the solution
        simulation = pd.DataFrame(data=solution.y.T, index=solution.t, columns=self.model.compartments)

        # verify stability condition
        self._verify_stability(simulation)

        # cache simulation
        self._previous_simulations.append(simulation)

        return simulation
