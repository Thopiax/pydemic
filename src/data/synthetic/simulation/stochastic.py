from typing import Collection, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random.mtrand import binomial, multinomial

from data.synthetic.simulation.base import Simulation
from data.synthetic.utils import average_dfs


class StochasticSimulation(Simulation):
    def __init__(self, graph_model):
        super().__init__(graph_model)

        self._multinomial_rates = []


    @staticmethod
    def plot_all(simulations: Collection[pd.DataFrame]):
        ax = plt.gca()

        simplified_simulations = [StochasticSimulation.aggregate_infection_compartments(sim) for sim in simulations]

        for sim in simplified_simulations:
            sim.plot(ax=ax, color='grey', alpha=0.2, legend=False)

        avg_sim = average_dfs(simplified_simulations)
        avg_sim.plot(ax=ax, linewidth=5.0)

        plt.show()

    def _diff(self, dt: float):
        state_diff = {comp: 0 for comp in self.model.compartments}

        for (src, edges) in self.model.transition_rates.items():
            if len(edges) == 1:
                [(dest, rate)] = edges

                flow = binomial(self.model[src], rate * dt)

                # remove flow from src component
                state_diff[src] -= flow
                # add flow to dest component
                state_diff[dest] += flow
            else:
                assert src == "E"

                # add the last compartment as the "catchall" i.e. the probability that they stay exposed.
                infectious_rates = np.array([rate for (_, rate) in edges] + [0])

                # for numerical consistency, we should choose dt for the expression below to approximate 1/2
                self._multinomial_rates.append(sum(infectious_rates) * dt)

                flow = multinomial(self.model[src], infectious_rates * dt)

                state_diff[src] -= sum(flow[:-1])

                for index, (dest, _) in enumerate(edges):
                    state_diff[dest] += flow[index]

        return state_diff

    def _run_simulation(self, T: int, dt: float):
        time_index = np.arange(0, T, dt)

        simulation = pd.DataFrame(index=time_index, columns=self.model.compartments)

        for t in time_index:
            simulation.loc[t, :] = self.model.state

            state_diff = self._diff(dt)

            if self._observer is not None:
                self._observer.observe(t, state_diff)

            self.model.update_state(state_diff)

        self._observer.flush(T)

        return simulation

    def run(self, T: int, dt: float = 0.05, n_sims: int = 10, average_sims: bool = False, aggregate_I: bool = False) -> Union[
        pd.DataFrame, Collection[pd.DataFrame]]:

        results = []

        for i in range(n_sims):
            self.model.set_initial_state()

            print(f"Running simulation #{i}...")
            simulation = self._run_simulation(T, dt)

            self._verify_stability(simulation)
            self._previous_simulations.append(simulation)

            if aggregate_I:
                simulation = StochasticSimulation.aggregate_infection_compartments(simulation)

            results.append(simulation)

        if n_sims == 1:
            results = results[0]

        if average_sims:
            results = average_dfs(results)

        return results
