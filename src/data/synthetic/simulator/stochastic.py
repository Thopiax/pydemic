from typing import Collection, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random.mtrand import binomial, multinomial

from data.synthetic.simulator.base import Simulator
from data.synthetic.simulator.utils import average_simulations
from utils.plot import save_figure


class StochasticSimulator(Simulator):
    def __init__(self, model):
        super().__init__(model)

        self._multinomial_rates = []

    # @save_figure(lambda simulations: f"synthetic/stochastic_{len(simulations)}sims.pdf")
    def plot_all(self, simulations: Collection[pd.DataFrame]):
        ax = plt.gca()

        for sim in simulations:
            agg_sim = self.aggregate_infection_compartments(sim)
            agg_sim.plot(ax=ax, color='grey', alpha=0.2, legend=False)

        avg_sim = average_simulations(simulations)
        avg_sim = self.aggregate_infection_compartments(avg_sim)
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
                rates = np.array([rate for (_, rate) in edges] + [0])

                # for numerical consistency, we should choose dt for the expression below to approximate 1/2
                self._multinomial_rates.append(sum(rates) * dt)

                flow = multinomial(self.model[src], rates * dt)

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

    def simulate(self, T: int, dt: float = 0.05, n_sims: int = 10) -> Union[pd.DataFrame, Collection[pd.DataFrame]]:
        results = []

        for i in range(n_sims):
            self.model.set_initial_state()

            print(f"Running simulation #{i}...")
            simulation = self._run_simulation(T, dt)

            results.append(simulation)

            self._verify_stability(simulation)
            self._previous_simulations.append(simulation)

        if n_sims == 1:
            results = results[0]

        return results