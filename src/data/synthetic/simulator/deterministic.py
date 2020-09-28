import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from data.synthetic.simulator.base import Simulator


class DeterministicSimulator(Simulator):
    def _diff(self, _t, X):
        self.model.set_state_from_vector(X)

        state_diff = {comp: 0 for comp in self.model.compartments}

        for (src, edge) in self.model.transition_rates.items():
            for (dest, rate) in edge:
                flow = self.model[src] * rate

                # remove flow from src component
                state_diff[src] -= flow
                # add flow to dest component
                state_diff[dest] += flow

        return [state_diff[comp] for comp in self.model.compartments]

    def simulate(self, T: int, dt: float = 0.05) -> pd.DataFrame:
        self.model.set_initial_state()

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