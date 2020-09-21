import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


class SimpleSEIRD:
    def __init__(self, N: int, alpha: float, beta: float, gamma: float, delta: float, rho: float):
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rho = rho

        self.R_0 = self.beta / self.gamma

    def _ode(self, t, X):
        S, E, I, R, D = X

        dS = -self.beta * S * I / self.N
        dE = self.beta * S * I / self.N - self.delta * E
        dI = self.delta * E - (1 - self.alpha) * self.gamma * I - self.alpha * self.rho * I
        dR = (1 - self.alpha) * self.gamma * I
        dD = self.alpha * self.rho * I

        return dS, dE, dI, dR, dD

    def simulate(self, T: int, dt: float = 0.01, E_0: float = 10.0, I_0: float = 0.0, R_0: float = 0.0, D_0: float = 0.0):
        S_0 = self.N - I_0 - E_0 - R_0 - D_0

        solution = solve_ivp(self._ode, (0, T), [S_0, E_0, I_0, R_0, D_0], t_eval=np.arange(0, T, dt))

        # build a history dataframe from the solution
        history = pd.DataFrame(data=solution.y.T, index=solution.t, columns=["S", "E", "I", "R", "D"])

        return history

    def plot(self, history: pd.DataFrame):
        ax = plt.gca()
        history.plot(ax=ax)
        plt.show()


# if __name__ == "__main__":
    # from sklearn.metrics import mean_absolute_error
    #
    # from src.data.synthetic.seird import SEI4RD, DeterministicSimulator
    # from data.synthetic.parameters import SEParameters, OutcomeParameters
    #
    # alpha = 0.2
    # beta = 1.5
    #
    # delta = 0.2
    #
    # gamma = 0.7
    # rho = 0.3
    #
    # T = 100
    #
    # complex_model = SEI4RD(1_000_000, alpha,
    #                        infection_parameters=SEParameters(beta_E=0, c_I=1.0, beta_I=beta, D_E=1 / delta),
    #                        recovery_parameters=OutcomeParameters(lambdaS=gamma, K=1),
    #                        death_parameters=OutcomeParameters(lambdaS=rho, K=1))
    #
    # complex_simulator = DeterministicSimulator(complex_model)
    # complex_history = complex_simulator.simulate(T)
    #
    # complex_simulator.plot(complex_history)
    #
    # # apples to apples
    # complex_history = complex_simulator.aggregate_infection_compartments(complex_history)
    #
    # basic_model = SimpleSEIRD(1_000_000, alpha, beta, gamma, delta, rho)
    # basic_history = basic_model.simulate(T)
    #
    # basic_model.plot(basic_history)
    #
    # print(f"MAE={mean_absolute_error(basic_history.values, complex_history.values)}")
    #
