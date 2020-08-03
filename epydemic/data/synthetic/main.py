import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numpy.random.mtrand import binomial
from scipy.integrate import solve_ivp

from outbreak import Outbreak
from outcome.distribution.discrete import NegBinomialOutcomeDistribution
from outcome.models.fatality import FatalityOutcomeModel
from utils.plot import plot_outbreak


class SEIRDModel:
    def __init__(self, N: int, alpha: float, beta: float, gamma: float, delta: float, rho: float):
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rho = rho

        self.R_0 = self.beta / self.gamma

    def _ode(self, t, y):
        S, E, I, R, D = y

        dS = -self.beta * S * I / self.N
        dE = self.beta * S * I / self.N - self.delta * E
        dI = self.delta * E - (1 - self.alpha) * self.gamma * I - self.alpha * self.rho * I
        dR = (1 - self.alpha) * self.gamma * I
        dD = self.alpha * self.rho * I

        return dS, dE, dI, dR, dD

    def solve(self, T: int, E_0: int = 0, I_0: int = 1, R_0: int = 0, D_0: int = 0):
        S_0 = self.N - I_0 - E_0 - R_0 - D_0

        return solve_ivp(self._ode, (0, T), [S_0, E_0, I_0, R_0, D_0], t_eval=range(0, T))

    def plot(self, T: int):
        solution = self.solve(T)

        S, E, I, R, D = solution.y

        plt.plot(solution.t, S + E + I + R + D, label="N")
        plt.plot(solution.t, S, label="S")
        plt.plot(solution.t, E, label="E")
        plt.plot(solution.t, I, label="I")
        plt.plot(solution.t, R, label="R")
        plt.plot(solution.t, D, label="D")

        plt.legend()

        plt.show()

    def generate_outbreak(self, T: int):
        solution = self.solve(T)

        S, E, I, R, D = solution.y

        return Outbreak(f"SEIRD_{self.N}", cases=I, deaths=D, recoveries=R)
