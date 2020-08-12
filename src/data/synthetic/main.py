import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


class SEI4RD:
    def __init__(self, N: int, alpha: float, d_E: int = 5,
                 c_I: float = 1.0, beta_I: float = 0.5, c_E: float = 1.0, beta_E: float = 0.5,
                 gamma_D: float = 1.0, K_D: int = 1, lambda_SD: float = 1 / 5.0, lambda_AD: float = 1.0,
                 gamma_R: float = 1.0, K_R: int = 1, lambda_SR: float = 1 / 5.0, lambda_AR: float = 1.0):
        self.N = N

        assert 0 < alpha < 1
        self.alpha = alpha

        # incubation period
        self.d_E = d_E

        # infection parameters - Infected
        self.c_I = c_I
        self.beta_I = beta_I

        # infection parameters - Exposed
        self.c_E = c_E
        self.beta_E = beta_E

        # death outcome parameters
        assert 0 <= gamma_D <= 1
        self.gamma_D = gamma_D  # symptomatic probability

        assert K_D > 0
        self.K_D = K_D  # number of death compartments

        self.lambda_SD = lambda_SD
        self.lambda_AD = lambda_AD

        # recovery outcome parameters
        assert 0 <= gamma_R <= 1
        self.gamma_R = gamma_R  # symptomatic probability

        assert K_D > 0
        self.K_R = K_R  # number of recovery compartments

        self.lambda_SR = lambda_SR
        self.lambda_AR = lambda_AR

    def grad(self, _t, X):
        assert len(X) == 6 + self.K_D + self.K_R

        # retrieve simple individual compartments
        S, E, I_AD, I_AR, D, R = X[:6]

        # retrieve composed infected compartments
        I_SD = np.array(X[6:6 + self.K_D])
        I_SR = np.array(X[6 + self.K_D:6 + self.K_D + self.K_R])

        # calculate total infected
        I = I_AD + sum(I_SD) + I_AR + sum(I_SR)

        # calculate dS
        S_in = 0
        S_out = (I * S * self.c_I * self.beta_I) / self.N + (E * S * self.c_E * self.beta_E) / self.N
        dS = S_in - S_out

        # calculate dE
        E_in = S_out
        E_out = E / self.d_E
        dE = E_in - E_out

        # calculate dI_AD
        I_AD_in = E_out * self.alpha * (1 - self.gamma_D)
        I_AD_out = I_AD * self.lambda_AD
        dI_AD = I_AD_in - I_AD_out

        # calculate dI_SD
        I_SD_out = I_SD * self.lambda_SD
        dI_SD = np.zeros(self.K_D)
        dI_SD[0] = E_out * self.alpha * self.gamma_D - I_SD_out[0]

        for k in range(1, self.K_D):
            dI_SD[k] = I_SD_out[k - 1] - I_SD_out[k]

        # calculate dD
        D_in = I_SD_out[-1] + I_AD_out
        D_out = 0
        dD = D_in - D_out

        # calculate dI_AR
        I_AR_in = E_out * (1 - self.alpha) * (1 - self.gamma_R)
        I_AR_out = I_AR * self.lambda_AR
        dI_AR = I_AR_in - I_AR_out

        # calculate dI_SR
        I_SR_out = I_SR * self.lambda_SR
        dI_SR = np.zeros(self.K_R)
        dI_SR[0] = E_out * (1 - self.alpha) * self.gamma_R - I_SR_out[0]

        for k in range(1, self.K_R):
            dI_SR[k] = I_SR_out[k - 1] - I_SR_out[k]

        # calculate dR
        R_in = I_SR_out[-1] + I_AR_out
        R_out = 0
        dR = R_in - R_out

        return dS, dE, dI_AD, dI_AR, dD, dR, *dI_SD, *dI_SR

    def _initial_state(self, E_0: int):
        return [self.N - E_0, E_0, 0, 0, 0, 0] + ([0] * self.K_D) + ([0] * self.K_R)

    def solve(self, T: int, dt: float = 0.01, E_0: int = 10):
        return solve_ivp(self.grad, (0, T), self._initial_state(E_0), t_eval=np.arange(0, T, dt))

    def plot(self, T: int):
        solution = self.solve(T)

        X = solution.y

        # retrieve simple individual compartments
        S, E, I_AD, I_AR, D, R = X[:6]

        # retrieve composed infected compartments
        I_SD = np.array(X[6:6 + self.K_D])
        I_SR = np.array(X[6 + self.K_D:6 + self.K_D + self.K_R])

        # calculate total infected
        I = I_AD + sum(I_SD) + I_AR + sum(I_SR)

        plt.plot(solution.t, S + E + I + R + D, label="N")
        plt.plot(solution.t, S, label="S")
        plt.plot(solution.t, E, label="E")
        plt.plot(solution.t, I, label="I")
        plt.plot(solution.t, R, label="R")
        plt.plot(solution.t, D, label="D")

        plt.legend()

        plt.show()


if __name__ == '__main__':
    model = SEI4RD(1_000_000, 0.2, K_D=5)

    model.plot(100)
