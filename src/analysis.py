# %% markdown
# # Mortality Rate Estimation
#
# The data found for previous disease outbreaks will come into use to tune the model for regression. We will mostly focus on other Coronaviruses outbreaks (i.e., SARS or MERS) to construct the estimates.
# %% codecell
import pandas as pd
from pandas.plotting import autocorrelation_plot

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

%matplotlib inline

plt.rcParams['figure.figsize']=[40,20]
plt.rcParams['font.size']=22
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 24
# %% markdown
# ## SARS
#
# The dataset on the 2003 SARS outbreak was found on Kaggle and based on WHO reports during the crisis.
# %% codecell
# sars_ts = pd.read_excel('data/time_series/SARS.xlsx', usecols=list(range(4)), parse_dates=[1]).set_index('Date', drop=True).rename({"Mortality": "Dead"}, axis=1)
sars_ts = pd.read_csv('data/time_series/SARS.csv', header=0)
# %% markdown
# ## Survivability Rate
#
# First, we must calculate the survival rate $S_{SARS}$ from our data. A value of $90.36\%$ is in line with the rate in [this paper](doi:10.1016/j.socscimed.2006.08.004) (as per Wikipedia, as I have no access to the paper).
# %% codecell
def calculate_survival_rate(ts, dead_col='Dead', infected_col='Infected'):
    return 1 - ts.iloc[-1][dead_col] / ts.iloc[-1][infected_col]
# %% codecell
S_sars = calculate_survival_rate(sars_ts)
S_sars
# %% markdown
# ## Estimate Mortality Curve
#
# Given that the mortality data available is only aggregate level, we must consider the sum of deltas to have a smoother objective function to estimate.
# %% codecell
smooth_sars_ts = sars_ts.diff().resample('W').sum()
# %% markdown
# Once we have our smoothed time series, we need to develop a model to estimate the mortality curve, and by consequence, estimate the hazard function of epidemic outbreaks.
#
# Let $t \in \{1, \dots, T\}$ be the discrete amount of time ellapsed since the beginning of an infection outbreak, where $T$ is the collection period of outbreak data. Furthermore, for any patient, let $k \in \{0, \dots, K\}$ represent the amount of time ellapsed since they have been infected (independent of $t$). In this case, $K$ is the time to outcome where the outcome can represent recovery or death.
#
# Given these two units of time, we can develop a simplistic framework to model the evolution of mortality during an infection. We define $X_t$ and $d_t$ as the number of new infections and deaths, respectively, at a time period $t$.
#
# We further the probability of a person passing away exactly at time period w (after initial infection) as $h_k$ and the number of people who were infected at time $t$ and are alive after $k$ time periods of infection (i.e., person is alive $t + k$ periods after initial outbreak) is $a_{t, k}$.
#
# First, note that we can simply define the number of mortality at time $t$ as
#
# $$ d_t = \sum^{t}_{i=1} h_{t-i} a_{i, t-i} $$
#
# Then, $a_{t, k}$ can also be defined recursively as:
#
# \begin{align*}
#     \begin{aligned}
#         a_{t, 0} &= X_t \\
#         a_{t, k+1} &= (1-h_k)a_{t, k} \quad \forall t \leq T, k \leq K
#     \end{aligned}
#     \implies
#     a_{t, k} = \prod_{j=1}^k(1-h_j)X_t \quad \forall t \leq T, k \leq K
# \end{align*}
#
# Now, let us assume that our hazard rate (the probability of dying after $k$ periods) is a constant $\alpha$ such that $h_0, ..., h_k = \alpha$. This implies that,
#
# $$ a_{t, k} = (1-\alpha)^kX_t $$
#
# , and that,
#
# $$ d_t = \sum^{\min(t, K)}_{i=0} \alpha (1 - \alpha)^k X_{t-k} $$
#
# Therefore, we can derive an estimator $\hat{d_t}$ for $d_t$ by solving the following optimization problem:
#
# \begin{align*}
#     \underset{\alpha, K}{\min} \quad& \left|\left| d - \hat{d} \right|\right| \\
#     s.t. \quad & \hat{d_t} = \sum^{\min(t, K)}_{i=0} \alpha (1 - \alpha)^k X_{t-k} \\
#          & 0 \leq \alpha \leq 1
# \end{align*}
#
# %% codecell
def calculate_d_hat(X, alpha, T, K):
    d_hat = np.zeros(T)

    for t in range(T):
        for k in range(min(t+1, K)):
            d_hat[t] += alpha * ((1 - alpha) ** k) * X[t - k]

    return d_hat

def estimate_death_curve(X, d, S=None, K=None, verbose=True):
    T = X.size

    minimum_alpha = None
    minimum_K = None
    minimum_distance = None
    minimum_d_hat = None

    change_K_flag = (K is None and S is not None)

    alpha_space = np.linspace(0.0001, 0.1, 1000)

    for alpha in alpha_space:
        if change_K_flag:
            K = int(np.ceil(np.log(S) / np.log(1 - alpha)))

        d_hat = calculate_d_hat(X, alpha, T, K)

        distance = np.linalg.norm(d - d_hat)

        if minimum_distance is None or distance < minimum_distance:
            minimum_distance = distance

            minimum_d_hat = d_hat
            minimum_K = K
            minimum_alpha = alpha

    if verbose:
        print(f"Minimum distance={minimum_distance}\nOptimal K={minimum_K}\nOptimal alpha={minimum_alpha}")

    return pd.DataFrame.from_dict({"d": d, "d_hat": minimum_d_hat})
# %% codecell
death_curve = estimate_death_curve(
    smooth_sars_ts['Infected'].values,
    smooth_sars_ts['Dead'].values,
    S_sars
)
# %% codecell
death_curve.plot()
# %% markdown
# ## Censored Data
# %% codecell
def mortality_interval(mortality_rate, interval_size=0.2):
    lower_valid_range = mortality_rate - (mortality_rate * interval_size)
    upper_valid_range = mortality_rate + (mortality_rate * interval_size)

    return lower_valid_range, upper_valid_range
# %% codecell
censor = lambda x, n: x.iloc[:n]
# %% codecell
def plot_censored_mortality_rate_estimation_evolution(ts, censor_start, censor_end, survival_rate=None, K=None):
    results = []

    for i in range(censor_start, censor_end):
        censored_ts = censor(ts, i)

        censored_ts['Estimate Dead'] = estimate_death_curve(
            censored_ts['Infected'].values,
            censored_ts['Dead'].values,
            S=survival_rate,
            K=K,
            verbose=False
        ).set_index(censored_ts.index)['d_hat']

        estimated_mortality_rate = 1 - calculate_survival_rate(censored_ts.cumsum(), dead_col='Estimate Dead')

        results.append(estimated_mortality_rate)


    fig, ax = plt.subplots()
    plt.plot(range(censor_start, censor_end), results)

    if survival_rate is not None:
        lower_mortality, upper_mortality = mortality_interval(1 - survival_rate)
        plt.hlines([lower_mortality, upper_mortality], censor_start, censor_end, colors='red')

        lower_mortality, upper_mortality = mortality_interval(1 - survival_rate, 0.1)
        plt.hlines([lower_mortality, upper_mortality], censor_start, censor_end, colors='orange')

        lower_mortality, upper_mortality = mortality_interval(1 - survival_rate, 0.05)
        plt.hlines([lower_mortality, upper_mortality], censor_start, censor_end, colors='green')

    plt.show()
# %% codecell
plot_censored_mortality_rate_estimation_evolution(smooth_sars_ts, 1, smooth_sars_ts.shape[0], S_sars)
# %% markdown
# ## MERS
# %% codecell
mers_ts = pd.read_csv('data/time_series/MERS_me.csv', parse_dates=[0], index_col=0)
# %% codecell
smooth_mers_ts = mers_ts.diff().resample('W').sum()
# %% codecell
S_mers = calculate_survival_rate(mers_ts)
S_mers
# %% codecell
estimate_death_curve(
    smooth_mers_ts['Infected'].values,
    smooth_mers_ts['Dead'].values,
    S_mers
).plot()
# %% markdown
# Clearly, there are two sepeare outbreaks in this dataset. One ending before week 60 (seems like week 57) and the other starting at week 90.
# %% codecell
first_mers_ts = smooth_mers_ts.loc[:"2018-01-28"]
second_mers_ts = smooth_mers_ts.loc["2018-09-09":]
# %% codecell
first_death_curve = estimate_death_curve(
    first_mers_ts['Infected'].values,
    first_mers_ts['Dead'].values,
    S_mers
)

first_death_curve.plot()
# %% codecell
plot_censored_mortality_rate_estimation_evolution(first_mers_ts, 1, 70, S_mers)
# %% codecell
second_death_curve = estimate_death_curve(
    second_mers_ts['Infected'].values,
    second_mers_ts['Dead'].values,
    S_mers
)

second_death_curve.plot()
# %% markdown
# ## Ebola
# %% codecell
ebola_guinea_ts = pd.read_csv('data/time_series/ebola_guinea.csv', parse_dates=[0], index_col=0)
# %% codecell
S_ebola = calculate_survival_rate(ebola_guinea_ts)
S_ebola
# %% codecell
smooth_ebola_ts = ebola_guinea_ts.diff().resample('W').sum()
# %% codecell
estimate_death_curve(
    smooth_ebola_ts['Infected'].values,
    smooth_ebola_ts['Dead'].values,
    S_ebola
).plot()
# %% markdown
# ## Coronavirus
# %% codecell
coronavirus_df = pd.read_csv('data/time_series/coronavirus.csv', parse_dates=[0], index_col=0).diff().fillna(0)
# %% codecell
smooth_coronavirus_df = coronavirus_df.resample('3D').sum()
# %% codecell
for k in range(smooth_coronavirus_df.shape[0]):
    estimate_death_curve(
        smooth_coronavirus_df['Infected'].values,
        smooth_coronavirus_df['Dead'].values,
        K=k
    )

# Best performance for K = 3
# %% codecell
coronavirus_death_curve = estimate_death_curve(
    coronavirus_df['Infected'].values,
    coronavirus_df['Dead'].values,
    K=3
)

coronavirus_death_curve.plot()
# %% codecell
plot_censored_mortality_rate_estimation_evolution(coronavirus_df, 1, 20, K=2)
# %% markdown
# # Simulate Confidence Interval
#
# ## Probability Distribution of Mortality
# %% codecell
corona_K = 3
corona_alpha = 0.008809720720720721
# %% codecell
def simulate_infected_patients(n_patients, alpha, k0, alpha_k0, K, size=1):
    assert 0 <= k0 and k0 <= K

    # recovery probability after K period
    probabilities = [(1 - alpha_k0) * (1 - alpha) ** (K-1)]

    for k in range(k0):
        probabilities.append(alpha * (1 - alpha) ** k)

    probabilities.append(alpha_k0 * (1 - alpha) ** k0)

    for k in range(k0 + 1, K):
        probabilities.append(alpha * (1 - alpha_k0) * (1 - alpha) ** (k-1))

    print(probabilities)

    return np.random.multinomial(
        n_patients,
        probabilities,
        size=size
    )

def mean_squared_error(estimates, mean):
    return ((estimates - mean) ** 2).mean(1)
# %% codecell
def estimate_hazard_rate(X, d, optimal_alpha=corona_alpha, optimal_K=corona_K, n_runs=100):
    T = X.shape[0]
    alpha_space = np.linspace(0.0001, 0.1, 1000)

    result = []

    for k0 in range(optimal_K):
        for alpha_k0 in alpha_space:
            alpha_death_curves = np.zeros((n_runs, T))

            for t, n_infected in enumerate(X):
                random_paths = simulate_infected_patients(n_infected, optimal_alpha, k0, alpha_k0, optimal_K)

                for j in range(n_runs):
                    alpha_death_curves[j, t:t+K] += random_paths[j][1:T - t + 1]


    for i, alpha in enumerate(alpha_space):
        print(f"({i}/{1000}) Simulating alpha={alpha}...")


        for t, n_infected in enumerate(X):
            random_paths = simulate_infected_patients(n_infected, alpha, K=K, size=n_runs)

            for j in range(n_runs):
                alpha_death_curves[j, t:t+K] += random_paths[j][1:T - t + 1]

        mean_death_curve = alpha_death_curves.mean(0)
    #         all_mean_squared_errors = mean_squared_error(alpha_death_curves, mean_death_curve)
        d_mse = mean_squared_error(d, mean_death_curve)
        print(f"\nConfidence Interval={lower_boundary, higher_boundary}\nTrue Mean Squared Error={true_mean_squared_error}\nIncluded?={true_mean_squared_error >= lower_boundary and true_mean_squared_error <= higher_boundary}\n")

# %% codecell
estimate_hazard_rate(
    smooth_coronavirus_df['Infected'].values,
    smooth_coronavirus_df['Dead'].values
)
# %% codecell
