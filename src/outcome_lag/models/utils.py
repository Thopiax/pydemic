import numpy as np

from outcome_lag.distributions.base import BaseOutcomeLagDistribution


def expected_case_outcome_lag(t: int, cases: np.array, incidence_rate: np.array):
    K = min(t + 1, len(incidence_rate))

    # select cases subset from (t - K, t] with size K
    cases_subset = cases[(t - K) + 1:t + 1]

    # reverse incidence_rate and take subset from [0, K)
    reverse_incidence_rate = incidence_rate[K-1::-1]

    return np.sum(cases_subset * reverse_incidence_rate)


def test_expected_case_outcome_lag():
    K = 3

    cases = np.array(range(100))

    # p_0 = 1/2, p_1 = 1/4, p_2 = 1/8
    incidence_rate = np.array([1 / (2 ** k) for k in range(1, K + 1)])

    assert expected_case_outcome_lag(1, cases, incidence_rate) == 1 / 2
    assert expected_case_outcome_lag(2, cases, incidence_rate) == (2 / 2) + (1 / 4)

    assert expected_case_outcome_lag(80, cases, incidence_rate) == (80 / 2) + (79 / 4) + (78 / 8)

