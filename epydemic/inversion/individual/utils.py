import numpy as np

from .exceptions import InvalidParameters


def verify_valid_K(K):
    if np.isnan(K) or np.isinf(K) or K < 1:
        raise InvalidParameters

    return True


def build_hazard_rate(incidence_rate):
    K = len(incidence_rate)

    cumulative_incidence_rate = np.cumsum(incidence_rate)

    result = np.zeros(K)
    result[0] = incidence_rate[0]
    result[1:] = incidence_rate[1:] / (1 - cumulative_incidence_rate[:-1])

    return result
