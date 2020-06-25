import numpy as np
import pandas as pd

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


def describe_rv(rv):
    mean, var, skew, kurtosis = rv.stats(moments="mvsk")

    interval_size = 0.95
    lower, upper = rv.interval(interval_size)

    return pd.Series(dict(
        mean=float(mean),
        variance=float(var),
        skew=float(skew),
        kurtosis=float(kurtosis),
        median=rv.median(),
        std=rv.std(),
        entropy=rv.entropy(),
        lower_interval_bound=lower,
        upper_interval_bound=upper,
        interval_size=interval_size
    ))
