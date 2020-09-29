import numpy as np
import pandas as pd

from resolution_delay.distributions.exceptions import InvalidParameterError

MAX_RATE_PPF = (10_000 - 1) / 10_000 # only 1 in 10_000 cases are not considered
MAX_SUPPORT_SIZE = 60  # days
MAX_RATE_VARIANCE = 1000  # days => std < 30 days


def verify_rate(rate: pd.Series):
    if np.isnan(rate).any() or np.isinf(rate).any():
        raise InvalidParameterError


def describe(random_variable):
    mean, var, skew, kurtosis = random_variable.stats(moments="mvsk")

    interval_size = 0.95
    lower, upper = random_variable.interval(interval_size)

    return pd.Series(dict(
        mean=float(mean),
        std=random_variable.std(),
        variance=float(var),
        skew=float(skew),
        kurtosis=float(kurtosis),
        median=random_variable.median(),
        entropy=random_variable.entropy(),
        lower_interval_bound=lower,
        upper_interval_bound=upper,
        interval_size=interval_size
    ))