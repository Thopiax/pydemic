import numpy as np
import pandas as pd

from outcome.distribution.exceptions import InvalidParameterError

MAX_RATE_PPF = 0.999  # only 1 in 1000 cases are not considered
MAX_RATE_SUPPORT_SIZE = 60  # days
MAX_RATE_VARIANCE = 1000  # days => std < 30 days


def verify_random_variable(random_variable):
    pass


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