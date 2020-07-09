import numpy as np
import pandas as pd
import scipy
from scipy.stats import stats

from .exceptions import InvalidParametersError

MAX_DEATH_DELAY_THRESHOLD = 60 # days
MAX_DEATH_DELAY_VARIANCE = 900 # days => std < 30 days
MAX_PPF = 0.999 # 99.9 % of occurences happen before this day


def verify_distribution(rv: stats):
    if rv.median() > MAX_DEATH_DELAY_THRESHOLD or rv.var() > MAX_DEATH_DELAY_VARIANCE:
        raise InvalidParametersError

    return True


def build_distribution_rates(rv: scipy.stats, max_delay: int = MAX_DEATH_DELAY_THRESHOLD):
    verify_distribution(rv)

    # TODO: change below
    delay = max_delay
    support = np.arange(delay)

    incidence_rate = np.trim_zeros(rv.pdf(support), trim="b")
    delay = len(incidence_rate)

    support = np.arange(delay)

    hazard_rate = incidence_rate / rv.sf(support)

    return incidence_rate, hazard_rate


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
