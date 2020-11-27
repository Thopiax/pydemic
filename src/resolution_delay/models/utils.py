import numpy as np

from resolution_delay.distributions.exceptions import InvalidParameterError


def verify_pred(pred: np.array):
    if np.isnan(pred).any() or np.isinf(pred).any() or np.equal(pred, 0).all():
        raise InvalidParameterError


def verify_probability(prob):
    if 0 < prob < 1:
        return

    raise InvalidParameterError
