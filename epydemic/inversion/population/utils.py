import numpy as np
from epydemic.inversion.individual.exceptions import InvalidParameters


def verify_prediction(prediction):
    if np.isnan(prediction).any() or np.isinf(prediction).any():
        raise InvalidParameters

    return True
