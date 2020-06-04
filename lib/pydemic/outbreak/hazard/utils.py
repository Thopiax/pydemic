import sys

# Logger Function

def build_logger(verbose, prefix="default"):
    def _silent_log(*s):
        pass

    def _log(*s):
        print(f"{prefix}:", *s)

    return _log if verbose else _silent_log


# Loss Functions

def naive_error(realization, estimate):
    return realization - estimate

def scale_free_error(realization, estimate, weights=None):
    errors = naive_error(realization, estimate)
    scale_coefficient = np.mean(np.abs(np.diff(realization))) if realization.shape[0] > 1 else realization[0]

    if weights is not None:
        assert weights.shape[0] == errors.shape[0]

        errors = weights * errors
        scale_coefficient = scale_coefficient * np.max(weights)

    return errors / scale_coefficient


def mean_absolute_error(realization, estimate, error=naive_error, **kwargs):
    errors = error(realization, estimate, **kwargs)

    return np.mean(np.abs(errors))