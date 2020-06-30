from skopt import gbrt_minimize, optimizer
from skopt.callbacks import DeltaYStopper, DeadlineStopper

from outcome.optimizer.cache import OutcomeOptimizerCache
from outcome.optimizer.loss import OutcomeOptimizerLoss
from outcome.optimizer.utils import get_initial_points


class OutcomeOptimizer:
    def __init__(self, model, skopt_minimize: optimizer = gbrt_minimize, tag: str = "",
                 verbose: bool = True, random_state: int = 1, **kwargs):
        self.model = model

        self.cache = OutcomeOptimizerCache(self, model)

        self.dimensions = model.dimensions

        self.random_state = random_state
        self.verbose = verbose

        self.skopt_minimize = skopt_minimize

        self.tag = self.skopt_minimize.__name__

        if tag != "":
            self.tag += "__" + tag

    def optimize(self, t: int, n_calls=200, max_calls=500, n_random_starts=20, delta=0.01, use_cache=True,
                 dry_run=False, **kwargs):
        tag = f"{self.tag}_{t}"
        cached_result = self.cache.get(tag)

        if cached_result is not None:
            if dry_run:
                return cached_result

            x0, y0 = get_initial_points(cached_result)
            n_past_calls = len(x0)
        else:
            x0, y0 = None, None
            n_past_calls = 0

        # ensure max_calls is respected over past calls
        n_calls = min(n_calls, max_calls - n_past_calls)

        # skopt minimize requires at least 20 calls
        if n_calls < 20:
            return cached_result

        loss = OutcomeOptimizerLoss(self.model, t)

        result = self.skopt_minimize(loss, dimensions=self.dimensions, acq_func="LCB",
                                     n_calls=n_calls, n_random_starts=n_random_starts,
                                     verbose=self.verbose, random_state=self.random_state,
                                     # callback=[DeltaYStopper(delta), DeadlineStopper(600)],
                                     x0=x0, y0=y0, n_jobs=-1, **kwargs)

        if use_cache:
            self.cache.add(tag, result, to_disk=True)

        return result
