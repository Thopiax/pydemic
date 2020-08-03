import numpy as np

from typing import Optional, List, Tuple, Collection

from skopt import gbrt_minimize, optimizer, gp_minimize, Space
from skopt.callbacks import DeltaYStopper, DeadlineStopper

from epydemic.outcome.optimizer.cache import OutcomeOptimizerCache
from epydemic.outcome.optimizer.loss import BaseOutcomeLoss
from epydemic.outcome.optimizer.utils import get_initial_points


def add_initial_points(x0: Optional[List[Collection[float]]], y0: Optional[List[Collection[float]]],
                       loss, initial_parameter_points: Optional[List[Collection[float]]]):
    if initial_parameter_points is None or len(initial_parameter_points) == 0:
        return x0, y0

    initial_loss_points = [loss(parameters=parameter_point) for parameter_point in initial_parameter_points]

    if x0 is None and y0 is None:
        return initial_parameter_points, initial_loss_points

    return x0 + initial_parameter_points, y0 + initial_loss_points


class OutcomeOptimizer:
    def __init__(self, model, skopt_minimize: optimizer = gbrt_minimize,
                 tag: str = "", verbose: bool = True, random_state: int = 1, **kwargs):
        self.dimensions = model.dimensions

        self.cache = OutcomeOptimizerCache(model.cache_path)

        self.random_state = random_state
        self.verbose = verbose

        self.skopt_minimize = skopt_minimize

        self.tag = f"{model.distribution.name}__{self.skopt_minimize.__name__}"

        if tag != "":
            self.tag += "__" + tag

    def load_cached_result(self, loss):
        return self.cache.get(tag=f"{self.tag}_{loss.tag}")

    def cache_result(self, loss, result):
        return self.cache.add(f"{self.tag}_{loss.tag}", result, to_disk=True)

    def optimize(self, loss: BaseOutcomeLoss, n_calls: int = 300, max_calls: int = 300, n_random_starts: int = 50,
                 use_cache=True, initial_parameter_points: Optional[np.array] = None, delta: float = 0.0, **kwargs):
        result = self.load_cached_result(loss)

        x0, y0 = get_initial_points(result)
        x0, y0 = add_initial_points(x0, y0, loss, initial_parameter_points)

        # ensure max_calls is respected over past calls
        n_past_calls = 0 if x0 is None else len(x0)
        n_calls = min(n_calls, max_calls - n_past_calls)

        # skopt minimize requires at least the n_random_starts
        if n_calls < n_random_starts:
            return result

        result = self.skopt_minimize(loss, dimensions=self.dimensions, acq_func="LCB",
                                     n_calls=n_calls, n_random_starts=n_random_starts,
                                     verbose=self.verbose, random_state=self.random_state,
                                     callback=[DeltaYStopper(delta)], x0=x0, y0=y0, n_jobs=-1, **kwargs)

        if use_cache:
            self.cache_result(loss, result)

        return result
