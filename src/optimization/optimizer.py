import numpy as np

from typing import Optional, List, Collection

from skopt import gbrt_minimize, forest_minimize, gp_minimize, optimizer
from skopt.callbacks import DeltaYStopper

from optimization.cache import Cache
from optimization.loss.base import BaseLoss
from optimization.utils import get_initial_points


class Optimizer:
    def __init__(self, dimensions, cache_path,
                 skopt_minimize: optimizer = forest_minimize, acq_func: str = "LCB", tag: str = "", **kwargs):
        self.cache = Cache(cache_path)

        self.dimensions = dimensions

        self.skopt_minimize = skopt_minimize
        self.acq_func = acq_func

        self.tag = self.skopt_minimize.__name__

        if tag != "":
            self.tag += "__" + tag

    @staticmethod
    def add_initial_points(x0: Optional[List[Collection[float]]], y0: Optional[List[Collection[float]]],
                           loss, initial_parameter_points: Optional[List[Collection[float]]]):
        if initial_parameter_points is None or len(initial_parameter_points) == 0:
            return x0, y0

        initial_loss_points = [loss(parameters=parameter_point) for parameter_point in initial_parameter_points]
        print(initial_loss_points)

        if x0 is None and y0 is None:
            return initial_parameter_points, initial_loss_points

        return x0 + initial_parameter_points, y0 + initial_loss_points

    def load_cached_result(self, loss: BaseLoss):
        return self.cache.get(f"{self.tag}_{self.acq_func}_{loss.tag}")

    def cache_result(self, loss: BaseLoss, result):
        return self.cache.add(f"{self.tag}_{self.acq_func}_{loss.tag}", result, to_disk=True)

    def optimize(self, loss: BaseLoss, n_calls: int = 100, max_calls: int = 200, n_random_starts: int = 30,
                 use_cache=True, initial_parameter_points: Optional[np.array] = None, delta: float = 0.0,
                 random_state: int = 1, verbose=True, **kwargs):
        result = self.load_cached_result(loss)

        x0, y0 = get_initial_points(result)
        if len(initial_parameter_points) > 0 and x0 is not None:
            x0 += initial_parameter_points

        # ensure max_calls is respected over past calls
        n_past_calls = 0 if x0 is None else len(x0)
        n_calls = min(n_calls, max_calls - n_past_calls)

        # skopt minimize requires at least the n_random_starts
        if n_calls < n_random_starts:
            print("Returning due to few remaining calls (n_calls < n_random_starts)")
            return result

        opts = dict(
            dimensions=self.dimensions, acq_func=self.acq_func, n_calls=n_calls,
            n_random_starts=n_random_starts, callback=[DeltaYStopper(delta)],
            x0=x0, y0=y0, n_jobs=-1, random_state=random_state, verbose=verbose
        )
        opts.update(kwargs)

        result = self.skopt_minimize(loss,  **opts)

        if use_cache:
            self.cache_result(loss, result)

        return result
