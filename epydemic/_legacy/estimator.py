import os

from skopt import Optimizer, dump, load
from skopt.space import Real

from .exceptions import *
from ..utils.logging import build_logger

# variables below found from summary statistics in study
WUHAN_BETA = 1.8429282373343958
WUHAN_LAMBDA = 10.018568258846706

INITIAL_DIMENSIONS = [
    Real(0.01, 0.20),
    Real(0.5, 10.0, WUHAN_BETA),
    Real(1.0, 20.0, WUHAN_LAMBDA)
]

RELAXED_DIMENSIONS = [
    Real(0.01, 1.0),
    Real(0.1, 40.0, WUHAN_BETA),
    Real(0.1, 200.0, WUHAN_LAMBDA)
]

TIGHTER_DIMENSIONS = [
    Real(0.01, 1.0),
    Real(0.1, 10.0, WUHAN_BETA),
    Real(3.0, 30.0, WUHAN_LAMBDA)
]

DIMENSION_NAME_MAP = {
    INITIAL_DIMENSIONS: "initial",
    RELAXED_DIMENSIONS: "relaxed",
    TIGHTER_DIMENSIONS: "tighter",
}


class HazardParameterEstimator:
    def __init__(self, model, n_random_starts=10, verbose=False,
                 exploration_exploitation_coefficient=0.01, dimensions=None, resultpath="../results/coronavirus",
                 **kwargs):
        self.model = model

        self._is_optimized = False
        self._result = None

        self.resultpath = os.path.join(resultpath, model.outbreak.region)

        self.dimensions = dimensions if dimensions else INITIAL_DIMENSIONS

        self.optimizer = Optimizer(
            acq_func="PI",
            acq_optimizer="lbfgs",
            acq_func_kwargs={
                "xi": exploration_exploitation_coefficient,
                "kappa": exploration_exploitation_coefficient
            },
            random_state=self.model.random_state,
            dimensions=self.dimensions,
            n_random_starts=n_random_starts,
            **kwargs
        )

        self.log = build_logger(verbose, prefix=f"{self.model.outbreak.region}.Optimizer")

    @property
    def dimensions_name(self):
        if self.dimensions == INITIAL_DIMENSIONS:
            return "initial"
        elif self.dimensions == RELAXED_DIMENSIONS:
            return "relaxed"
        elif self.dimensions == TIGHTER_DIMENSIONS:
            return "tighter"

        return "custom"

    def run(self, score_fun, n_iters):
        self.log("Optimization started.")

        for iter_id in range(n_iters):
            parameters = self.optimizer.ask()

            score = score_fun(parameters)

            current_result = self.optimizer.tell(parameters, score)

            best_score = current_result.fun
            best_parameters = current_result.x

            self.log(
                f"Iteration #{iter_id}: {score} ({parameters[0]}, {parameters[1]}, {parameters[2]}) | best={best_score}, ({best_parameters[0]}, {best_parameters[1]}, {best_parameters[2]}) ")

        self.log("Optimization finished.")

        self._is_optimized = True
        self._result = self.optimizer.get_result()

    def tell_samples_from_result(self):
        if self.result is None:
            self.log("No results to tell optimizer.")
            return

        parameters = self.result.x_iters
        scores = self.result.func_vals

        self.optimizer.tell(parameters, list(scores))
        self.log(f"Optimizer updated with {len(scores)} samples.")

    # return (or load) the skopt result for the current configuration
    @property
    def result(self):
        return self._result

    @property
    def iters(self):
        if self.result:
            return len(self.result.x_iters)

        return 0

    def load_result(self, configuration_prefix):
        filepath = os.path.join(self.resultpath, f"{configuration_prefix}_optimization_result.pkl")
        self.log(f"Loading results from {filepath}...")

        if os.path.isfile(filepath):
            self.log(f"Results found.")
            self._result = load(filepath)
        else:
            self.log(f"[WARN] Results not found.")

        return self._result

    def save_result(self, configuration_prefix):
        if self._result is None:
            raise NoResultException

        if os.path.isdir(self.resultpath) is False:
            os.mkdir(self.resultpath)
            self.log(f"Created folder {self.resultpath}.")

        filepath = os.path.join(self.resultpath, f"{configuration_prefix}_optimization_result.pkl")
        self.log(f"Saving results to {filepath}...")

        dump(self._result, filepath)
        self.log(f"Results saved.")
