import os
from typing import Optional

from skopt import gbrt_minimize, gp_minimize, forest_minimize, optimizer, load, dump, expected_minimum
from skopt.callbacks import DeltaYStopper, CheckpointSaver

from epydemic.inversion.population.models.base import PopulationFatalityModel
from utils.path import CACHE_ROOTPATH

from .objective_function import LearnerObjectiveFunction
from .dimensions import DIMENSIONS
from .loss import LearnerLoss, MASELearnerLoss


def _n_calls_remaining(n_calls, n_calls_until_max, min_calls=20):
    return max(min_calls, min(n_calls_until_max, n_calls))


class PopulationFatalityLearner:
    def __init__(self, model: PopulationFatalityModel, loss: Optional[LearnerLoss] = None,
                 skopt_minimize: optimizer = gp_minimize, dimensions_name: str = "relaxed",
                 verbose: Optional[bool] = None, random_state: Optional[int] = None):

        if os.path.isdir(CACHE_ROOTPATH) is False:
            os.mkdir(CACHE_ROOTPATH)

        self.model = model
        self.otw = model.otw

        self.verbose = model.verbose if verbose is None else verbose
        self.random_state = model.random_state if random_state is None else random_state

        self.dimensions_name = dimensions_name
        self.dimensions = DIMENSIONS[dimensions_name]
        self.random_state = random_state

        self.loss = MASELearnerLoss(name="MARSE") if loss is None else loss
        self.objective_function = LearnerObjectiveFunction(self.model, loss=self.loss)

        self.skopt_minimize = skopt_minimize
        self._skopt_result = None

    @property
    def tag(self):
        return f"{self.otw.region}__{self.otw.start}_{self.otw.end}__{str(self.loss)}__{self.dimensions_name}_dims"

    def minimize_loss(self, n_calls=200, n_random_starts=20, n_retries=5, model_queue_size=None, delta=0.01,
                      use_cache=True, overwrite_cache=True, **kwargs):

        x0, y0 = None, None

        for trie_id in range(n_retries):
            if use_cache:
                x0, y0 = self.initial_points

            result = self.skopt_minimize(
                self.objective_function,
                dimensions=self.dimensions,
                acq_func="LCB",
                n_calls=n_calls,
                n_random_starts=n_random_starts,
                verbose=self.verbose,
                random_state=self.random_state,
                model_queue_size=model_queue_size,
                callback=[
                    DeltaYStopper(delta) # stop the optimization if best 5 solutions are delta width apart
                    # CheckpointSaver(self.tag, compress=9)
                ],
                x0=x0,
                y0=y0,
                n_jobs=-1,
                **kwargs
            )

            # reset internal result and initial points
            self._skopt_result = result

            # cache result
            self.save_result(result, overwrite_cache=overwrite_cache)

        return self.best_loss, self.best_parameters

    @property
    def skopt_result(self):
        if self._skopt_result is None:
            self._skopt_result = self.load_result()

        return self._skopt_result

    @property
    def initial_points(self):
        if self.skopt_result is not None:
            return self.skopt_result.x_iters, self.skopt_result.func_vals

        return None, None

    @property
    def best_expected_loss(self):
        if self.skopt_result is not None:
            parameters, loss = expected_minimum(
                self.skopt_result,
                n_random_starts=100,
                random_state=self.random_state
            )

            return tuple(parameters), loss

        return (None, None, None), None

    @property
    def best_parameters(self):
        if self.skopt_result is not None:
            return tuple(self.skopt_result.x)

        return None, None, None

    @property
    def best_loss(self):
        if self.skopt_result is not None:
            return self.skopt_result.fun

        return None

    @property
    def _result_file_extension(self):
        return "gz"

    def load_result(self):
        filepath = CACHE_ROOTPATH / f"{self.tag}.{self._result_file_extension}"

        if os.path.isfile(filepath):
            return load(filepath)

        return None

    def save_result(self, result, overwrite_cache=True):
        if result is None:
            raise Exception

        if overwrite_cache:
            filepath = CACHE_ROOTPATH / f"{self.tag}.{self._result_file_extension}"

            del result.specs['args']['func']
            dump(result, filepath, compress=True)
