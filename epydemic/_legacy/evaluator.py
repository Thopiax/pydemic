import os

import matplotlib.pyplot as plt
import numpy as np
from skopt.plots import plot_objective, plot_convergence

from .estimator import HazardParameterEstimator
from .exceptions import *
from ..utils.logging import build_logger
from ..utils.loss import scale_free_error, mean_absolute_error


class HazardEstimateEvaluator:
    def __init__(self, model, test_days=0, test_pre_peak=False, verbose=False, warm_start=True,
                 warm_start_min_deaths_observed=10, cumulative_error=True, error=scale_free_error, **kwargs):
        self.model = model
        self.estimator = HazardParameterEstimator(model, verbose=verbose, **kwargs)

        self.test_pre_peak = test_pre_peak

        # perform a pre_peak test only if peak has not been reached
        if self.test_pre_peak and self.model.outbreak.is_peak_reached():
            outbreak_peak = self.model.outbreak.peak_id()

            assert 0 <= test_days < outbreak_peak
            self.test_split = outbreak_peak - test_days

        # otherwise, perform test at the end of the outbreak
        else:
            assert 0 <= test_days < model.T
            self.test_split = model.T - test_days

        # warm start
        self.warm_start = warm_start
        self.warm_start_min_deaths_observed = warm_start_min_deaths_observed

        self.t0 = self.model.outbreak.ffx_deaths(self.warm_start_min_deaths_observed)[0] if self.warm_start else 0
        assert self.t0 < self.test_split

        # error parameters
        self.error = error
        self.cumulative_error = cumulative_error

        # scoring functions
        self.training_score = self._build_score(self._training_slice)
        self.testing_score = self._build_score(self._testing_slice)

        self.log = build_logger(verbose, prefix=f"{self.model.outbreak.region}.Validator")

    @property
    def configuration_prefix(self):
        result = [self.model.__str__(), str(self.test_split)]

        # TODO: convert parameters to class where we can move these processings to __str__ methods

        if self.test_pre_peak:
            result.append("pre_peak")

        # cumulative_error
        if self.cumulative_error:
            if self.cumulative_error == "both":
                result.append("both")

            elif self.cumulative_error == "resolved":
                result.append("resolved_weigth")

            else:
                result.append("cumulative")
        else:
            result.append("daily")

        result.append("scaled_error" if self.error == scale_free_error else "naive_error")
        result.append(f"{self.estimator.dimensions_name}_dimensions")

        return "_".join(result)

    @property
    def optimizer_result(self):
        return self.estimator.result

    @property
    def best_estimate(self):
        self.model.parameters = self.best_training_parameters
        return self.model.predict()

    @property
    def best_training_parameters(self):
        return tuple(self.optimizer_result.x) if self.optimizer_result else (None, None, None)

    @property
    def best_training_score(self):
        return self.optimizer_result.fun if self.optimizer_result else None

    def train(self, n_iters=100, use_cache=True, dry_run=False, max_iters=500, overwrite_cache=False):
        previous_result = self.estimator.load_result(self.configuration_prefix)

        if use_cache and previous_result is not None:
            # if max iterations have already been calculated, return right away
            if self.estimator.iters >= max_iters:
                return self.best_training_parameters, self.best_training_score

            # otherwise, update the optimizer with the previously seen samples
            self.estimator.tell_samples_from_result()

        if dry_run is not True:
            self.log(f"Start training estimator until T={self.test_split}...")
            self.estimator.run(self.training_score, n_iters)

            if use_cache or overwrite_cache:
                self.estimator.save_result(self.configuration_prefix)

            self.log("Finish training estimator.")

        return self.best_training_parameters, self.best_training_score

    def test(self):
        if self.test_split == self.model.T:
            raise NoTestingDataException

        return self.testing_score(self.best_training_parameters)

    def train_test_split(self, data):
        return self._training_slice(data), self._testing_slice(data)

    def _training_slice(self, data):
        return data[self.t0:self.test_split]

    def _testing_slice(self, data):
        return data[self.test_split:]

    def _build_score(self, _slice):
        def _score(parameters, cumulative_error=None):
            cumulative_error = self.cumulative_error if cumulative_error is None else cumulative_error

            try:
                self.model.parameters = tuple(parameters)
                estimate = _slice(self.model.predict())
                realization = _slice(self.model.deaths)

                if cumulative_error is True:
                    estimate = np.cumsum(estimate)
                    realization = np.cumsum(realization)
                elif cumulative_error is "both":
                    # mix both errors
                    return mean_absolute_error(realization, estimate, error=self.error) + mean_absolute_error(
                        np.cumsum(realization), np.cumsum(estimate), error=self.error)
                elif cumulative_error is "resolved":
                    weights = _slice(self.model.outbreak.resolved_case_rate)
                    # mix both errors and include alpha as an error
                    return mean_absolute_error(realization, estimate, error=self.error, weights=weights)

                return mean_absolute_error(realization, estimate, error=self.error)

            except InvalidHazardRateException:
                # return a much larger number to discourage the combination in later iterations
                return 1_000

        return _score


# %% md

### Plotting

# %%

from matplotlib.offsetbox import AnchoredText
from matplotlib.cm import tab20

def plot_fatality_rates(estimators, right_truncation=100):
    ax = plt.gca()
    ax.set_xlim(left=0, right=right_truncation)

    for i, estimator in enumerate(estimators):
        fatality_rate = estimator.fatality_rate

        if type(right_truncation) is int:
            rate_padding = right_truncation - estimator.K

            fatality_rate = np.append(fatality_rate, [0 for _ in range(rate_padding)])

        ax.plot(fatality_rate, label=estimator.outbreak.region, c=tab20.colors[i])

    plt.legend()
    plt.show()


CUMULATIVE_ERROR_NAME = {
    False: "Daily",
    True: "Cumulative",
    "both": "MCD",
    "resolved": "MARSE"
}

class HazardEstimateInspector:
    def __init__(self, validator, iteration_name, resultpath="../plots/coronavirus", save_figure=False):
        self.validator = validator
        self.estimator = validator.estimator

        self.resultpath = os.path.join(resultpath, iteration_name, self.estimator.outbreak.region)

        if os.path.isdir(self.resultpath) is False:
            if os.path.isdir(os.path.join(resultpath, iteration_name)) is False:
                os.mkdir(os.path.join(resultpath, iteration_name))
            os.mkdir(self.resultpath)

        self.save_figure = save_figure

    def _plot_fatality_estimate(self, ax, estimate, realization, cumulative_error):
        ax.set_title(f"{CUMULATIVE_ERROR_NAME[cumulative_error]} Fatality Estimate")
        ax.set_xlabel("Days since outbreak began (t)")
        ax.set_ylabel("Number of Deaths (d_t)")

        training_estimate, test_estimate = self.validator.train_test_split(estimate)
        training_realization, test_realization = self.validator.train_test_split(realization)

        ax.plot(np.append(training_realization, test_realization), label="realization")

        if test_estimate.shape[0] == 0:
            if self.validator.cumulative_error == "both" or cumulative_error and self.validator.cumulative_error or not (
                cumulative_error or self.validator.cumulative_error):
                mase = self.validator.training_score(self.validator.best_training_parameters, cumulative_error)

                text = AnchoredText(f"MASE={mase:.3}", loc="upper left")
                ax.add_artist(text)

            ax.plot(training_estimate, c='red', lw=3, label="estimate")
        else:
            if self.validator.cumulative_error == "both" or cumulative_error and self.validator.cumulative_error or not (
                cumulative_error or self.validator.cumulative_error):
                training_mase = self.validator.training_score(self.validator.best_training_parameters, cumulative_error)
                testing_mase = self.validator.training_score(self.validator.best_training_parameters, cumulative_error)

                text = AnchoredText(
                    f"Train MASE: {self.validator.best_training_score:.3}\nTest  MASE: {self.validator.test():.3}",
                    loc="upper left")
                ax.add_artist(text)

            ax.plot(np.append(training_estimate, test_estimate), c='red', lw=3, label="test estimate")
            ax.plot(training_estimate, c='orange', lw=3, label="training estimate")

        ax.legend(loc="lower right" if cumulative_error else "upper right")

    def _plot_case_fatality_density(self, ax, right_truncation=None):
        ax.set_title("Case Fatality Density")
        ax.set_xlabel("Days since infection confirmation (k)")
        ax.set_ylabel("Probability of Death | Death")

        ax.set_xlim(0, right_truncation if right_truncation else self.estimator.K)

        text = AnchoredText(
            f"alpha={self.validator.best_training_parameters[0]:.3}\nbeta={self.validator.best_training_parameters[1]:.3}\nlambda={self.validator.best_training_parameters[2]:.3}",
            loc="upper center")
        ax.add_artist(text)

        fatality_rate = self.estimator.fatality_rate
        hazard_rate = self.estimator.hazard_rate

        if type(right_truncation) is int:
            rate_padding = right_truncation - self.estimator.K

            fatality_rate = np.append(fatality_rate, [0 for _ in range(rate_padding)])
            hazard_rate = np.append(hazard_rate, [hazard_rate[-1] for _ in range(rate_padding)])

        ax.plot(fatality_rate, label="Fatality rate")
        ax.plot(hazard_rate, label="Hazard rate")

        ax.legend()

    def _plot_convergence(self, ax):
        ax.set_title("Convergence")
        ax.set_xlabel("MASE")
        ax.set_ylabel("Iterations")

        plot_convergence(self.validator.optimizer_result, dimensions=["alpha", "beta", "lambda"], ax=ax)

    def plot_estimate_summary(self, right_truncation=None):
        best_estimate = self.validator.best_estimate
        realization = self.estimator.deaths

        fig, axes = plt.subplots(ncols=2, nrows=2)
        fig.suptitle(f"{self.estimator.outbreak.region} - {self.validator.configuration_prefix}")

        self._plot_fatality_estimate(axes[0][0], best_estimate, realization, cumulative_error=False)
        self._plot_fatality_estimate(axes[0][1], np.cumsum(best_estimate), np.cumsum(realization),
                                     cumulative_error=True)
        self._plot_case_fatality_density(axes[1][0], right_truncation=right_truncation)
        self._plot_convergence(axes[1][1])

        if self.save_figure:
            filepath = os.path.join(self.resultpath, f"{self.validator.configuration_prefix}_estimate_summary.pdf")

            plt.savefig(filepath)

        plt.show()

    def plot_optimization_objective(self):
        axes = plot_objective(self.validator.optimizer_result, size=4, n_samples=100, minimum='result',
                              sample_source='result', dimensions=["alpha", "beta", "lambda"])

        plt.suptitle(f"{self.estimator.outbreak.region}")

        if self.save_figure:
            filepath = os.path.join(self.resultpath, f"{self.validator.configuration_prefix}_optimization_objective.pdf")

            plt.savefig(filepath)

        plt.show()

