import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from skopt.plots import plot_objective

from epydemic.inversion.population.models.base import AbstractPopulationModel
from epydemic.utils.decorators import save_figure


@save_figure(lambda model, **kwargs: f"predictions/{model.tag}.pdf")
def plot_prediction(model: AbstractPopulationModel, cumulative: bool = False, parameters = None):

    if parameters is not None:
        model.parameters = parameters

    ax = plt.gca()

    ax.set_title(f"{model.otw.region} - Population {'Cumulative' if cumulative else 'Daily'} Deaths")
    ax.set_xlabel("Days since outbreak began (t)")
    ax.set_ylabel("Number of deaths")

    plt.plot(model.otw.deaths.index, model.otw.deaths.values, label="y_true")
    plt.plot(model.otw.deaths.index, model.predict(), label="y_pred")

    plt.show()


@save_figure(lambda model: f"partial_dependence/{model.tag}.pdf")
def plot_partial_dependence(model: AbstractPopulationModel):
    plot_objective(model.learner.skopt_result, size=5, n_samples=100, minimum='result',
                   sample_source='result', dimensions=["alpha", "beta", "lambda"])

    plt.show()


@save_figure(lambda model, **kwargs: f"individual_rates/{model.tag}.pdf")
def plot_individual_rates(model: AbstractPopulationModel, xlim_upper: int = 20):
    ax = plt.gca()

    ax.set_title(f"{model.otw.region} - Individual Rates")
    ax.set_xlabel("Days since diagnosis (k)")
    ax.set_ylabel("Probability of death")

    fatality_rate, hazard_rate = model.individual_model.fatality_rate, model.individual_model.hazard_rate

    xlim_upper_padding = xlim_upper - model.individual_model.K

    if xlim_upper_padding > 0:
        # pad fatality rate with zeros
        fatality_rate = np.append(fatality_rate, [0 for _ in range(xlim_upper_padding)])
        # pad hazard rate with last value
        hazard_rate = np.append(hazard_rate, [hazard_rate[-1] for _ in range(xlim_upper_padding)])

    ax.set_xlim(0, xlim_upper)

    text = AnchoredText(
        f"alpha={model.individual_model.alpha:.3}\nbeta={model.individual_model.beta:.3}\neta={model.individual_model.eta:.3}",
        loc="upper center"
    )
    ax.add_artist(text)

    plt.plot(fatality_rate, label="Fatality Rate")
    plt.plot(hazard_rate, label="Hazard Rate")

    plt.show()
