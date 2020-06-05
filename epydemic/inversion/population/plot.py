import matplotlib.pyplot as plt
from skopt.plots import plot_objective

from epydemic.inversion.population.models.base import PopulationFatalityModel
from epydemic.utils.decorators import save_figure


@save_figure(lambda model, **kwargs: f"predictions/{model.tag}.pdf")
def plot_prediction(model: PopulationFatalityModel, cumulative: bool = False, parameters = None):

    if parameters is not None:
        model.parameters = parameters

    ax = plt.gca()

    ax.set_title(f"{model.otw.region} - Population {'Cumulative' if cumulative else 'Daily'} Deaths")
    ax.set_xlabel("Days since outbreak began (t)")
    ax.set_ylabel("Number of deaths")

    plt.plot(model.otw.deaths.values, label="y_true")
    plt.plot(model.predict(), label="y_pred")

    plt.show()


@save_figure(lambda model: f"partial_dependence/{model.tag}.pdf")
def plot_partial_dependence(model: PopulationFatalityModel):
    plot_objective(model.learner.skopt_result, size=5, n_samples=100, minimum='result',
                   sample_source='result', dimensions=["alpha", "beta", "lambda"])

    plt.show()
