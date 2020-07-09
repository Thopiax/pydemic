import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from epydemic.inversion.individual import BaseIndividualModel
from epydemic.utils.decorators import save_figure


def plot_individual_rates(model: BaseIndividualModel, region: str, xlim_upper: int = 20):
    ax = plt.gca()

    ax.set_title(f"{region} - Individual Rates")
    ax.set_xlabel("Days since diagnosis (k)")
    ax.set_ylabel("Probability of death")

    fatality_rate, hazard_rate = model.fatality_rate, model.hazard_rate

    xlim_upper_padding = xlim_upper - len(model.fatality_rate)

    if xlim_upper_padding > 0:
        # pad cfr rate with zeros
        fatality_rate = np.append(fatality_rate, [0 for _ in range(xlim_upper_padding)])
        # pad hazard rate with last value
        hazard_rate = np.append(hazard_rate, [hazard_rate[-1] for _ in range(xlim_upper_padding)])

    ax.set_xlim(0, xlim_upper)

    text = AnchoredText(
        f"alpha={model.alpha:.3}\nbeta={model.beta:.3}\neta={model.eta:.3}",
        loc="upper center"
    )
    ax.add_artist(text)

    plt.plot(fatality_rate, label="Fatality Rate")
    plt.plot(hazard_rate, label="Hazard Rate")

    plt.show()
