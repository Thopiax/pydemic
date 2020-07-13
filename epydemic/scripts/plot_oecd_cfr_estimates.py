import matplotlib.pyplot as plt
import pandas as pd

from cfr.utils import build_estimates
from epydemic.data.covid19 import load_coronavirus_epidemic

from epydemic.cfr.models.naive import FatalityCFRModel, RecoveryCFRModel, MixedCFRModel
from epydemic.cfr.models.corrected import FatalityCorrectedCFRModel, RecoveryCorrectedCFRModel, MixedCorrectedCFRModel
from epydemic.data.regions import OECD
from outbreak import Outbreak


def plot_outbreak_estimates(outbreak: Outbreak):
    estimates = build_estimates(outbreak,
                                FatalityCFRModel, RecoveryCFRModel, MixedCFRModel,
                                FatalityCorrectedCFRModel, RecoveryCorrectedCFRModel, MixedCorrectedCFRModel,
                                window_size=3)

    latest_estimate = FatalityCFRModel(outbreak).estimate(-1)

    ax: plt.axis = plt.gca()
    ax.set_title(f"CFR - {outbreak.region}")
    estimates.plot(ax=ax)
    ax.hlines(latest_estimate, 0, estimates.index[-1], colors="red", linestyles="--")
    plt.show()

    return estimates


def main():
    epidemic = load_coronavirus_epidemic(include_policies=False)

    OECD_outbreak = epidemic.combine("OECD", *OECD)

    plot_outbreak_estimates(OECD_outbreak)

    for _, outbreak in epidemic[OECD]:
        plot_outbreak_estimates(outbreak)


if __name__ == '__main__':
    main()
