import os
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.insert(Path(__file__).parent.parent.parent)

from epydemic.utils.regions import OECD
from epydemic.utils.helpers import build_coronavirus_epidemic
from epydemic.inversion.population.models import AnalyticalPopulationFatalityModel
from epydemic.inversion.population.plot import plot_partial_dependence, plot_prediction
from epydemic.outbreak import OutbreakTimeWindow
from epydemic.utils.path import DATA_ROOTPATH, ROOTPATH


def build_oecd_full_outbreak_df():
    # if os.path.isfile(DATA_ROOTPATH / "oecd_full_outbreak_df.csv"):
    #     return pd.read_csv(DATA_ROOTPATH / "oecd_full_outbreak_df.csv", index_col=0)

    epidemic = build_coronavirus_epidemic()

    ptr = 0
    oecd_full_outbreak_df = pd.DataFrame(columns=["region", "alpha", "beta", "lambda", "loss"])
    oecd_expected_full_outbreak_df = pd.DataFrame(columns=["region", "alpha", "beta", "lambda", "loss"])

    for region, outbreak in epidemic.get_outbreaks(OECD).items():

        # skip outbreaks with fewer than 1,000 deaths
        if outbreak.ffx(1_000, x_type="deaths") is None:
            continue

        otw = OutbreakTimeWindow(outbreak)

        print(region, "started")

        model = AnalyticalPopulationFatalityModel(otw, verbose=True)

        model.fit(n_calls=1_000, n_retries=10, xi=300, delta=0.005, model_queue_size=100)

        payload = [region, *model.best_parameters, model.best_loss]
        oecd_full_outbreak_df.loc[ptr, :] = payload
        print(region, "actual", payload)

        plot_prediction(model, parameters=model.best_parameters, save_figure=False)

        parameters, loss = model.learner.best_expected_loss
        payload = [region, *parameters, loss]

        print(region, "expected", payload)
        oecd_expected_full_outbreak_df.loc[ptr, :] = payload

        plot_prediction(model, parameters=parameters, save_figure=False)

        ptr += 1

    oecd_full_outbreak_df.to_csv(DATA_ROOTPATH / "oecd_outbreak_record.csv")
    oecd_expected_full_outbreak_df.to_csv(DATA_ROOTPATH / "oecd_expected_outbreak_record.csv")

    return oecd_full_outbreak_df, oecd_expected_full_outbreak_df

def main():
    epidemic = build_coronavirus_epidemic()

    result = {}

    for region, outbreak in epidemic.get_outbreaks(OECD).items():
        # evaluator = CFREvaluator(outbreak,
        #                          NaiveCFREstimator,
        #                          ResolvedCFREstimator,
        #                          MortalityRateCFREstimator,
        #                          ExpectedMortalityRateCFREstimator
        #             )
        #
        # result[region] = evaluator.estimates
        # result[region].to_csv(DATA_ROOTPATH / "cfr" / f"{region}_weekly_estimates.csv")

        result[region] = pd.read_csv(DATA_ROOTPATH / "cfr" / f"{region}_weekly_estimates.csv")

    return result

data = None

if __name__ == "__main__":
    build_oecd_full_outbreak_df()
    # data = main()

