import os
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import sys

from epydemic.inversion.individual import FatalityIndividualModel

sys.path.append(Path(__file__).parent.parent.parent)

from epydemic.utils.regions import OECD
from epydemic.utils.helpers import build_coronavirus_epidemic
from epydemic.inversion.population.models import AnalyticalPopulationModel
from epydemic.inversion.population.plot import plot_partial_dependence, plot_prediction, plot_individual_rates
from epydemic.outbreak import OutbreakTimeWindow
from epydemic.utils.path import DATA_ROOTPATH, ROOTPATH


def build_oecd_full_outbreak_df():
    # if os.path.isfile(DATA_ROOTPATH / "oecd_outbreak_record.csv"):
    #     return pd.read_csv(DATA_ROOTPATH / "oecd_outbreak_record.csv", index_col=0)

    epidemic = build_coronavirus_epidemic()

    ptr = 0

    oecd_full_outbreak_df = pd.DataFrame(columns=["region", "otw_start", "otw_end", "alpha", "beta", "eta", "loss", "MTTF"])

    for region, outbreak in epidemic.get_outbreaks(OECD).items():
        # skip outbreaks with fewer than 1,000 deaths
        if outbreak.ffx(1_000, x_type="deaths") is None:
            continue

        for otw in outbreak.expanding_windows():
            model = AnalyticalPopulationModel(otw, verbose=False)

            model.fit(
                n_calls=100,
                n_retries=0,
                delta=0.05,
            )

            model_stats = model.individual_model.describe()

            payload = [region, otw.start, otw.end, *model.best_parameters, model.best_loss, model_stats["mean"]]
            oecd_full_outbreak_df.loc[ptr, :] = payload

            ptr += 1
            print(region, payload)

    oecd_full_outbreak_df.to_csv(DATA_ROOTPATH / "oecd_outbreak_record.csv")

    return oecd_full_outbreak_df

df = None
data = None

if __name__ == "__main__":
    data: pd.DataFrame = build_oecd_full_outbreak_df()

    df = pd.DataFrame(columns=['mean', 'median', 'std', 'variance', 'skew', 'kurtosis', 'entropy', 'p05', 'p95'])

    for index, row in data.iterrows():
        model = FatalityIndividualModel(parameters=[row["alpha"], row["beta"], row["eta"]])

        print(model.describe())

        df.loc[row["region"], :] = model.describe()

    # data = main()

