import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skopt.plots import plot_objective

import sys

from epydemic.outcome.distribution.discrete import NegBinomialOutcomeDistribution
from outcome.models.ensemble.dual import DualOutcomeRegression
from epydemic.outbreak import Outbreak

from outcome.models.exceptions import TrivialTargetError
from outcome.models.fatality import FatalityOutcomeModel
from outcome.models.recovery import RecoveryOutcomeModel
from outcome.optimizer.utils import get_optimal_parameters, get_optimal_loss

sys.path.append(Path(__file__).parent.parent.parent)

from data.regions import OECD
from epydemic.utils.helpers import build_coronavirus_epidemic
from epydemic.utils.path import DATA_ROOTPATH


# def build_oecd_full_outbreak_df():
#     if os.path.isfile(DATA_ROOTPATH / "oecd_outbreak_record.csv"):
#         return pd.read_csv(DATA_ROOTPATH / "oecd_outbreak_record.csv", index_col=0)
#
#     epidemic = build_coronavirus_epidemic()
#
#     ptr = 0
#
#     oecd_full_outbreak_df = pd.DataFrame(
#         columns=["region", "otw_start", "otw_end", "alpha", "beta", "eta", "loss", "MTTF"])
#
#     for region, outbreak in epidemic[OECD].items():
#         # skip outbreaks with fewer than 1,000 deaths
#         if outbreak.ffx(1_000, x_type="deaths") is None:
#             continue
#
#         for otw in outbreak.expanding_windows():
#             model = DualPopulationModel(otw, verbose=True)
#
#             model.fit(
#                 n_calls=50,
#                 n_retries=3,
#                 delta=0.05,
#             )
#
#             model_stats = model.individual_model.describe()
#
#             payload = [region, otw.start, otw.end, *model.best_parameters, model.best_loss, model_stats["mean"]]
#             oecd_full_outbreak_df.loc[ptr, :] = payload
#
#             ptr += 1
#             print(region, payload)
#
#     oecd_full_outbreak_df.to_csv(DATA_ROOTPATH / "oecd_outbreak_record.csv")
#
#     return oecd_full_outbreak_df
#

if __name__ == "__main__":
    outbreak = Outbreak.from_csv("Germany")

    for model in [FatalityOutcomeModel(outbreak, NegBinomialOutcomeDistribution()), RecoveryOutcomeModel(outbreak, NegBinomialOutcomeDistribution())]:
        for t in outbreak.expanding_cutoffs():
            try:
                optimization_result = model.fit(t, n_calls=300, max_calls=300, verbose=False)

            except TrivialTargetError:
                print("Trivial target")
                pass

