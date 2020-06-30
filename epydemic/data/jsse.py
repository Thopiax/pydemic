import os

import pandas as pd

from data import load_oxford_policy_measures, load_oxford_policy_indices
from epydemic.outbreak.epidemic import Epidemic
from epydemic.utils.path import DATA_ROOTPATH


def load_coronavirus_epidemic(**data):
    coronavirus_confirmed_df = pd.read_csv(DATA_ROOTPATH / "clean/coronavirus_confirmed_global.csv",
                                           index_col=0, parse_dates=[0])
    coronavirus_death_df = pd.read_csv(DATA_ROOTPATH / "clean/coronavirus_death_global.csv", index_col=0,
                                       parse_dates=[0])
    coronavirus_recovered_df = pd.read_csv(DATA_ROOTPATH / "clean/coronavirus_recovered_global.csv",
                                           index_col=0, parse_dates=[0])

    return Epidemic(
        "Coronavirus",
        coronavirus_confirmed_df,
        coronavirus_death_df,
        coronavirus_recovered_df,
        **load_oxford_policy_measures(),
        **load_oxford_policy_indices(),
        **data
    )
