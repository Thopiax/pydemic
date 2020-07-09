from typing import Type

import pandas as pd

from cfr.models.base import BaseCFRModel
from outbreak import Outbreak


def build_estimates(outbreak: Outbreak, *model_types: Type[BaseCFRModel], **kwargs):
    cutoffs = outbreak.expanding_cutoffs(**kwargs)

    models = [mt(outbreak) for mt in model_types]

    return pd.DataFrame({
        model.name: [model.estimate(t) for t in cutoffs] for model in models
    }, index=outbreak.df.index[cutoffs])
