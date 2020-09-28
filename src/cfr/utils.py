from typing import Type

import pandas as pd

from cfr.estimates.base import BaseCFREstimate
from outbreak import Outbreak


def build_estimates(outbreak: Outbreak, *estimate_methods: Type[BaseCFREstimate], **kwargs):
    assert len(estimate_methods) >= 1

    cutoffs = outbreak.expanding_cutoffs(**kwargs)

    models = [mt(outbreak) for mt in estimate_methods]

    return pd.DataFrame({
        model.name: [model.estimate(t) for t in cutoffs] for model in models
    }, index=outbreak.df.index[cutoffs])
