from typing import Type

import pandas as pd

from cfr.estimates.base import BaseCFREstimate
from cfr.estimates import *
from outbreak import Outbreak

ALL_ESTIMATES = [
    NaiveCFREstimate, NaiveComplementCFREstimate,
    ResolvedCFREstimate, ECRFatalityCFREstimate, ECRRecoveryCFREstimate, ECRHybridCFREstimate
]

ALL_ESTIMATES_MAP = {est.name: est for est in ALL_ESTIMATES}


def build_estimates(outbreak: Outbreak, *estimators: Type[BaseCFREstimate], **kwargs):
    assert len(estimators) >= 1

    estimates = [est(outbreak) for est in estimators]

    cutoffs = outbreak.expanding_cutoffs(**kwargs)

    return pd.DataFrame({
        estimate.__class__.name: [estimate.estimate(t) for t in cutoffs] for estimate in estimates
    }, index=outbreak.df.index[cutoffs])
