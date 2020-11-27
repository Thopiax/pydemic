from cfr.estimators import *

ALL_ESTIMATES = [
    NaiveCFREstimator, NaiveComplementCFREstimator,
    ResolvedCFREstimator, ECRFatalityCFREstimator, ECRRecoveryCFREstimator, ECRHybridCFREstimator
]

ALL_ESTIMATES_MAP = {est.name: est for est in ALL_ESTIMATES}

