import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skopt.plots import plot_objective

import sys

from epydemic.outcome.distribution.discrete import NegBinomialOutcomeDistribution
from epydemic.outbreak import Outbreak

from outcome.models.exceptions import TrivialTargetError
from outcome.models.fatality import FatalityOutcomeModel
from outcome.models.recovery import RecoveryOutcomeModel

sys.path.append(Path(__file__).parent.parent.parent)

if __name__ == "__main__":
    outbreak = Outbreak.from_csv("Germany")

    for model in [FatalityOutcomeModel(outbreak, NegBinomialOutcomeDistribution()), RecoveryOutcomeModel(outbreak, NegBinomialOutcomeDistribution())]:
        for t in outbreak.expanding_cutoffs():
            try:
                optimization_result = model.fit(t, n_calls=300, max_calls=300, verbose=False)

            except TrivialTargetError:
                print("Trivial target")
                pass

