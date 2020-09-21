from typing import Collection

import pandas as pd


def average_simulations(simulations: Collection[pd.DataFrame]):
    assert len(simulations) > 1
    result = simulations[0]

    for sim in simulations[1:]:
        result.add(sim, axis=1)

    result.div(float(len(simulations)))

    return result