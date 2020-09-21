import numpy as np
from src.outbreak import Outbreak

def observe_synthetic_simulation(simulation, region: str = "synthetic", dt: float = 1.0) -> Outbreak:
    observation_index = np.arange(0, simulation.index[-1], dt)

    return Outbreak(
        region,
        cases=simulation.loc[observation_index, "I"].astype(int),
        deaths=simulation.loc[observation_index, "D"].astype(int),
        recoveries=simulation.loc[observation_index, "R"].astype(int),
    )

