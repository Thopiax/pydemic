from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from epydemic.outbreak import Outbreak


class BaseCFRModel(ABC):
    name: str = "base"

    def __init__(self, outbreak: Outbreak):
        self.outbreak = outbreak

    @abstractmethod
    def estimate(self, t: int) -> float:
        # if period of analysis is in the future, return NaN.
        if t > len(self.outbreak):
            return np.nan()

        pass
