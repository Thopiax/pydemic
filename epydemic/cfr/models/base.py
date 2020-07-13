from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
import pandas as pd

from epydemic.outbreak import Outbreak


class BaseCFRModel(ABC):
    name: str = "base"

    def __init__(self, outbreak: Outbreak, corrected: bool = False):
        self.outbreak = outbreak

        self.corrected = corrected

    @property
    def name(self):
        if self.corrected:
            return self.__class__.name + "_corrected"

        return self.__class__.name

    @abstractmethod
    def estimate(self, t: int, start: int = 0) -> float:
        pass
