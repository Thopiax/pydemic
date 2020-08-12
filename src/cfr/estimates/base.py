from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
import pandas as pd

from src.outbreak import Outbreak


class BaseCFREstimate(ABC):
    name: str = "base"

    def __init__(self, outbreak: Outbreak):
        self.outbreak = outbreak

    def _verify_inputs(self, t: int, start: int):
        assert 0 <= t < len(self.outbreak)
        assert 0 <= start < t

    @abstractmethod
    def estimate(self, t: int, start: int = 0) -> float:
        pass
