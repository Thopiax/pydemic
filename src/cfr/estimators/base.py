from abc import ABC, abstractmethod


class BaseCFREstimator(ABC):
    name: str = "base"

    def __init__(self, outbreak):
        self.outbreak = outbreak

    def _verify_inputs(self, t: int, start: int):
        assert 0 <= t < len(self.outbreak)
        assert 0 <= start < t

    @abstractmethod
    def estimate(self, t: int, start: int = 0) -> float:
        pass
