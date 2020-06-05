from .base import AbstractCFREstimator


class NaiveCFREstimator(AbstractCFREstimator):
    def __repr__(self):
        return "nCFR"

    def estimate(self, t : int = -1) -> float:
        super().estimate(t)

        return self.outbreak.cumulative_deaths.iloc[t] / self.outbreak.cumulative_cases.iloc[t]