from .base import BaseCFRModel


class NaiveCFRModel(BaseCFRModel):
    name: str = "naive"

    def estimate(self, t: int) -> float:
        super().estimate(t)

        return self.outbreak.cumulative_deaths.iloc[t] / self.outbreak.cumulative_cases.iloc[t]