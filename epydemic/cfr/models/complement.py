from .base import BaseCFRModel


class ComplementCFRModel(BaseCFRModel):
    name: str = "complement"

    def estimate(self, t: int) -> float:
        super().estimate(t)

        return 1 - (self.outbreak.cumulative_recoveries.iloc[t] / self.outbreak.cumulative_cases.iloc[t])