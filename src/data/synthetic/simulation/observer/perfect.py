from typing import Dict

from .base import SimulationObserver


class PerfectObserver(SimulationObserver):
    columns = ["cases", "deaths", "recoveries"]

    @staticmethod
    def _observe_cases(state_diff: Dict[str, float]):
        return max(- (state_diff["E"] + state_diff["S"]), 0)

    def _observe_state(self, state_diff: Dict[str, float]):
        cases = PerfectObserver._observe_cases(state_diff)

        self._state["cases"] += cases
        self._state["deaths"] += state_diff["D"]
        self._state["recoveries"] += state_diff["R"]