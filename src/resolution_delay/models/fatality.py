from typing import Optional

import numpy as np

from resolution_delay.models.single import SingleResolutionDelayModel
from resolution_delay.models.utils import verify_probability


class FatalityResolutionDelayModel(SingleResolutionDelayModel):
    name: str = "FOL"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._target = self.outbreak.cumulative_deaths.to_numpy(dtype="float32")

    def alpha(self, t: int, start: int = 0) -> float:
        cecr = self._calculate_cecr(t, start=start)

        return self._calculate_alpha(t, start=start, cecr=cecr)

