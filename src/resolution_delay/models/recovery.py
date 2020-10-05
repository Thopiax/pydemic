from resolution_delay.models.single import SingleResolutionDelayModel


class RecoveryResolutionDelayModel(SingleResolutionDelayModel):
    name: str = "RRD"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._target = self.outbreak.cumulative_recoveries.to_numpy(dtype="float32")

    def alpha(self, t: int, start: int = 0) -> float:
        cecr = self._calculate_cecr(t, start=start)

        return 1 - self._calculate_alpha(t, start=start, cecr=cecr)

