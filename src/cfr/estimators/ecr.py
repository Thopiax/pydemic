from functools import lru_cache

from outbreak import Outbreak
from resolution_delay.models.fatality import FatalityResolutionDelayModel
from resolution_delay.models.recovery import RecoveryResolutionDelayModel
from src.cfr.estimators.base import BaseCFREstimator


class ECRFatalityCFREstimator(BaseCFREstimator):
    name: str = "ECR_fatality"

    def __init__(self, outbreak: Outbreak, **kwargs):
        super().__init__(outbreak)

        self._model = FatalityResolutionDelayModel(outbreak, **kwargs)

    @lru_cache(maxsize=32)
    def estimate(self, t: int, start: int = 0, **kwargs) -> float:
        self._verify_inputs(t, start)

        print(f"[t={t}, start={start}] Fitting model...")
        self._model.fit(t, start=start, verbose=True, **kwargs)
        print(f"[t={t}, start={start}] Finished fitting.")

        return self._model.alpha(t, start=start)


class ECRRecoveryCFREstimator(BaseCFREstimator):
    name: str = "ECR_recovery"

    def __init__(self, outbreak: Outbreak, **kwargs):
        super().__init__(outbreak)

        self._model = RecoveryResolutionDelayModel(outbreak, **kwargs)

    @lru_cache(maxsize=32)
    def estimate(self, t: int, start: int = 0, **kwargs) -> float:
        self._verify_inputs(t, start)

        print(f"[t={t}, start={start}] Fitting model...")
        self._model.fit(t, start=start, verbose=True, **kwargs)
        print(f"[t={t}, start={start}] Finished fitting.")

        return self._model.alpha(t, start=start)


class ECRHybridCFREstimator(BaseCFREstimator):
    name: str = "ECR_hybrid"

    def __init__(self, outbreak: Outbreak, **kwargs):
        super().__init__(outbreak)

        self._fatality_estimate = ECRFatalityCFREstimator(outbreak, **kwargs)
        self._recovery_estimate = ECRRecoveryCFREstimator(outbreak, **kwargs)
        self._model = RecoveryResolutionDelayModel(outbreak, **kwargs)

    @lru_cache(maxsize=32)
    def estimate(self, t: int, start: int = 0, **kwargs) -> float:
        self._verify_inputs(t, start)
        fatality_alpha = self._fatality_estimate.estimate(t, start=start, **kwargs)
        recovery_alpha = self._recovery_estimate.estimate(t, start=start, **kwargs)

        return (fatality_alpha + recovery_alpha) / 2

        print(f"[t={t}, start={start}] Fitting model...")
        self._model.fit(t, start=start, verbose=True, **kwargs)
        print(f"[t={t}, start={start}] Finished fitting.")

        return self._model.alpha(t, start=start)

