import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from abc import ABC, abstractmethod
from typing import Optional
from sklearn.metrics import mean_absolute_error


def build_scaling_coefficient(y_true):
    if y_true.shape[0] == 1:
        return y_true[0]
    else:
        return np.mean(np.abs(np.diff(y_true)))


def verify_equal_length(y_true, y_pred, sample_weight=None):
    assert y_true.shape[0] == y_pred.shape[0]

    if sample_weight is not None:
        assert y_true.shape[0] == sample_weight.shape[0]


class LearnerLoss(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        pass


class MAELearnerLoss(LearnerLoss):
    def __repr__(self):
        if self.name is not None:
            return self.name

        return "MAE"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        verify_equal_length(y_true, y_pred)

        return mean_absolute_error(y_true, y_pred)


class MASELearnerLoss(LearnerLoss):
    def __repr__(self):
        if self.name is not None:
            return self.name

        return "MASE"

    def __call__(self, y_true, y_pred, sample_weight: Optional[np.ndarray] = None):
        verify_equal_length(y_true, y_pred, sample_weight=sample_weight)

        scaling_coefficient = build_scaling_coefficient(y_true)

        if sample_weight is not None:
            if np.max(sample_weight) == 0:
                # no sample weights if they are all zero
                sample_weight = None

        return mean_absolute_error(y_true, y_pred, sample_weight=sample_weight) / scaling_coefficient
