import os
from pathlib import PosixPath
from typing import Optional, List, Tuple

import numpy as np
from scipy.optimize import OptimizeResult

from skopt import dump, load, expected_minimum

from utils.path import CACHE_ROOTPATH


def load_result(tag: str, path: PosixPath = CACHE_ROOTPATH, extension: str = "pkl"):
    filepath = path / f"{tag}.{extension}"

    if os.path.isfile(filepath):
        return load(filepath)

    return None


def save_result(tag: str, result, path: PosixPath = CACHE_ROOTPATH, extension: str = "pkl"):
    filepath = path / f"{tag}.{extension}"

    if os.path.exists(path) is False:
        os.makedirs(path)

    del result.specs['args']['func']
    dump(result, filepath, compress=True)


def get_optimal_parameters(result) -> Optional[List[float]]:
    return result.x


def get_optimal_loss(result) -> Optional[float]:
    return result.fun


def get_initial_points(result: OptimizeResult) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    return result.x_iters, result.func_vals


def get_expected_minimum(result: OptimizeResult, **kwargs) -> Optional[List[float]]:
    if len(result.models) == 0:
        return None

    return expected_minimum(result, **kwargs)