import os
from pathlib import PosixPath
from typing import Optional, List, Tuple, Union

from heapq import nsmallest
from scipy.optimize import OptimizeResult

from skopt import dump, load, expected_minimum

from utils.path import CACHE_ROOTPATH


def load_result(tag: str, path: PosixPath = CACHE_ROOTPATH, extension: str = "pkl"):
    filepath = path / f"{tag}.{extension}"

    if os.path.isfile(filepath):
        return load(filepath)

    return None


def save_result(tag: str, result: OptimizeResult, path: PosixPath = CACHE_ROOTPATH, extension: str = "pkl"):
    filepath = path / f"{tag}.{extension}"

    if os.path.exists(path) is False:
        os.makedirs(path)

    del result.specs['args']['func']
    dump(result, filepath, compress=True)


def get_optimal_parameters(result: Union[OptimizeResult, None]) -> List:
    return result.x if result is not None else []


def get_optimal_loss(result: Union[OptimizeResult, None]) -> float:
    return result.fun if result is not None else -1.0


def get_n_best_parameters(n, result: Union[OptimizeResult, None]) -> List[Tuple[float, List[float]]]:
    if result is None:
        return []

    return nsmallest(n, zip(result.func_vals, result.x_iters))


def get_initial_points(result: Union[OptimizeResult, None]) -> Union[Tuple[List, List], Tuple[None, None]]:
    if result is None:
        return None, None

    return list(result.x_iters), list(result.func_vals)


def get_expected_minimum(result: OptimizeResult, **kwargs) -> Optional[List[float]]:
    if len(result.models) == 0:
        return None

    return expected_minimum(result, **kwargs)

