import os
from typing import Union
from pathlib import PosixPath

from outcome_lag.optimizer.utils import load_result, save_result
from utils.path import CACHE_ROOTPATH


class OutcomeOptimizerCache:
    def __init__(self, path: Union[str, PosixPath]):
        self._path = CACHE_ROOTPATH / path

        if os.path.exists(self._path) is False:
            os.makedirs(self._path)

        self._cache = {}

    def __contains__(self, item):
        return item in self._cache

    def get(self, tag):
        result = self._cache.get(tag)

        if result is None:
            result = load_result(tag, path=self._path)

        return result

    def add(self, tag, result, to_disk: bool = True) -> None:
        self._cache[tag] = result

        if to_disk:
            save_result(tag, result, path=self._path)
