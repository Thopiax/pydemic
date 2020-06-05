import functools
import os
from typing import Union

import matplotlib.pyplot as plt

from config import setup_matplotlib
from utils.path import PLOTS_ROOTPATH

setup_matplotlib()

def save_figure(dest: Union[str, callable], overwrite: bool = True):
    def decorator_save_figure(func: callable):
        @functools.wraps(func)
        def wrapper_func(*args, save_figure: bool = True, **kwargs):
            fig = plt.gcf()
            func(*args, **kwargs)

            if save_figure is False:
                return

            path = dest
            if callable(dest):
                path = dest(*args, **kwargs)

            path = PLOTS_ROOTPATH / path
            path_dir = os.path.dirname(path)

            if os.path.isdir(path_dir) is False:
                os.mkdir(path_dir)

            if os.path.isfile(path) and overwrite is False:
                return

                fig.savefig(path, optimize=True, format="pdf")

        return wrapper_func

    return decorator_save_figure
