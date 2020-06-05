import os
from pathlib import Path

# ROOTPATH = "./../../"
ROOTPATH = Path(__file__).parent.parent.parent

PLOTS_ROOTPATH = ROOTPATH / "plots"
if os.path.exists(PLOTS_ROOTPATH) is False:
    os.mkdir(PLOTS_ROOTPATH)

DATA_ROOTPATH = ROOTPATH / "data"
if os.path.exists(DATA_ROOTPATH) is False:
    os.mkdir(DATA_ROOTPATH)

CACHE_ROOTPATH = ROOTPATH / ".cache"
if os.path.exists(CACHE_ROOTPATH) is False:
    os.mkdir(CACHE_ROOTPATH)
