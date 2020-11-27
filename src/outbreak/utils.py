import pandas as pd

from outbreak.main import Outbreak


def extract_first_nonzero_index(df: pd.DataFrame):
    return df.cumsum(axis=0).eq(0).idxmin(axis=0)


# take the regions that cases, deaths and recoveries have in common, sorted alphabetically
def extract_columns_in_common(*dfs):
    return sorted(set.intersection(*[set(df.columns) for df in dfs]))


