import os
from functools import cached_property

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional

from outbreak.main import Outbreak
from outbreak.utils import extract_first_nonzero_index, extract_columns_in_common
from utils.path import DATA_ROOTPATH



class Epidemic:
    def __init__(self, name: str):
        self.name: str = name

        self.outbreaks: Dict[str, Outbreak] = {}

    def __getitem__(self, region):
        if type(region) is list:
            return {reg: self.outbreaks.get(reg) for reg in region}.items()

        return self.outbreaks.get(region)

    @cached_property
    def regions(self):
        assert len(self.outbreaks) > 0

        return sorted(self.outbreaks.keys())

    def populate_outbreaks_from_dataframes(self, cases: pd.DataFrame, deaths: pd.DataFrame,
                                           recoveries: pd.DataFrame, **dfs: pd.DataFrame):
        # list of the dates where the first case is reported per region
        first_case_dates = None
        if type(cases.index) is pd.DatetimeIndex:
            first_case_dates = extract_first_nonzero_index(cases)

        regions = extract_columns_in_common(cases, deaths, recoveries)

        for region in regions:
            first_case_date = 0
            if first_case_dates is not None:
                first_case_date = first_case_dates[region]

            region_dfs = dict()
            for key, df in dfs.items():
                if region in df.columns:
                    region_dfs[key] = df.loc[first_case_date:, region]

            self.outbreaks[region] = Outbreak(
                region,
                cases=cases.loc[first_case_date:, region],
                deaths=deaths.loc[first_case_date:, region],
                recoveries=recoveries.loc[first_case_date:, region],
                epidemic=self,
                **region_dfs
            )

    @staticmethod
    def from_dataframes(name: str, cases: pd.DataFrame, deaths: pd.DataFrame, recoveries: pd.DataFrame, **dfs: pd.DataFrame):
        epidemic = Epidemic(name)

        epidemic.populate_outbreaks_from_dataframes(cases, deaths, recoveries, **dfs)

        return epidemic

    @staticmethod
    def from_observations(name: str, observations: pd.DataFrame):
        cases = observations.xs("cases", level=1, axis=1)
        deaths = observations.xs("deaths", level=1, axis=1)
        recoveries = observations.xs("recoveries", level=1, axis=1)
        # TODO: implement other_dfs

        return Epidemic.from_dataframes(name, cases, deaths, recoveries)

    @staticmethod
    def from_csv(name: str, **kwargs):
        epidemic = Epidemic(name)

        dirpath = DATA_ROOTPATH / f"{name}"

        if os.path.isdir(dirpath):
            for filename in os.listdir(dirpath):
                assert filename.endswith(".csv")

                region = filename.split(".")[0]

                epidemic.outbreaks[region] = Outbreak.from_csv(region, name, **kwargs)

        raise Exception(f"Directory for epidemic `{name}` not found.")

    def to_csv(self):
        self.apply(lambda outbreak: outbreak.to_csv())

    def combine(self, name, *regions):
        assert len(regions) > 0
        assert all(region in self.regions for region in regions)

        df = self[regions[0]].df
        for region in regions[1:]:
            df = df.combine(other=self[region].df, func=np.add, fill_value=0)

        return Outbreak(name, df=df)

    def search(self, query: str) -> Dict[str, Outbreak]:
        return {region: self.outbreaks[region] for region in self.regions if query in region}

    def apply(self, func: Callable, *params, regions: Optional[List[str]] = None, **kwargs) -> Any:
        if regions is None:
            regions = self.regions

        result = {region: None for region in regions}

        for region, outbreak in self[regions]:
            result[region] = func(outbreak, *params, **kwargs)

        return result

    def print_regions_coverage(self, *regions: str, columns: Optional[List[str]] = None):
        if columns is None:
            columns = Outbreak.required_fields

        print(f"Regions: {', '.join(regions)}")

        for col in columns:
            regions_total = sum([self[region].df[col].sum(axis=0) for region in regions if region in self.regions])
            global_total = sum([self[region].df[col].sum(axis=0) for region in self.regions])

            print(f"{col} coverage={(regions_total / global_total) * 100:.2f}")

