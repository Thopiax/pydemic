import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional

from outbreak import Outbreak


def _build_regions(*dfs) -> List[str]:
    regions = set.intersection(*[set(df.columns) for df in dfs])

    return sorted(regions)


class Epidemic:
    def __init__(self, name: str, cases: pd.DataFrame, deaths: pd.DataFrame, recoveries: pd.DataFrame,
                 **dfs: pd.DataFrame):
        self.name: str = name

        self.regions = _build_regions(cases, deaths, recoveries)

        # define start and end of epidemic by cases
        assert type(cases.index) is pd.DatetimeIndex
        self.start_date, self.end_date = cases.index.min(), cases.index.max()
        self.first_case_dates = cases.cumsum(axis=0).eq(0).idxmin(axis=0)

        self._outbreaks: Dict[str, Outbreak] = {}

        for region in self.regions:
            first_case_date = self.first_case_dates[region]

            region_dfs = dict()
            for key, df in dfs.items():
                if region in df.columns:
                    region_dfs[key] = df.loc[first_case_date:, region]

            self._outbreaks[region] = Outbreak(
                region,
                cases=cases.loc[first_case_date:, region],
                deaths=deaths.loc[first_case_date:, region],
                recoveries=recoveries.loc[first_case_date:, region],
                **region_dfs
            )

    def __getitem__(self, region):
        if type(region) is list:
            return {reg: self._outbreaks.get(reg) for reg in region}.items()

        return self._outbreaks.get(region)

    def combine(self, name, *regions):
        assert len(regions) > 0
        assert all(region in self.regions for region in regions)

        df = self[regions[0]].df
        for region in regions[1:]:
            df = df.combine(other=self[region].df, func=np.add, fill_value=0)

        return Outbreak(name, df=df)

    def search(self, query: str) -> Dict[str, Outbreak]:
        return {region: self._outbreaks[region] for region in self.regions if query in region}

    def apply(self, func: Callable, *params, regions: Optional[List[str]] = None, **kwargs) -> Any:
        if regions is None:
            regions = self.regions

        result = pd.Series(index=regions)

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

