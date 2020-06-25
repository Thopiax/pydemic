import pandas as pd
from typing import Optional, Dict

from .outbreak import Outbreak


class Epidemic:
    def __init__(self, name: str, cases: pd.DataFrame, deaths: pd.DataFrame, recoveries: pd.DataFrame, **data):
        self.name: str = name

        self.cases: pd.DataFrame = cases
        self.deaths: pd.DataFrame = deaths
        self.recoveries: pd.DataFrame = recoveries

        self._secondary_data: Dict[str, pd.DataFrame] = data

        self._build_regions()
        self._build_temporal_index()
        self._build_outbreaks()

    def _build_regions(self):
        cases_regions = set(self.cases.columns)
        deaths_regions = set(self.deaths.columns)
        recovery_regions = set(self.recoveries.columns)

        assert cases_regions == deaths_regions and cases_regions == recovery_regions

        regions = cases_regions.intersection(deaths_regions).intersection(recovery_regions)

        self._regions = pd.Series(sorted(list(regions)))

    def _build_temporal_index(self):
        dates = pd.to_datetime(self.cases.index)
        self.start_date, self.end_date = min(dates), max(dates)
        self.date_range = pd.date_range(self.start_date, self.end_date, name='Dates', freq='D')

        def convert_to_timedelta(sequence):
            # remove daily gaps in types
            # (assumption: if data on a specific date is unreported, the latest values at that date are kept.)
            sequence = sequence.reindex(self.date_range).dropna(how="all").fillna(method="ffill")
            # set sequence indices to time delta
            sequence.index = pd.to_datetime(sequence.index) - self.start_date

            return sequence

        self.cases = convert_to_timedelta(self.cases)
        self.deaths = convert_to_timedelta(self.deaths)
        self.recoveries = convert_to_timedelta(self.recoveries)

        for key, data in self._secondary_data.items():
            self._secondary_data[key] = convert_to_timedelta(data)

    def _build_outbreaks(self):
        self.outbreaks = {}

        # truncate the sequences based on the first recorded case
        self.first_case_delays = self.cases.cumsum(0).eq(0).sum().apply(lambda x: pd.Timedelta(days=x))

        for region in self.regions:
            self.outbreaks[region] = self._build_regional_outbreak(region, self.first_case_delays[region])

        return True

    def _build_regional_outbreak(self, region, first_case_date, region_name=None):
        is_multiregion = type(region) is list

        if is_multiregion:
            # if the region is composed of multiple regions (i.e., a list), then a name must necessarily be passed in
            assert type(region_name) is str and region_name != ""
        else:
            region_name = region

        region_cases = self.cases.loc[first_case_date:, region]
        region_cases.index -= first_case_date

        region_deaths = self.deaths.loc[first_case_date:, region]
        region_deaths.index -= first_case_date

        region_recoveries = self.recoveries.loc[first_case_date:, region]
        region_recoveries.index -= first_case_date

        if is_multiregion:
            region_cases = region_cases.sum(axis=1)
            region_deaths = region_deaths.sum(axis=1)
            region_recoveries = region_recoveries.sum(axis=1)

        region_secondary_data = {}
        for key, data in self._secondary_data.items():
            if region in data.columns:
                region_secondary_data[key] = data.loc[first_case_date:, region]
                region_secondary_data[key].index -= first_case_date

                if is_multiregion:
                    region_secondary_data[key] = region_secondary_data[key].sum(axis=1)

        return Outbreak(
            region_name,
            region_cases,
            region_deaths,
            region_recoveries,
            **region_secondary_data
        )

    def multiregional_outbreak(self, name, regions):
        if name not in self.outbreaks:
            first_case_date = self.first_case_delays[regions].min()

            self.outbreaks[name] = self._build_regional_outbreak(regions, first_case_date, region_name=name)

        return self.outbreaks[name]

    def get_top_regions(self, top_n=10, exclude=None):
        if exclude is None:
            exclude = []

        sorted_regions = self.cases.sum(axis=0).sort_values(ascending=False).index.tolist()

        return [region for region in sorted_regions if region not in exclude][:top_n]

    def get_outbreaks(self, regions=None):
        if regions is None:
            regions = self.regions

        return {region: outbreak for (region, outbreak) in self.outbreaks.items() if region in regions}

    def search_region(self, query):
        return self.regions[self.regions.str.contains(query)]

    @property
    def regions(self):
        return self._regions

    def apply(self, func, *params, regions=None, **kwargs):
        if regions is None:
            regions = self.regions

        result = pd.Series(index=regions)

        for region, outbreak in self.get_outbreaks(regions).items():
            result[region] = func(outbreak, *params, **kwargs)

        return result

    def print_regions_coverage(self, regions):
        regions_total_number_of_cases = self.cases[regions].cumsum(axis=0).sum(axis=1)[-1]
        global_total_number_of_cases = self.cases.cumsum(axis=0).sum(axis=1)[-1]

        regions_total_number_of_deaths = self.deaths[regions].cumsum(axis=0).sum(axis=1)[-1]
        global_total_number_of_deaths = self.deaths.cumsum(axis=0).sum(axis=1)[-1]

        regions_total_number_of_healed = self.recoveries[regions].cumsum(axis=0).sum(axis=1)[-1]
        global_total_number_of_healed = self.recoveries.cumsum(axis=0).sum(axis=1)[-1]

        print(f"Regions: {', '.join(regions)}")
        print("Case coverage=%.2f" % ((regions_total_number_of_cases / global_total_number_of_cases) * 100))
        print("Death coverage=%.2f" % ((regions_total_number_of_deaths / global_total_number_of_deaths) * 100))
        print("Recovery coverage=%.2f" % ((regions_total_number_of_healed / global_total_number_of_healed) * 100))
