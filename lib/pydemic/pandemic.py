import pandas as pd
import numpy as np
from .outbreak import Outbreak

OECD_COUNTRIES = [
    "Australia", "Austria", "Belgium", "Canada", "Chile", "Colombia", "Czechia", "Denmark", "Estonia", "Finland",
    "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Israel", "Italy", "Japan", "Latvia", "Lithuania",
    "Luxembourg", "Mexico", "Netherlands", "New Zealand", "Norway", "Poland", "Sweden", "Switzerland", "Turkey",
    "United Kingdom", "US"
]

class Pandemic:
    def __init__(self, name, cases : pd.Series, deaths : pd.Series, **data):
        self.name = name

        assert cases is not None and deaths is not None

        self.cases = cases
        self.deaths = deaths
        self.recoveries = data.get("recoveries")

        self.advanced_testing_policy_adopted = data.get("advanced_testing_policy_adopted")
        self.latest_testing_per_thousand = data.get("latest_testing_per_thousand")

        self._normalize_sequence_regions()
        self._normalize_sequence_indices()
        
        self._build_outbreaks()

    def _normalize_sequence_regions(self):
        cases_regions = set(self.cases.columns)
        deaths_regions = set(self.deaths.columns)

        assert cases_regions == deaths_regions

        regions = cases_regions.intersection(deaths_regions)

        if self.recoveries is not None:
            recovery_regions = set(self.recoveries.columns)

            regions = regions.intersection(recovery_regions)

        self._regions = sorted(list(regions))

    def _normalize_sequence_indices(self):
        dates = pd.to_datetime(self.cases.index)
        self.start_date, self.end_date = min(dates), max(dates)
        self.date_range = pd.date_range(self.start_date, self.end_date, name='Dates', freq='D')

        def convert_to_timedelta(sequence):
            # remove daily gaps in data (assumption: if a date is unreported, the latest values are kept)
            sequence = sequence.reindex(self.date_range, method="bfill")
            # set sequence indices to time delta
            sequence.index = pd.to_datetime(sequence.index) - self.start_date

            return sequence

        self.cases = convert_to_timedelta(self.cases)
        self.deaths = convert_to_timedelta(self.deaths)

        if self.recoveries is not None:
            self.recoveries = convert_to_timedelta(self.recoveries)

    def _build_outbreaks(self):
        self.outbreaks = {}

        # truncate the sequences based on the first recorded case
        self.first_case_dates = self.cases.cumsum(0).eq(0).sum().apply(lambda x: pd.Timedelta(days=x))

        for region in self.regions:
            self.outbreaks[region] = self._build_regional_outbreak(region, self.first_case_dates[region])

        return True

    def multiregional_outbreak(self, name, regions):
        if name not in self.outbreaks:
            first_case_date = self.first_case_dates[regions].min()

            self.outbreaks[name] = self._build_regional_outbreak(regions, first_case_date, region_name=name)

        return self.outbreaks[name]

    def _build_regional_outbreak(self, region, first_case_date, region_name = None):

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

        if is_multiregion:
            region_cases = region_cases.sum(axis=1)
            region_deaths = region_deaths.sum(axis=1)

        region_recoveries = None
        if self.recoveries is not None:
            region_recoveries = self.recoveries.loc[first_case_date:, region]
            region_recoveries.index -= first_case_date

            if is_multiregion:
                region_recoveries = region_recoveries.sum(axis=1)

        region_advanced_testing_policy_adopted = None
        if self.advanced_testing_policy_adopted is not None:
            region_advanced_testing_policy_adopted = self.advanced_testing_policy_adopted.get(region, False)

        region_latest_testing_per_thousand = None
        if self.latest_testing_per_thousand is not None:
            region_latest_testing_per_thousand = self.latest_testing_per_thousand.get(region)

        return Outbreak(
            region_name,
            region_cases,
            region_deaths,
            recoveries=region_recoveries,
            advanced_testing_policy_adopted=region_advanced_testing_policy_adopted,
            latest_testing_per_thousand=region_latest_testing_per_thousand
        )

    def get_top_regions(self, top_n=10, exclude=[]):
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


