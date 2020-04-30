import pandas as pd
import numpy as np
from .outbreak import Outbreak 

class Pandemic:
    def __init__(self, name, epidemic_curve, fatality_curve, recovery_curve):
        self.name = name
        
        # the columns in the two timelines should mat
        assert (epidemic_curve.columns == fatality_curve.columns).all() and (epidemic_curve.columns == recovery_curve.columns).all()
        
        self.epidemic_curve = epidemic_curve
        self.fatality_curve = fatality_curve
        self.recovery_curve = recovery_curve
        
        self.convert_indices_to_timedelta_since_epidemic_start_date()
        
        self.outbreaks = {}
        self.build_outbreaks()
        
    def get_top_regions(self, top_n=10, exclude=[]):
        sorted_regions = self.epidemic_curve.sum(axis=0).sort_values(ascending=False).index.tolist()
        
        return [region for region in sorted_regions if region not in exclude][:top_n]
    
    def get_outbreaks(self, regions):
        return {region: outbreak for (region, outbreak) in self.outbreaks.items() if region in regions}
    
    def search_region(self, query):
        return self.regions[self.regions.str.contains(query)]
    
    @property
    def regions(self):
        return self.epidemic_curve.columns
    
    def apply(self, func, *params, regions=None, **kwargs):
        if regions is None:
            regions = self.regions

        result = pd.Series(index=regions)

        for region, outbreak in self.get_outbreaks(regions).items():
            result[region] = func(outbreak, *params, **kwargs)
        
        return result

    def set_smoothing_coefficient(self, k):
        for outbreak in self.outbreaks.values():
            outbreak.smoothing_coefficient = k
    
    def build_outbreaks(self):
        first_confirmed_case_dates = self.epidemic_curve.cumsum(0).eq(0).sum().apply(lambda x: pd.Timedelta(days=x))
    
        for region in self.regions:
            region_index = self.epidemic_curve.index
            
            first_confirmed_case_in_region = first_confirmed_case_dates[region]
            
            region_epidemic_curve = self.epidemic_curve.loc[first_confirmed_case_in_region:, region]
            region_epidemic_curve.index -= first_confirmed_case_in_region
            
            region_fatality_curve = self.fatality_curve.loc[first_confirmed_case_in_region:, region]
            region_fatality_curve.index -= first_confirmed_case_in_region
            
            region_recovery_curve = self.recovery_curve.loc[first_confirmed_case_in_region:, region]
            region_recovery_curve.index -= first_confirmed_case_in_region
            
            self.outbreaks[region] = Outbreak(region, region_epidemic_curve, region_fatality_curve, region_recovery_curve)
    
        return True
    
    def print_regions_coverage(self, regions):
        regions_total_number_of_cases = self.epidemic_curve[regions].sum(axis=1)[-1]
        global_total_number_of_cases = self.epidemic_curve.sum(axis=1)[-1]

        regions_total_number_of_deaths = self.fatality_curve[regions].sum(axis=1)[-1]
        global_total_number_of_deaths = self.fatality_curve.sum(axis=1)[-1]
        
        regions_total_number_of_healed = self.recovery_curve[regions].sum(axis=1)[-1]
        global_total_number_of_healed = self.recovery_curve.sum(axis=1)[-1]
        
        print(f"Regions: {', '.join(regions)}")
        print("Case coverage=%.2f" % ((regions_total_number_of_cases / global_total_number_of_cases) * 100))
        print("Death coverage=%.2f" % ((regions_total_number_of_deaths / global_total_number_of_deaths) * 100))
        print("Recovery coverage=%.2f" % ((regions_total_number_of_healed / global_total_number_of_healed) * 100))

    def convert_indices_to_timedelta_since_epidemic_start_date(self):
        assert self.epidemic_curve is not None and self.fatality_curve is not None
        
        self.epidemic_start_date = min(pd.to_datetime(self.epidemic_curve.index))
        self.epidemic_end_date = max(pd.to_datetime(self.epidemic_curve.index))
        
        index_range = pd.date_range(self.epidemic_start_date, self.epidemic_end_date, name='Dates')

        # reindex curves to ensure there are no gaps
        self.epidemic_curve = self.epidemic_curve.reindex(index_range, method='bfill')
        self.fatality_curve = self.fatality_curve.reindex(index_range, method='bfill')
        self.recovery_curve = self.recovery_curve.reindex(index_range, method='bfill')
        
        self.epidemic_curve.index = pd.to_datetime(self.epidemic_curve.index) - self.epidemic_start_date
        self.fatality_curve.index = pd.to_datetime(self.fatality_curve.index) - self.epidemic_start_date
        self.recovery_curve.index = pd.to_datetime(self.recovery_curve.index) - self.epidemic_start_date
