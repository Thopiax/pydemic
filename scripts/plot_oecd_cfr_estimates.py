import pandas as pd
import matplotlib.pyplot as plt

from lib.pydemic import Pandemic, OECD_COUNTRIES
from lib.pydemic.outbreak.cfr import NaiveCFREstimator, ResolvedCFREstimator, WeightedResolvedCFREstimator, CFREvaluator

coronavirus_confirmed_df = pd.read_csv("../data/clean/coronavirus_confirmed_global.csv", index_col=0,
                                       parse_dates=[0])
coronavirus_death_df = pd.read_csv("../data/clean/coronavirus_death_global.csv", index_col=0, parse_dates=[0])
coronavirus_recovered_df = pd.read_csv("../data/clean/coronavirus_recovered_global.csv", index_col=0,
                                       parse_dates=[0])
coronavirus_advanced_testing_policy_adopted = pd.read_csv(
    "../data/clean/coronavirus_advanced_testing_policy_adopted.csv", index_col=0, squeeze=True)
coronavirus_latest_testing_per_thousand = pd.read_csv("../data/clean/coronavirus_latest_testing_per_thousand.csv",
                                                      index_col=0, squeeze=True)

pandemic = Pandemic(
    "Coronavirus",
    coronavirus_confirmed_df,
    coronavirus_death_df,
    recoveries=coronavirus_recovered_df,
    advanced_testing_policy_adopted=coronavirus_advanced_testing_policy_adopted,
    latest_testing_per_thousand=coronavirus_latest_testing_per_thousand
)

OECD_outbreak = pandemic.multiregional_outbreak("OECD", OECD_COUNTRIES)
mea = CFREvaluator(OECD_outbreak, NaiveCFREstimator, ResolvedCFREstimator, WeightedResolvedCFREstimator)

ax = plt.gca()
ax.set_title(f"CFR Estimators - OECD")
mea.plot_estimates()
plt.show()

for region, outbreak in pandemic.get_outbreaks(OECD_COUNTRIES).items():
    mea = CFREvaluator(outbreak, NaiveCFREstimator, ResolvedCFREstimator, WeightedResolvedCFREstimator)

    ax = plt.gca()
    ax.set_title(f"CFR Estimators - {region}")
    mea.plot_estimates()
    plt.show()
