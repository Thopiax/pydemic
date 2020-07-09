import pandas as pd

from epydemic.utils.path import DATA_ROOTPATH

POLICY_INDICES = [
    'StringencyIndex',
    'GovernmentResponseIndex',
    'ContainmentHealthIndex',
    'EconomicSupportIndex'
]

POLICY_MEASURES = [
    'C1_School_closing',
    'C1_Flag',
    'C2_Workplace_closing',
    'C2_Flag',
    'C3_Cancel_public_events',
    'C3_Flag',
    'C4_Restrictions_on_gatherings',
    'C4_Flag',
    'C5_Close_public_transport',
    'C5_Flag',
    'C6_Stay_at_home_requirements',
    'C6_Flag',
    'C7_Restrictions_on_internal_movement',
    'C7_Flag',
    'C8_International_travel_controls',
    'E1_Income_support',
    'E1_Flag',
    'E2_Debt_contract_relief',
    'E3_Fiscal_measures',
    'E4_International_support',
    'H1_Public_information_campaigns',
    'H1_Flag',
    'H2_Testing_policy',
    'H3_Contact_tracing',
    'H4_Emergency_investment_in_healthcare',
    'H5_Investment_in_vaccines',
    'M1_Wildcard'
]


def load_oxford_policy_measures():
    return _load_oxford(POLICY_MEASURES, "measure")


def load_oxford_policy_indices():
    return _load_oxford(POLICY_INDICES, "index")


def _load_oxford(policies, suffix):
    result = {}

    for policy in policies:
        result[policy] = pd.read_csv(DATA_ROOTPATH / f"clean/coronavirus_oxford_{policy}_{suffix}.csv", index_col=0, parse_dates=[0])

    return result
