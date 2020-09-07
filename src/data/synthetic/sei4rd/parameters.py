from collections import namedtuple


# TODO: make Infectious Chains more generic
# InfectiousChainParameters = namedtuple("InfectiousChainParameters", ["p", "c", "beta", "K", "lam"],
#                                        defaults=[1.0, 1.0, 1, 0.2])

OutcomeParameters = namedtuple("OutcomeParameters", ["gamma", "K", "lambdaS", "lambdaA"],
                               defaults=[1.0, 1, 1 / 5.0, 1 / 5.0])

InfectionParameters = namedtuple("InfectionParameters", ["c", "beta", "D_E"],
                                 defaults=[1.0, 0.5, 5])