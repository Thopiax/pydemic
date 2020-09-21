from collections import namedtuple


InfectiousChainParameters = namedtuple("InfectiousChainParameters", ["K", "rate"], defaults=[1, 1 / 5.0])

OutcomeParameters = namedtuple("OutcomeParameters", ["gamma", "K", "lambdaS", "lambdaA"],
                               defaults=[1.0, 1, 1 / 5.0, 1 / 5.0])

InfectionParameters = namedtuple("InfectionParameters", ["c", "beta", "D_E"],
                                 defaults=[1.0, 0.5, 5])