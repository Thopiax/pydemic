from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np

from epydemic.inversion.individual import BaseIndividualModel, FatalityIndividualModel
from epydemic.outbreak import OutbreakSlice


class BasePopulationModel(ABC):
    IndividualModel = BaseIndividualModel

    def __init__(self, otw: OutbreakSlice, individual_model: Optional[BaseIndividualModel] = None,
                 verbose: bool = True, random_state: int = 1, **kwargs):

        self.otw = otw
        self.individual_model: BaseIndividualModel = self.IndividualModel(**kwargs) if individual_model is None else individual_model

        self.random_state: int = random_state
        self.verbose: bool = verbose

        self.learner = None
        self.best_parameters = (None, None, None)
        self.best_loss = None

    @property
    def tag(self):
        return f"{self.otw.region}__{self.otw.start}_{self.otw.end}__{self.individual_model.tag}"

    @property
    def parameters(self):
        return self.individual_model.parameters

    @parameters.setter
    def parameters(self, parameters):
        self.individual_model.parameters = parameters
        self._update_model()

    def _build_expectation_matrix(self, rate):
        assert self.otw is not None

        T = len(self.otw)
        K = len(rate)

        # set cases to be the entire outbreak cases (consider pre burn-in data)
        cases = self.otw.outbreak.cases

        result = np.zeros((T, K))

        for t in range(T):
            otw_t = self.otw.start + t

            for k in range(min(otw_t + 1, K)):
                result[t, k] = rate[k] * cases[otw_t - k]

        return result

    @abstractmethod
    def _update_model(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def target(self):
        pass

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass
