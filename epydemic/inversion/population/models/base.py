from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from epydemic.inversion.individual import BaseIndividualModel, FatalityIndividualModel
from epydemic.outbreak import OutbreakTimeWindow


class AbstractPopulationModel(ABC):
    def __init__(self, otw: OutbreakTimeWindow, individual_model: Optional[BaseIndividualModel] = None,
                 verbose: bool = True, random_state: int = 1):

        self.otw = otw
        self.individual_model = FatalityIndividualModel() if individual_model is None else individual_model

        self.random_state = random_state
        self.verbose = verbose

        self.learner = None
        self.best_parameters = (None, None, None)
        self.best_loss = None

    @property
    def tag(self):
        tag = f"{self.otw.region}__{self.otw.start}_{self.otw.end}__{self.individual_model.tag}"

        if self.learner is None:
            return tag

        return f"{tag}__{self.learner.tag}"

    @property
    def parameters(self):
        return self.individual_model.parameters

    @parameters.setter
    def parameters(self, parameters):
        self.individual_model.parameters = parameters
        self._update_model()

    @abstractmethod
    def _update_model(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass
