

class OutcomeOptimizerLoss(object):
    def __init__(self, model, t: int):
        self.model = model
        self.t = t

    def __call__(self, parameters):
        try:
            self.model.parameters = parameters

            return self.model.loss(self.t)

        except Exception:
            # return a large penalty for parameters that lead to invalid fatality rate
            return 1_000
