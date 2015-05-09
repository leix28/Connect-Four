import numpy

import theano
import theano.tensor as T


from LogisticRegression import LogisticRegression
from HiddenLayer import HiddenLayer

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden#,
            #activation=T.tanh
        )

        self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_hidden#,
            #activation=T.tanh
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer2.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.hiddenLayer2.params + self.logRegressionLayer.params
