import numpy as np


class Sigmoid():

    def __init__(self, a=1):
        self.a = a

    def evaluate(self, x):
        #return 1 / (1 + np.exp(-self.a * x))
        # questa è sostanzialmente analoga ma tiene in conto i problemi di overflow
        return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))

    def derivative(self, x):
        x = self.evaluate(x)
        return self.a * x * (1 - x)
