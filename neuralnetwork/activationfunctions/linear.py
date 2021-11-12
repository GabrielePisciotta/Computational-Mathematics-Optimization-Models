import numpy as np


class Linear():

    def evaluate(self, x):
        return x

    def derivative(self, x):
        return np.ones(x.shape)
