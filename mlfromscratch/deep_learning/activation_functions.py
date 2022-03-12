import numpy as np


class Sigmoid():
    def __call__(self, x):
        return 1. / (1. + np.exp(-x))

    def gradient(self, x):
        # \sigma(x)(1 - \sigma(x))
        return self.__call__(x) * (1. - self.__call__(x))