import numpy as np


class Cost(object):
    def cost(self, computed, actual):
        raise NotImplementedError

    def gradients(self, computed, actual):
        raise NotImplementedError


class MSE(Cost):
    def cost(self, computed, actual):
        return np.mean((computed - actual) ** 2)

    def gradients(self, computed, actual):
        return 2 * (computed - actual)
