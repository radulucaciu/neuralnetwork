import numpy as np

from neuralnetwork.layers.activation import ActivationLayer


class TanhLayer(ActivationLayer):
    def activation(self, x):
        return np.tanh(x)

    def activation_prime(self, x):
        activation = self.activation(x)
        return 1 - activation ** 2
