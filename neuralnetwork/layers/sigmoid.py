import numpy as np

from neuralnetwork.layers.activation import ActivationLayer


class SigmoidLayer(ActivationLayer):
    def activation(self, x):
        return 1.0 / (1 + np.exp(-x))

    def activation_prime(self, x):
        activation = self.activation(x)
        return activation * (1 - activation)
