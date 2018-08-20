import numpy as np

from neuralnetwork.layers.layer import Layer


class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, gradients):
        # y = f(g(x))
        # g(x) = w * x +b
        # df / dw = x * f'(x)
        # df / dx = w * f'(x)
        # df / db = f'(x))
        self.gradients['weights'] = np.dot(self.inputs.T, gradients)
        self.gradients['bias'] = np.sum(gradients, axis=0)
        return np.dot(gradients, self.weights.T)
