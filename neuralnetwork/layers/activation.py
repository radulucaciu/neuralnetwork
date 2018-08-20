from neuralnetwork.layers.layer import Layer


class ActivationLayer(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        return self.activation(inputs)

    def backward(self, gradients):
        return self.activation_prime(self.inputs) * gradients

    def activation(self, x):
        raise NotImplementedError

    def activation_prime(self, x):
        raise NotImplementedError
