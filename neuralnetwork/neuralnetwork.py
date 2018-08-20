from batch import Batch
from cost import MSE

from layers.layer import Layer
from layers.linear import LinearLayer


class NeuralNetwork(object):
    layers = []

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def add_layer(self, layer):
        assert issubclass(layer.__class__, Layer)
        self.layers.append(layer)

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, gradients):
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)

        return gradients

    def train(self, inputs, labels, epochs=1000, loss=MSE()):
        batcher = Batch()
        for epoch in range(epochs):
            epoch_cost = 0
            for input_set, label_set in batcher(inputs, labels):
                output = self.forward(input_set)
                epoch_cost += MSE().cost(output, label_set)
                gradient = MSE().gradients(output, label_set)
                self.backward(gradient)

                for layer in self.layers:
                    if issubclass(layer.__class__, LinearLayer):
                        self.sgd(layer)
            print "Epoch: {} - Cost: {}".format(epoch, epoch_cost)

    def sgd(self, layer):
        layer.bias -= layer.gradients['bias'] * self.learning_rate
        layer.weights -= layer.gradients['weights'] * self.learning_rate
