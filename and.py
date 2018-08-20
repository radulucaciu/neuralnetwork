import numpy as np

from neuralnetwork import NeuralNetwork
from layers.linear import LinearLayer


inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

""" Results for the AND function for each line from the inputs array
    The first column is False, the second Column is for True
"""
labels = np.array([
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
])

nn = NeuralNetwork()
nn.add_layer(LinearLayer(2, 2))
nn.train(inputs, labels, epochs=5000)


for index, row in enumerate(inputs):
    prediction = nn.forward(row)
    print row, prediction, labels[index]
