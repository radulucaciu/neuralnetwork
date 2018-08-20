import numpy as np

from neuralnetwork.neuralnetwork import NeuralNetwork
from neuralnetwork.layers.linear import LinearLayer
# from neuralnetwork.layers.sigmoid import SigmoidLayer
from neuralnetwork.layers.tanh import TanhLayer


inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

labels = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

nn = NeuralNetwork()
nn.add_layer(LinearLayer(2, 2))
# nn.add_layer(SigmoidLayer())
nn.add_layer(TanhLayer())
nn.add_layer(LinearLayer(2, 2))
nn.train(inputs, labels, epochs=5000)


for index, row in enumerate(inputs):
    prediction = nn.forward(row)
    print row, prediction, labels[index]
