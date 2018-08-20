class Layer(object):
    weights = []
    bias = []
    gradients = {}

    def __init__(self):
        self.weights = []
        self.bias = []
        self.gradients = {}

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, gradients):
        raise NotImplementedError

    def function(self, inputs):
        raise NotImplementedError
