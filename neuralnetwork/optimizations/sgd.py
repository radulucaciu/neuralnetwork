from optimizations.optimizations import Optimization


class Sgd(Optimization):
    def run(self, layer):
        layer.bias -= layer.gradients['bias'] * self.learning_rate
        layer.weights -= layer.gradients['weights'] * self.learning_rate
