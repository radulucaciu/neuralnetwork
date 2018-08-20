class Optimization(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def run(self, layer):
        raise NotImplementedError
