class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, parameters, gradients):
        for name in parameters:
            parameters[name] -= (
                self.lr * gradients[name]
            )