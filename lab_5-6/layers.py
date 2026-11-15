import numpy as np


class Dense:
    def __init__(self, input_size, output_size, activator, weights=None):
        if weights is None:
            weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.W = weights
        self.b = np.zeros((1, output_size))
        self.activator = activator
        self.z = None
        self.a = None
        self.input = None

    def forward(self, x):
        self.input = x
        self.z = x @ self.W + self.b
        self.a = self.activator.forward(self.z)
        return self.a

    def backward(self, da):
        dz = self.activator.backward(da, self.z, self.a)
        dw = self.input.T @ dz
        db = np.sum(dz, axis=0, keepdims=True)
        da_prev = dz @ self.W.T
        return da_prev, {"W": dw, "b": db}

    def parameters(self):
        return {"W": self.W, "b": self.b}