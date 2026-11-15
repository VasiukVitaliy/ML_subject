import numpy as np


class Relu:
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, da, z, a):
        return da * (z > 0)


class Sigmoid:
    def forward(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def backward(self, da, z, a):
        return da * a * (1 - a)


class SoftMax:
    def forward(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward(self, da, z, a):

        s = np.sum(da * a, axis=1, keepdims=True)
        return a * (da - s)