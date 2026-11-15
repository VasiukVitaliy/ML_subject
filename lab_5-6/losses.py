import numpy as np


class CategoricalCrossEntropy:
    def forward(self, P, y_true):
        n = P.shape[0]
        p = np.clip(P[np.arange(n), y_true], 1e-12, 1.0)
        return -np.mean(np.log(p))

    def backward(self, P, y_true):
        n = P.shape[0]
        dP = np.zeros_like(P)
        p = np.clip(P[np.arange(n), y_true], 1e-12, 1.0)
        dP[np.arange(n), y_true] = -1.0 / (n * p)
        return dP