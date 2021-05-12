import numpy as np
from autodiff.layer import Layer


class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) non-linearity.
    """

    def __init__(self):
        super().__init__('ReLU')

    def forward(self, X):
        self.cache = np.copy(X)
        return np.clip(X, 0, None)

    def backward(self, deltaL):
        X = self.cache
        return np.where(X > 0, deltaL, 0)


class TanH(Layer):
    """
    Hyperbolic Tangent non-linearity.
    """

    def __init__(self):
        super().__init__('TanH')

    def forward(self, X):
        self.cache = np.copy(X)
        return np.tanh(X)

    def backward(self, deltaL):
        X = self.cache
        return deltaL * (1 - np.square(np.tanh(X)))


class Sigmoid(Layer):
    """
    Sigmoid non-linearity.
    """

    def __init__(self):
        super().__init__('Sigmoid')

    def forward(self, X):
        self.cache = np.copy(X)
        return 1 / (1 + np.exp(-X))

    def backward(self, deltaL):
        X = self.cache
        return deltaL * X * (1 - X)
