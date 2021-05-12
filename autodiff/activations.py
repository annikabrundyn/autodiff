import numpy as np
from autodiff.layer import Layer


class ReLU(Layer):

    def __init__(self):
        super().__init__('ReLU')

    def forward(self, X):
        self.cache = np.copy(X)
        return np.clip(X, 0, None)

    def backward(self, new_deltaL):
        X = self.cache
        return np.where(X > 0, new_deltaL, 0)


class TanH(Layer):

    def __init__(self):
        super().__init__('TanH')

    def forward(self, X):
        self.cache = np.copy(X)
        return np.tanh(X)

    def backward(self, new_deltaL):
        X = self.cache
        return new_deltaL * (1 - np.square(np.tanh(X)))


class Sigmoid(Layer):

    def __init__(self):
        super().__init__('Sigmoid')

    def forward(self, X):
        self.cache = np.copy(X)
        return 1 / (1 + np.exp(-X))

    def backward(self, new_deltaL):
        X = self.cache
        return new_deltaL * X * (1 - X)
