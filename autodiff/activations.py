import numpy as np
from layer import *


class ReLU(Layer):

    def __init__(self):
        super().__init__('ReLU', 1)
        self.cache = None

    def forward(self, X):
        self.cache = X
        return np.maximum(X, 0, X)

    def backward(self, new_deltaL):
        """
            Finishes computation of error by multiplying new_deltaL by the
            derivative of tanH.
            Parameters:
            - new_deltaL: error previously computed.
        """
        X = self.cache
        return new_deltaL * np.where(X <= 0, 0.0, 1.0)


class TanH(Layer):

    def __init__(self):
        super().__init__('TanH', 1)
        self.cache = None

    def forward(self, X):
        self.cache = X
        return np.tanh(X)

    def backward(self, new_deltaL):
        X = self.cache
        return new_deltaL * (1 - np.square(np.tanh(X)))


class Sigmoid(Layer):

    def __init__(self):
        super().__init__('Sigmoid', 1)
        self.cache = None

    def forward(self, X):
        self.cache = X
        return 1 / (1 + np.exp(-X))

    def backward(self, new_deltaL):
        X = self.cache
        return new_deltaL * X * (1 - X)


### softmax is a WIP
# class Softmax(Layer):
#
#     def __init__(self, output_dim):
#         super().__init__('Softmax', output_dim)
#
#     def forward(self, X):
#         """
#             Compute softmax values for each sets of scores in X.
#             Parameters:
#             - X: input vector.
#             # For numerical stability: make the maximum of z's to be 0.
#         """
#         shift_x = X - np.max(X)
#         e_x = np.exp(shift_x)
#         return e_x / np.sum(e_x, axis=1)[:, np.newaxis]
#
#     def backward(self, y_pred, y):
#         return y_pred - y