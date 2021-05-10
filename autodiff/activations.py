import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from layer import *


# class ReLU(Layer):
#     """
#     ReLU non-linearity.
#     """
#
#     def __init__(self, output_dim: int):
#         """
#         Args:
#             output_dim: number of neurons
#         """
#         super().__init__('ReLU', output_dim)
#
#     def forward(self, input_val: np.ndarray) -> np.ndarray:
#         """Forward.
#
#         Args:
#             input_val: Forward propagation of the previous layer.
#
#         Returns:
#             activation: Forward propagation of this layer.
#
#         """
#         self._prev_val = np.maximum(0, input_val)
#         return self._prev_val
#
#     def backward(self, dJ: np.ndarray) -> np.ndarray:
#         """Backward pass.
#
#         Args:
#             dJ: Gradient of the next layer.
#
#         Returns:
#             delta: Upcoming gradient.
#
#         """
#         #return dJ * np.heaviside(self._prev_val, 0)
#         return dJ * np.where(self._prev_val <= 0, 0.0, 1.0)


class Sigmoid(Layer):
    """
    Sigmoid non-linearity.
    """

    def __init__(self, output_dim: int):
        """
        Args:
            output_dim: Number of neurons in this layers
        """
        super().__init__('Sigmoid', output_dim)

    def forward(self, input_val: np.ndarray) -> np.ndarray:
        """Forward.

        Args:
            input_val: Forward propagation of the previous layer.

        Returns:
            activation: Forward propagation of this layer.

        """
        self._prev_val = 1 / (1 + np.exp(-input_val))
        return self._prev_val

    def backward(self, dJ: np.ndarray):
        """Backward.

        Args:
            dJ: Gradient of this layer.

        Returns:
            delta: Upcoming gradient.
        """
        sig = self._prev_val
        return dJ * sig * (1 - sig)


class Softmax(Layer):

    def __init__(self, output_dim):
        super().__init__('Softmax', output_dim)

    def forward(self, X):
        """
            Compute softmax values for each sets of scores in X.
            Parameters:
            - X: input vector.
            # For numerical stability: make the maximum of z's to be 0.
        """
        shift_x = X - np.max(X)
        e_x = np.exp(shift_x)
        return e_x / np.sum(e_x, axis=1)[:, np.newaxis]

    def backward(self, y_pred, y):
        return y_pred - y


class TanH(Layer):

    def __init__(self):
        super().__init__('TanH', 1)
        self.cache = None

    def forward(self, X):
        self.cache = X
        return np.tanh(X)

    def backward(self, new_deltaL):
        """
            Finishes computation of error by multiplying new_deltaL by the
            derivative of tanH.
            Parameters:
            - new_deltaL: error previously computed.
        """
        X = self.cache
        return new_deltaL * (1 - np.square(np.tanh(X)))



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