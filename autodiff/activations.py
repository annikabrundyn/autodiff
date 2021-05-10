import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from layer import *


class ReLU(Layer):
    """
    ReLU non-linearity.
    """

    def __init__(self, output_dim: int):
        """
        Args:
            output_dim: number of neurons
        """
        super().__init__('ReLU', output_dim)

    def forward(self, input_val: np.ndarray) -> np.ndarray:
        """Forward.

        Args:
            input_val: Forward propagation of the previous layer.

        Returns:
            activation: Forward propagation of this layer.

        """
        self._prev_val = np.maximum(0, input_val)
        return self._prev_val

    def backward(self, dJ: np.ndarray) -> np.ndarray:
        """Backward pass.

        Args:
            dJ: Gradient of the next layer.

        Returns:
            delta: Upcoming gradient.

        """
        #return dJ * np.heaviside(self._prev_val, 0)
        return dJ * np.where(self._prev_val <= 0, 0.0, 1.0)


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


class Tanh(Layer):
    """Tanh.
    """

    def __init__(self, output_dim: int):
        """
        Args:
            output_dim: number of neurons
        """
        super().__init__('Tanh', output_dim)

    def forward(self, input_val: np.ndarray) -> np.ndarray:
        """Forward.

        Args:
            input_val: Forward propagation of the previous layer.

        Returns:
            activation: Forward propagation of this layer.
        """
        self._prev_val = np.tanh(input_val)
        return self._prev_val

    def backward(self, dJ: np.ndarray) -> np.ndarray:
        """Backward.

        Args:
            dJ: Gradient of the next layer.

        Returns:
            delta : numpy.Array
        """
        return dJ * (1 - np.square(self._prev_val))
