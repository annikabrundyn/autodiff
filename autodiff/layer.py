import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class Layer(ABC):
    """abstract layer class"""

    def __init__(self, layer_type: str, output_dim: int):
        self.type = layer_type
        self.units = output_dim
        self._prev_val = None

    def __len__(self) -> int:
        return self.units

    def __str__(self):
        return f"{self.type} Layer"

    @abstractmethod
    def forward(self, *args, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, *args, **kwargs) -> np.ndarray:
        pass


class Linear(Layer):
    """
    Linear layer used in fully-connected network.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: number of  dimensions in the input
            output_dim: number of dimensions in the output
        """
        super().__init__('Linear', output_dim)
        self.weights = np.random.rand(output_dim, input_dim)
        self.biases = np.random.rand(output_dim, 1)

    def forward(self, input_val: np.ndarray) -> np.ndarray:
        """Performs forward pass of this layer.

        Args:
            input_val: value being input to the layer

        Returns:
            weights * inputs + biases

        """
        self._prev_val = input_val
        return np.matmul(self.weights, self._prev_val) + self.biases

    def backward(self, dJ: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Back propagation of this layer.

        Args:
            dJ : gradient of the next layer.

        Returns:
            delta : upcoming gradient, usually from an activation function.
            dW : weights gradient of this layer.
            dB : biases gradient of this layer.

        """
        dW = np.dot(dJ, self._prev_val.T)
        dB = dJ.mean(axis=1, keepdims=True)

        delta = np.dot(self.weights.T, dJ)

        return delta, dW, dB

    def optimize(self, dW: np.ndarray, dB: np.ndarray, learning_rate: float):
        """Optimizes. Updates the weights according to gradient descent.

        Note:
            For now, optimization can only be performed using gradient descent.

        Args:
            dW : Weights gradient.
            dB : Biases gradient.
            learning_rate: Learning rate of the gradient descent.

        """
        self.weights = self.weights - learning_rate * dW
        self.biases = self.biases - learning_rate * dB




class Conv2D(Layer):

    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: number of  dimensions in the input
            output_dim: number of dimensions in the output
        """
        super().__init__('Conv', output_dim)
        self.weights = np.random.rand(output_dim, input_dim)
        self.biases = np.random.rand(output_dim, 1)

    def forward(self, input_val: np.ndarray) -> np.ndarray:
        """Performs forward pass of this layer.

        Args:
            input_val: value being input to the layer

        Returns:
            weights * inputs + biases

        """
        self._prev_val = input_val
        return np.matmul(self.weights, self._prev_val) + self.biases

    def backward(self, dJ: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Back propagation of this layer.

        Args:
            dJ : gradient of the next layer.

        Returns:
            delta : upcoming gradient, usually from an activation function.
            dW : weights gradient of this layer.
            dB : biases gradient of this layer.

        """
        dW = np.dot(dJ, self._prev_val.T)
        dB = dJ.mean(axis=1, keepdims=True)

        delta = np.dot(self.weights.T, dJ)

        return delta, dW, dB

    def optimize(self, dW: np.ndarray, dB: np.ndarray, learning_rate: float):
        """Optimizes. Updates the weights according to gradient descent.

        Note:
            For now, optimization can only be performed using gradient descent.

        Args:
            dW : Weights gradient.
            dB : Biases gradient.
            learning_rate: Learning rate of the gradient descent.

        """
        self.weights = self.weights - learning_rate * dW
        self.biases = self.biases - learning_rate * dB






