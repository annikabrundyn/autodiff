import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class Layer(ABC):
    """abstract class of Layer"""

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
    """Linear Layer.

    Equivalent to Dense in Keras and to torch.nn.Linear in torch.

    Parameters
    ----------
    input_dim : int
        Number of input features of this layer.
    output_dim : int
        Number of output features of this layer.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__('Linear', output_dim)
        self.weights = np.random.rand(output_dim, input_dim)
        self.biases = np.random.rand(output_dim, 1)

    def forward(self, input_val: np.ndarray) -> np.ndarray:
        """Forward.

        Performs forward propagation of this layer.

        Parameters:
        input_val : numpy.Array
            Forward propagation of the previous layer.
        Returns
        -------
        activation : numpy.Array
            Forward propagation operation of the linear layer.
        """
        self._prev_val = input_val
        return np.matmul(self.weights, self._prev_val) + self.biases

    def backward(self, dJ: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward.

        Computes backward propagation pass of this layer.

        Parameters
        ----------
        dJ : numpy.Array
            Gradient of the next layer.
        Returns
        -------
        delta : numpy.Array
            Upcoming gradient, usually from an activation function.
        dW : numpy.Array
            Weights gradient of this layer.
        dB : numpy.Array
            Biases gradient of this layer.
        """
        dW = np.dot(dJ, self._prev_val.T)
        dB = dJ.mean(axis=1, keepdims=True)

        delta = np.dot(self.weights.T, dJ)

        return delta, dW, dB

    def optimize(self, dW: np.ndarray, dB: np.ndarray, rate: float):
        """Optimizes.

        Performs the optimization of the parameters. For now,
        optimization can only be performed using gradient descent.

        Parameters
        ----------
        dW : numpy.Array
            Weights gradient.
        dB : numpy.Array
            Biases gradient.
        rate: float
            Learning rate of the gradient descent.
        """
        self.weights = self.weights - rate * dW
        self.biases = self.biases - rate * dB


class ReLU(Layer):
    """ReLU Layer.

    Parameters
    ----------
    output_dim : int
        Number of neurons in this layer.
    """

    def __init__(self, output_dim: int):
        super().__init__('ReLU', output_dim)

    def forward(self, input_val: np.ndarray) -> np.ndarray:
        """Forward.

        Computes forward propagation pass of this layer.

        Parameters
        ----------
        input_val : numpy.Array
            Forward propagation of the previous layer.
        Returns
        -------
        activation : numpy.Array
            Forward propagation of this layer.
        """
        self._prev_val = np.maximum(0, input_val)
        return self._prev_val

    def backward(self, dJ: np.ndarray) -> np.ndarray:
        """Backward.

        Computes backward propagation pass of this layer.

        Parameters
        ----------
        dJ : numpy.Array
            Gradient of the next layer.
        Returns
        -------
        delta : numpy.Array
            Upcoming gradient.
        """
        return dJ * np.heaviside(self._prev_val, 0)


class Sigmoid(Layer):
    """Sigmoid Layer.

    Parameters
    ----------
    output_dim : int
        Number of neurons in this layers.
    """

    def __init__(self, output_dim: int):
        super().__init__('Sigmoid', output_dim)

    def forward(self, input_val: np.ndarray) -> np.ndarray:
        """Forward.

        Computes forward propagation pass of this layer.

        Parameters
        ----------
        input_val : numpy.Array
            Forward propagation of the previous layer.
        Returns
        -------
        activation : numpy.Array
            Forward propagation of this layer.
        """
        self._prev_val = 1 / (1 + np.exp(-input_val))
        return self._prev_val

    def backward(self, dJ: np.ndarray):
        """Backward.

        Computes backward propagation pass of this layer.

        Parameters
        -------
        dJ : numpy.Array
            Gradient of this layer.
        Returns
        -------
        delta : numpy.Array
            Upcoming gradient.
        """
        sig = self._prev_val
        return dJ * sig * (1 - sig)


class Tanh(Layer):
    """Tanh.

    Hyperbolic tangent layer.

    Parameters
    ----------
    output_dim : int
        Number of neurons in this layers.
    """

    def __init__(self, output_dim: int):
        super().__init__('Tanh', output_dim)

    def forward(self, input_val: np.ndarray) -> np.ndarray:
        """Forward.

        Computes forward propagation pass of this layer.

        Parameters
        ----------
        input_val : numpy.Array
            Forward propagation of the previous layer.
        Returns
        -------
        activation : numpy.Array
            Forward propagation of this layer.
        """
        self._prev_val = np.tanh(input_val)
        return self._prev_val

    def backward(self, dJ: np.ndarray) -> np.ndarray:
        """Backward.

        Computes backward propagation pass of this layer.

        Parameters
        ----------
        dJ : numpy.Array
            Gradient of the next layer.
        Returns
        -------
        delta : numpy.Array
        """
        return dJ * (1 - np.square(self._prev_val))
