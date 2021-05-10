import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
import math


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

    # @abstractmethod
    # def backward(self, *args, **kwargs) -> np.ndarray:
    #     pass



class Flatten(Layer):
    """
    Flattens a contiguous range of dimensions. Used when going from Conv2D --> Linear Layer
    """
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__('Flatten', 1)
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, X):
        self.old_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, deltaL):
        return deltaL.reshape(*self.old_shape)








class Conv2D(Layer):

    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0):

        super().__init__('Conv', out_channels)
        self.n_F = out_channels
        self.f = filter_size
        self.n_C = in_channels
        self.s = stride
        self.p = padding

        # Xavier initialization.
        # TODO: should math be replaced with np
        bound = 1 / math.sqrt(self.f * self.f)
        self.W = {'val': np.random.uniform(-bound, bound, size=(self.n_F, self.n_C, self.f, self.f)),
                  'grad': np.zeros((self.n_F, self.n_C, self.f, self.f))}

        self.b = {'val': np.random.uniform(-bound, bound, size=(self.n_F)),
                  'grad': np.zeros((self.n_F))}

        self.cache = None

    def forward(self, X):
        self.cache = X
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        # Define output size.
        n_C = self.n_F
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        out = np.zeros((m, n_C, n_H, n_W))

        for i in range(m):  # For each image.

            for c in range(n_C):  # For each channel.

                for h in range(n_H):  # Slide the filter vertically.
                    h_start = h * self.s
                    h_end = h_start + self.f

                    for w in range(n_W):  # Slide the filter horizontally.
                        w_start = w * self.s
                        w_end = w_start + self.f

                        # Element wise multiplication + sum.
                        out[i, c, h, w] = np.sum(X[i, :, h_start:h_end, w_start:w_end]
                                                 * self.W['val'][c, ...]) + self.b['val'][c]
        return out

    def backward(self, dout):

        X = self.cache

        m, n_C, n_H, n_W = X.shape
        m, n_C_dout, n_H_dout, n_W_dout = dout.shape

        dX = np.zeros(X.shape)

        # Compute dW.
        for i in range(m):  # For each example.

            for c in range(n_C_dout):  # For each channel.

                for h in range(n_H_dout):  # Slide the filter vertically.
                    h_start = h * self.s
                    h_end = h_start + self.f

                    for w in range(n_W_dout):  # Slide the filter horizontally.
                        w_start = w * self.s
                        w_end = w_start + self.f

                        self.W['grad'][c, ...] += dout[i, c, h, w] * X[i, :, h_start:h_end, w_start:w_end]
                        dX[i, :, h_start:h_end, w_start:w_end] += dout[i, c, h, w] * self.W['val'][c, ...]
        # Compute db.
        for c in range(self.n_F):
            self.b['grad'][c, ...] = np.sum(dout[:, c, ...])

        return dX, self.W['grad'], self.b['grad']

    def optimize(self, dW: np.ndarray, dB: np.ndarray, learning_rate: float):

        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * dB



class Linear(Layer):

    def __init__(self, column, row):
        super().__init__("FC", 1)
        self.row = row
        self.col = column

        # Xavier-Glorot initialization - used for sigmoid, tanh.
        self.W = {'val': np.random.randn(self.row, self.col) * np.sqrt(1. / self.col), 'grad': 0}
        self.b = {'val': np.random.randn(1, self.row) * np.sqrt(1. / self.row), 'grad': 0}

        self.cache = None

    def forward(self, fc):
        """
            Performs a forward propagation between 2 fully connected layers.
            Parameters:
            - fc: fully connected layer.

            Returns:
            - A_fc: new fully connected layer.
        """
        self.cache = fc
        A_fc = np.dot(fc, self.W['val'].T) + self.b['val']
        return A_fc

    def backward(self, deltaL):
        """
            Returns the error of the current layer and compute gradients.
            Parameters:
            - deltaL: error at last layer.

            Returns:
            - new_deltaL: error at current layer.
            - self.W['grad']: weights gradient.
            - self.b['grad']: bias gradient.
        """
        fc = self.cache
        m = fc.shape[0]

        # Compute gradient.
        self.W['grad'] = (1 / m) * np.dot(deltaL.T, fc)
        self.b['grad'] = (1 / m) * np.sum(deltaL, axis=0)

        # Compute error.
        new_deltaL = np.dot(deltaL, self.W['val'])
        # We still need to multiply new_deltaL by the derivative of the activation
        # function which is done in TanH.backward().
        return new_deltaL, self.W['grad'], self.b['grad']




# x = np.random.rand(2, 1, 10, 10)
#
# conv1 = Conv2D(nb_filters=6, filter_size=5, nb_channels=1)
# conv2 = Conv2D(nb_filters=1, filter_size=3, nb_channels=6)
#
# y1 = conv1.forward(x)
# y2 = conv2.forward(y1)
#
# deltaL2 = np.random.rand(*y2.shape)
# deltaL1, dw2, db2 = conv2.backward(deltaL2)
#
# deltaL, dw, db = conv1.backward(deltaL1)
#
# print("hi")



# class Linear(Layer):
#     """
#     Linear layer used in fully-connected network.
#     """
#
#     def __init__(self, input_dim: int, output_dim: int):
#         """
#         Args:
#             input_dim: number of  dimensions in the input
#             output_dim: number of dimensions in the output
#         """
#         super().__init__('Linear', output_dim)
#         self.weights = np.random.rand(output_dim, input_dim)
#         self.biases = np.random.rand(output_dim, 1)
#
#     def forward(self, input_val: np.ndarray) -> np.ndarray:
#         """Performs forward pass of this layer.
#
#         Args:
#             input_val: value being input to the layer
#
#         Returns:
#             weights * inputs + biases
#
#         """
#         self._prev_val = input_val
#         return np.matmul(self.weights, self._prev_val) + self.biases
#
#     def backward(self, dJ: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """Back propagation of this layer.
#
#         Args:
#             dJ : gradient of the next layer.
#
#         Returns:
#             delta : upcoming gradient, usually from an activation function.
#             dW : weights gradient of this layer.
#             dB : biases gradient of this layer.
#
#         """
#         dW = np.dot(dJ, self._prev_val.T)
#         dB = dJ.mean(axis=1, keepdims=True)
#
#         delta = np.dot(self.weights.T, dJ)
#
#         return delta, dW, dB
#
#     def optimize(self, dW: np.ndarray, dB: np.ndarray, learning_rate: float):
#         """Optimizes. Updates the weights according to gradient descent.
#
#         Note:
#             For now, optimization can only be performed using gradient descent.
#
#         Args:
#             dW : Weights gradient.
#             dB : Biases gradient.
#             learning_rate: Learning rate of the gradient descent.
#
#         """
#         self.weights = self.weights - learning_rate * dW
#         self.biases = self.biases - learning_rate * dB





