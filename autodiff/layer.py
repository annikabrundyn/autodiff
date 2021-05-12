import numpy as np
from abc import ABC, abstractmethod

from autodiff.utils import im2col, col2im


class Layer(ABC):
    """abstract layer class"""

    def __init__(self, layer_type: str):
        self.type = layer_type
        self.cache = None

    def __str__(self):
        return f"{self.type} Layer"

    @abstractmethod
    def forward(self, *args, **kwargs) -> np.ndarray:
        pass


class Flatten(Layer):
    """
    Flattens a contiguous range of dimensions. Used when going from Conv2D --> Linear Layer
    """
    def __init__(self, start_dim=1, end_dim=-1):
        """

        Args:
            start_dim: first dim to flatten (default: 1).
            end_dim: last dim to flatten (default: -1).
        """
        super().__init__('Flatten')
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, X):
        self.old_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, deltaL):
        return deltaL.reshape(*self.old_shape)


class Linear(Layer):
    """
    Fully connected (linear) layer. Similar to PyTorch Linear Layer.
    """
    def __init__(self, in_features, out_features):
        """
        Applies a linear transformation to the incoming data: y = xA^{T} + b.
        This implementation is adapted from: https://github.com/3outeille/CNNumpy/blob/master/src/fast/layers.py

        Note:
            By default, it learns an additive bias. In future, we plan to generalize this.

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
        """
        super().__init__("Linear")
        self.col = in_features
        self.row = out_features

        self._init_weights()

    def _init_weights(self, type="xavier"):
        if type == "xavier":
            bound = np.sqrt(1. / self.col)
            self.W = {'val': np.random.randn(self.row, self.col) * bound, 'grad': 0}
            self.b = {'val': np.random.randn(1, self.row) * bound, 'grad': 0}

        elif type == "random":
            self.W = {'val': np.random.randn(self.row, self.col), 'grad': 0}
            self.b = {'val': np.random.randn(1, self.row), 'grad': 0}

    def forward(self, X):
        self.cache = X
        return np.dot(X, self.W['val'].T) + self.b['val']

    def backward(self, deltaL):

        X = self.cache
        m = X.shape[0]

        # Compute and store the gradient.
        self.W['grad'] = (1 / m) * np.dot(deltaL.T, X)
        self.b['grad'] = (1 / m) * np.sum(deltaL, axis=0)

        # Compute error.
        new_deltaL = np.dot(deltaL, self.W['val'])
        return new_deltaL, self.W['grad'], self.b['grad']


class Conv2D(Layer):
    """
    2D Convolutional Layer (Optimized Im2Col Version)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 filter_size: int,
                 stride: int = 1,
                 padding: int = 0
                 ):
        """
        The Conv2D layer is inspired by the PyTorch API.
        The implementation is adapted from: https://github.com/3outeille/CNNumpy/blob/master/src/fast/layers.py

        Args:
            in_channels: number of channels in the input image
            out_channels: number of channels produced by the convolution
            filter_size: size of the convolving kernel. assumed to be square (filter_size x filter_size)
            stride: stride of the convolution (default: 1).
            padding: zero-padding added to both sides of the input (default: 0).

        """
        super().__init__('Conv')
        self.n_C = in_channels
        self.n_F = out_channels
        self.f = filter_size
        self.s = stride
        self.p = padding

        # Initialize the weights
        self._init_weights()

    def _init_weights(self, type="xavier"):
        if type == "xavier":
            bound = np.sqrt(1 / (self.f))
            self.W = {'val': np.random.randn(self.n_F, self.n_C, self.f, self.f) * bound,
                      'grad': np.zeros((self.n_F, self.n_C, self.f, self.f))}
            self.b = {'val': np.random.randn(self.n_F) * bound,
                      'grad': np.zeros((self.n_F))}

        elif type == "random":
            self.W = {'val': np.random.randn(self.n_F, self.n_C, self.f, self.f),
                      'grad': np.zeros((self.n_F, self.n_C, self.f, self.f))}
            self.b = {'val': np.random.randn(self.n_F),
                      'grad': np.zeros((self.n_F))}

    def forward(self, X):
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        # Calculate dimensions of output matrix
        n_C = self.n_F
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        # Im2Col transformation
        X_col = im2col(X, self.f, self.f, self.s, self.p)
        w_col = self.W['val'].reshape((self.n_F, -1))
        b_col = self.b['val'].reshape(-1, 1)

        # Perform matrix multiplication.
        out = w_col @ X_col + b_col

        # Reshape back matrix to image.
        out = np.array(np.hsplit(out, m)).reshape((m, n_C, n_H, n_W))

        self.cache = X, X_col, w_col

        return out

    def backward(self, dout):
        X, X_col, w_col = self.cache
        m, _, _, _ = X.shape

        # Compute bias gradient.
        self.b['grad'] = np.sum(dout, axis=(0, 2, 3))

        # Reshape dout properly.
        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, m))
        dout = np.concatenate(dout, axis=-1)

        # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
        dX_col = w_col.T @ dout

        # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
        dw_col = dout @ X_col.T

        # Reshape back to image (col2im).
        dX = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)

        # Reshape dw_col into dw.
        self.W['grad'] = dw_col.reshape((dw_col.shape[0], self.n_C, self.f, self.f))

        return dX, self.W['grad'], self.b['grad']


# class Conv2D(Layer):
#     """
#     Original non-optimized version
#     """
#     def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0):
#         super().__init__('Conv', out_channels)
#         self.n_C = in_channels
#         self.n_F = out_channels
#         self.f = filter_size
#         self.s = stride
#         self.p = padding
#
#         self._init_weights()
#
#     def _init_weights(self, type="xavier"):
#         if type == "xavier":
#             bound = 1 / np.sqrt(self.f * self.f)
#             self.W = {'val': np.random.uniform(-bound, bound, size=(self.n_F, self.n_C, self.f, self.f)),
#                       'grad': np.zeros((self.n_F, self.n_C, self.f, self.f))}
#
#             self.b = {'val': np.random.uniform(-bound, bound, size=(self.n_F)),
#                       'grad': np.zeros((self.n_F))}
#
#         elif type == "random":
#             self.W = {'val': np.random.randn(self.n_F, self.n_C, self.f, self.f),
#                       'grad': np.zeros((self.n_F, self.n_C, self.f, self.f))}
#             self.b = {'val': np.random.randn(self.n_F),
#                       'grad': np.zeros((self.n_F))}
#
#     def forward(self, X):
#         self.cache = X
#         m, n_C_prev, n_H_prev, n_W_prev = X.shape
#
#         # Define output size.
#         n_C = self.n_F
#         n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
#         n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1
#
#         out = np.zeros((m, n_C, n_H, n_W))
#
#         for i in range(m):  # For each image.
#
#             for c in range(n_C):  # For each channel.
#
#                 for h in range(n_H):  # Slide the filter vertically.
#                     h_start = h * self.s
#                     h_end = h_start + self.f
#
#                     for w in range(n_W):  # Slide the filter horizontally.
#                         w_start = w * self.s
#                         w_end = w_start + self.f
#
#                         # Element wise multiplication + sum.
#                         out[i, c, h, w] = np.sum(X[i, :, h_start:h_end, w_start:w_end]
#                                                  * self.W['val'][c, ...]) + self.b['val'][c]
#         return out
#
#     def backward(self, dout):
#
#         X = self.cache
#
#         m, n_C, n_H, n_W = X.shape
#         m, n_C_dout, n_H_dout, n_W_dout = dout.shape
#
#         dX = np.zeros(X.shape)
#
#         # Compute dW.
#         for i in range(m):  # For each example.
#
#             for c in range(n_C_dout):  # For each channel.
#
#                 for h in range(n_H_dout):  # Slide the filter vertically.
#                     h_start = h * self.s
#                     h_end = h_start + self.f
#
#                     for w in range(n_W_dout):  # Slide the filter horizontally.
#                         w_start = w * self.s
#                         w_end = w_start + self.f
#
#                         self.W['grad'][c, ...] += dout[i, c, h, w] * X[i, :, h_start:h_end, w_start:w_end]
#                         dX[i, :, h_start:h_end, w_start:w_end] += dout[i, c, h, w] * self.W['val'][c, ...]
#         # Compute db.
#         for c in range(self.n_F):
#             self.b['grad'][c, ...] = np.sum(dout[:, c, ...])
#
#         return dX, self.W['grad'], self.b['grad']
