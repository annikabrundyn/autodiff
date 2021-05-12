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

    def backward(self, prev_grad):
        return prev_grad.reshape(*self.old_shape)


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
        self.in_feat = in_features
        self.out_feat = out_features

        self._init_weights()

    def _init_weights(self, type="xavier"):
        if type == "xavier":
            bound = np.sqrt(1. / self.in_feat)
            self.W = {'val': np.random.randn(self.out_feat, self.in_feat) * bound, 'grad': 0}
            self.b = {'val': np.random.randn(1, self.out_feat) * bound, 'grad': 0}

        elif type == "random":
            self.W = {'val': np.random.randn(self.out_feat, self.in_feat), 'grad': 0}
            self.b = {'val': np.random.randn(1, self.out_feat), 'grad': 0}

    def forward(self, X):
        self.cache = np.copy(X)
        return np.dot(X, self.W['val'].T) + self.b['val']

    def backward(self, prev_grad):
        X = self.cache

        # Compute and store the gradient.
        batch_size = X.shape[0]
        self.W['grad'] = (1 / batch_size) * np.dot(prev_grad.T, X)
        self.b['grad'] = (1 / batch_size) * np.sum(prev_grad, axis=0)

        # Compute error.
        new_grad = np.dot(prev_grad, self.W['val'])
        return new_grad, self.W['grad'], self.b['grad']


class Conv2D(Layer):
    """
    2D Convolutional Layer (Optimized Im2Col Version)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0
                 ):
        """
        The Conv2D layer is inspired by the PyTorch API.
        The implementation is adapted from: https://github.com/3outeille/CNNumpy/blob/master/src/fast/layers.py

        Note:
            Applying the Im2Col Transformation trades memory for efficiency. We found the forward calculation of a
            Conv layer with the transformation to be roughly 100 times faster than without.

        Args:
            in_channels: number of channels in the input image
            out_channels: number of channels produced by the convolution
            kernel_size: size of the convolving kernel. assumed to be square (filter_size x filter_size)
            stride: stride of the convolution (default: 1).
            padding: zero-padding added to both sides of the input (default: 0).

        """
        super().__init__('Conv')
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding

        # Initialize the weights
        self._init_weights()

    def _init_weights(self, type="xavier"):
        if type == "xavier":
            bound = np.sqrt(1 / (self.kernel_size))
            self.W = {'val': np.random.randn(self.out_ch, self.in_ch, self.kernel_size, self.kernel_size) * bound,
                      'grad': np.zeros((self.out_ch, self.in_ch, self.kernel_size, self.kernel_size))}
            self.b = {'val': np.random.randn(self.out_ch) * bound,
                      'grad': np.zeros((self.out_ch))}

        elif type == "random":
            self.W = {'val': np.random.randn(self.out_ch, self.in_ch, self.kernel_size, self.kernel_size),
                      'grad': np.zeros((self.out_ch, self.in_ch, self.kernel_size, self.kernel_size))}
            self.b = {'val': np.random.randn(self.out_ch),
                      'grad': np.zeros((self.out_ch))}

    def forward(self, X):
        # batch_size, input channels, height, width of input matrix
        bs, ch_prev, h_prev, w_prev = X.shape

        # Calculate dimensions of output matrix
        ch_out = self.out_ch
        h_out = int((h_prev + 2 * self.pad - self.kernel_size) / self.stride) + 1
        w_out = int((w_prev + 2 * self.pad - self.kernel_size) / self.stride) + 1

        # Im2Col transformation
        X_col = im2col(X, self.kernel_size, self.kernel_size, self.stride, self.pad)
        w_col = self.W['val'].reshape((self.out_ch, -1))
        b_col = self.b['val'].reshape(-1, 1)

        # Perform matrix multiplication.
        out = w_col @ X_col + b_col

        # Reshape back matrix to image.
        out = np.array(np.hsplit(out, bs)).reshape((bs, ch_out, h_out, w_out))

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
        dX = col2im(dX_col, X.shape, self.kernel_size, self.kernel_size, self.stride, self.pad)

        # Reshape dw_col into dw.
        self.W['grad'] = dw_col.reshape((dw_col.shape[0], self.in_ch, self.kernel_size, self.kernel_size))

        return dX, self.W['grad'], self.b['grad']
