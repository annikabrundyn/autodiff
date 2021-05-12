import numpy as np
from autodiff.layer import Layer


class Model:
    """
    Base model class.
    """
    def __init__(self):
        self.layers = []

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def add(self, layer: Layer):
        """
        Add layer to sequential list of model layers.

        Args:
            layer: specific layer to be added.

        """
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass to calculate the predictions given the input X.

        Args:
            X: input to the model

        Returns:
            pred: predicted value (output of the network)

        """
        pred = None
        for i, _ in enumerate(self.layers):
            pred = self.layers[i].forward(X)
            X = pred
        return pred

    def backward(self, loss):
        """
        Backprop to calculate the gradients for weight update.
        Calculated gradients are stored as attributes (with the parameter value) of the given layer.

        Args:
            loss: final loss layer

        """
        deltaL = loss.backward()

        for i, layer in reversed(list(enumerate(self.layers))):
            if layer.type == "Linear" or layer.type == "Conv":
                deltaL, dW, db = self.layers[i].backward(deltaL)
            else:
                deltaL = self.layers[i].backward(deltaL)

    def update_params_sgd(self, lr: float):
        """
        Update the parameter values using Gradient Descent.

        Note:
            In future we intend to generalize this to work with other optimization algorithms, such as Adam.

        Args:
            lr: learning rate

        """
        for i, layer in enumerate(self.layers):
            if layer.type == "Linear" or layer.type == "Conv":
                layer.W['val'] += -lr * layer.W['grad']
                layer.b['val'] += -lr * layer.b['grad']

    def zero_grad(self):
        """
        Zero the stored gradients (after weight update)
        """
        for i, layer in enumerate(self.layers):
            if layer.type == "Conv":
                layer.W['grad'] = np.zeros((layer.n_F, layer.n_C, layer.f, layer.f))
                layer.b['grad'] = np.zeros((layer.n_F))
            elif layer.type == "Linear":
                layer.W['grad'] = 0
                layer.b['grad'] = 0
