import numpy as np
from autodiff.layer import Layer


class Model:
    """
    Base model class
    """

    def __init__(self):
        self.layers = []
        self.loss = []

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def add(self, layer: Layer):
        # Add layer to sequential list of model layers
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Forward pass
        forward = None
        for i, _ in enumerate(self.layers):
            forward = self.layers[i].forward(X)
            X = forward
        return forward

    def backward(self, loss):
        deltaL = loss.backward()

        for i, layer in reversed(list(enumerate(self.layers))):
            if layer.type == "Linear" or layer.type == "Conv":
                deltaL, dW, db = self.layers[i].backward(deltaL)
            else:
                deltaL = self.layers[i].backward(deltaL)

    def update_params_sgd(self, lr):
        """
        Note: currently this only works with SGD
        """
        for i, layer in enumerate(self.layers):
            if layer.type == "Linear" or layer.type == "Conv":
                layer.W['val'] += -lr * layer.W['grad']
                layer.b['val'] += -lr * layer.b['grad']

    def zero_grad(self):
        for i, layer in enumerate(self.layers):
            if layer.type == "Conv":
                layer.W['grad'] = np.zeros((layer.n_F, layer.n_C, layer.f, layer.f))
                layer.b['grad'] = np.zeros((layer.n_F))
            elif layer.type == "Linear":
                layer.W['grad'] = 0
                layer.b['grad'] = 0
