import numpy as np
from layer import Layer


class Model:

    def __init__(self):
        self.layers = []
        self.loss = []

    def add(self, layer: Layer):
        # Add layer to sequential list of model layers
        self.layers.append(layer)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Forward pass
        forward = None
        for i, _ in enumerate(self.layers):
            forward = self.layers[i].forward(X)
            X = forward
        return forward

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)



