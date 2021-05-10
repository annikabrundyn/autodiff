import numpy as np
from layer import Layer


class Model:

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

    # def update_weights(self, loss_f, lr):
    #     gradient = loss_f.backward()
    #     # backprop and weight update
    #     for i, _ in reversed(list(enumerate(self.layers))):
    #         if self.layers[i].type != 'Linear':
    #             gradient = self.layers[i].backward(gradient)
    #         else:
    #             gradient, dW, dB = self.layers[i].backward(gradient)
    #             self.layers[i].optimize(dW, dB, lr)

    def update_params_sgd(self, lr):
        """
        Note: currently this only works with SGD - future todo is make this work with Adam optimizer for example
        """
        for i, layer in enumerate(self.layers):
            if layer.type == "Linear" or layer.type == "Conv":
                layer.W['val'] += -lr * layer.W['grad']
                layer.b['val'] += -lr * layer.b['grad']





