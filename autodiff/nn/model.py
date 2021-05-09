class Model:

    def __init__(self):
        self.layers = []
        self.loss = []

    def add(self, layer):
        # Add layer to sequential list of model layers
        self.layers.append(layer)

    def predict(self, X):
        # Forward pass
        for i, _ in enumerate(self.layers):
            forward = self.layers[i].forward(X)
            X = forward
        return forward

    def __call__(self, X):
        return self.predict(X)



