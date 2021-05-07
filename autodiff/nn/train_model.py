from model import Model
from layer import *
from loss import BCE, MSE
from sklearn.datasets import load_iris, load_boston, load_breast_cancer

import numpy as np

X, Y = load_breast_cancer(return_X_y=True)

# our model takes in the shape (feats, samples)
X = X.T
Y = np.reshape(Y, (1, len(Y)))
# define the data - two features, single sample
#X = np.random.rand(2, 1000)
#Y = np.random.rand(1000)

losses = []

# define the model
model = Model()
model.add(Linear(30, 5))
model.add(ReLU(5))
model.add(Linear(5, 2))
model.add(ReLU(2))
model.add(Linear(2, 1))
model.add(Sigmoid(1))


for epoch in range(100000):

    # forward
    pred = model(X)

    loss_f = MSE(pred, Y)

    error = loss_f()
    losses.append(error)

    gradient = loss_f.backward()

    # Backpropagation
    for i, _ in reversed(list(enumerate(model.layers))):
        if model.layers[i].type != 'Linear':
            gradient = model.layers[i].backward(gradient)
        else:
            gradient, dW, dB = model.layers[i].backward(gradient)
            #self.weights = self.weights - rate * dW
            #self.biases = self.biases - rate * dB
            model.layers[i].optimize(dW, dB, 0.01)

    print(error)

preds = model(X)

print("hi")