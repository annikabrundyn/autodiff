from model import Model
from layer import *
from loss import BCE, MSE
from sklearn.datasets import load_iris

import numpy as np

X, Y = load_iris(return_X_y=True)

# our model takes in the shape (feats, samples)
X = X.T
# define the data - two features, single sample
#X = np.random.rand(2, 1000)
#Y = np.random.rand(1000)

losses = []

# define the model
model = Model()
model.add(Linear(4, 5))
model.add(ReLU(5))
model.add(Linear(5, 2))
model.add(ReLU(2))
model.add(Linear(2, 3))
model.add(Sigmoid(1))


for epoch in range(2):

    # forward
    pred = model(X)

    loss_f = MSE()

    error = loss_f(pred, Y)
    losses.append(error)

    gradient = loss_f.backward()

    # Backpropagation
    for i, _ in reversed(list(enumerate(model.layers))):
        if model.layers[i].type != 'Linear':
            gradient = model.layers[i].backward(gradient)
        else:
            gradient, dW, dB = model.layers[i].backward(gradient)
            model.layers[i].optimize(dW, dB, 0.1)

    print(error)