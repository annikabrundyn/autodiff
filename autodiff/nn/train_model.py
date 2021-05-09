from model import Model
from layer import *
from loss import BCE, MSE
from data import generate_data
from sklearn.datasets import load_iris, load_boston, load_breast_cancer
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np


X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = preprocessing.MinMaxScaler().fit(X_train)
X = scaler.transform(X_train)
X_test = scaler.transform(X_test)

Y = y_train


losses = []

# define the model
model = Model()
model.add(Linear(2, 5))
model.add(ReLU(5))
model.add(Linear(5, 2))
model.add(ReLU(2))
model.add(Linear(2, 1))
model.add(Sigmoid(1))


for epoch in range(1000000):

    # forward
    # our model takes in the shape (feats, samples)
    pred = model(X.T)

    loss_f = BCE(pred, Y)

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
            model.layers[i].optimize(dW, dB, 0.05)

    if epoch % 100:
        print("current loss: ", error)

