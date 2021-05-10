import autodiff as ad
import numpy as np
from sklearn.datasets import make_classification


# Make dataset
X = np.random.rand(100, 3)
Y = np.random.rand(100, 1)


# Define the model
model = ad.Model()
model.add(ad.Linear(3, 10))
model.add(ad.ReLU())
model.add(ad.Linear(10, 5))
model.add(ad.ReLU())
model.add(ad.Linear(5, 1))
model.add(ad.Sigmoid())

# Define the criterion
loss_f = ad.BCE()

# Train the model
losses = []

for epoch in range(1000):

    # forward - our model takes input with shape (feats, samples)
    pred = model(X)

    error = loss_f(pred, Y)
    losses.append(error)

    # Backprop
    model.backward(loss_f)

    # update the weights using SGD
    model.update_params_sgd(lr=0.05)

    if (epoch % 100) == 0:
        print("current loss: ", error)

