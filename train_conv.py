import autodiff as ad

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# Load the dataset
digits = load_digits()
n_samples = len(digits.data)
data = digits.images.reshape(n_samples, 1, 8, 8)

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, shuffle=False)

# Define the model
model = ad.Model()
model.add(ad.Conv2D(in_channels=1, out_channels=3, filter_size=3))
model.add(ad.ReLU())
model.add(ad.Conv2D(in_channels=3, out_channels=1, filter_size=3))
model.add(ad.Flatten())
model.add(ad.Linear(16, 10))

# Define the loss function - this already applies softmax to the output (like PyTorch)
criterion = ad.CategoricalCrossEntropy()

# Training Loop
losses = []
for epoch in range(500):
    # Forward pass
    y_pred = model.forward(x_train)

    # Compute loss
    loss = criterion.forward(y_pred, y_train)
    losses.append(loss)

    # Back propagation
    model.backward(criterion)

    # Update the parameters using Stochastic Gradient Descent
    model.update_params_sgd(lr=0.001)

    # zero the gradients
    model.zero_grad()

    if (epoch % 20) == 0:
        print(f"epoch: {epoch}, training loss: {loss}")






# See test set performance
pred_scores = model(x_test)
pred_labels = np.argmax(pred_scores, axis=1)
print("Test accuracy: ", (pred_labels == y_test).sum())