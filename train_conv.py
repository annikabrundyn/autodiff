import autodiff as ad
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer

x, y = load_digits(return_X_y=True)
x = x.reshape(x.shape[0], 1, 8, 8)

model = ad.Model()
model.add(ad.Conv2D(in_channels=1, out_channels=3, filter_size=3))
model.add(ad.ReLU())
model.add(ad.Conv2D(in_channels=3, out_channels=1, filter_size=3))
model.add(ad.Flatten())  #(2, 1, 4, 4)
model.add(ad.Linear(16, 10))
#model.add(ad.Softmax())

# define loss - like pytorch this already applies softmax then negative loglikelihood
criterion = ad.CategoricalCrossEntropy()

losses = []
for epoch in range(100):
    # forward
    y_pred = model.forward(x)

    loss = criterion.forward(y_pred, y)
    losses.append(loss)

    # backprop
    model.backward(criterion)

    # update weights
    model.update_params_sgd(lr=0.001)

    if (epoch % 10) == 0:
        print(f"epoch: {epoch}, current loss: {loss}")

print("hi")