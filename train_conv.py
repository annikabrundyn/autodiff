import autodiff as ad
import numpy as np
from sklearn.datasets import load_digits

#x = np.random.rand(2, 1, 28, 28)
#y = np.random.rand(2, 10)

x, y = load_digits(return_X_y=True)
x = x.reshape(x.shape[0], 1, 8, 8)

lr = 0.01

model = ad.Model()
model.add(ad.Conv2D(in_channels=1, out_channels=3, filter_size=3))
model.add(ad.ReLU())
model.add(ad.Conv2D(in_channels=3, out_channels=1, filter_size=3))
model.add(ad.Flatten())  #(2, 1, 4, 4)
model.add(ad.Linear(16, 10))
model.add(ad.Softmax())

criterion = ad.CategoricalCrossEntropy()

# forward
y_pred = model.forward(x)

loss = criterion.forward(y_pred, y)

# backprop
model.backward(criterion)

# update weights
model.update_params_sgd(lr=0.01)

print("hi")